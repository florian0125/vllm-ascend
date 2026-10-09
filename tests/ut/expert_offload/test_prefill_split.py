# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the expert-offload prefill split-GMM row partition.

The split is exercised against a fake GMM whose per-expert "weights" are
distinct scalars, so any row routed through the wrong weight tensor (or any
row lost/duplicated by the partition/reassembly) changes the output and fails
the comparison against a single-GMM reference.  Runs on CPU tensors; the
forward-context probe used by prefill_split_apply is stubbed out.
"""

import types

import pytest
import torch

from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.expert_offload.prefill_split import (
    PrefillSplitCtx,
    _group_counts,
    prefill_split_apply,
)


@pytest.fixture(autouse=True)
def _stub_forward_context(monkeypatch):
    """prefill_split_apply probes _EXTRA_CTX.capturing; satisfy the proxy
    without a real vLLM forward context."""
    import vllm_ascend.ascend_forward_context as afc

    monkeypatch.setattr(
        afc, "get_forward_context",
        lambda: types.SimpleNamespace(capturing=False))


NTOTAL = 256
NSLOTS = 32
HIDDEN = 4


def make_inputs(n_rows, resident_slots, seed):
    """Dispatch-like inputs: rows sorted by expert id with a count-type
    group_list, plus per-row dynamic scales."""
    g = torch.Generator().manual_seed(seed)
    eids = torch.randint(0, NTOTAL, (n_rows,), generator=g)
    eids, _ = eids.sort()  # dispatch output order
    group_list = _group_counts(eids, NTOTAL)

    # hidden value uniquely identifies the row position
    hidden = torch.arange(n_rows, dtype=torch.float32).reshape(-1, 1).expand(
        n_rows, HIDDEN).contiguous()
    dynamic_scale = torch.randn(n_rows, 2, generator=g)

    resident = resident_slots[eids] >= 0
    return hidden, dynamic_scale, group_list, eids, resident


def fake_apply_mlp(mlp_in):
    """Scalar-weight GMM: out_row = w[expert(row)] * x_row * scale_row."""
    x = mlp_in.hidden_states.to(torch.float32)  # fp8 inputs: cast for math
    gl = mlp_in.group_list.tolist()
    w1 = mlp_in.weights.w1  # [E, HIDDEN] or [E, 1]
    ds = mlp_in.dynamic_scale
    if ds is not None:
        ds = ds.to(torch.float32)
    outs = []
    start = 0
    for eid, cnt in enumerate(gl):
        cnt = int(cnt)
        if cnt == 0:
            continue
        rows = x[start:start + cnt]
        w = w1[eid].reshape(1, -1)
        chunk = rows * w
        if ds is not None:
            chunk = chunk * ds[start:start + cnt, :1]
        outs.append(chunk)
        start += cnt
    return torch.cat(outs), None


def reference_output(hidden, dynamic_scale, eids, resident_slots,
                     decode_w, pool_w):
    w = torch.where(
        (resident_slots[eids] >= 0).unsqueeze(1),
        decode_w[resident_slots[eids].clamp(min=0)],
        pool_w[eids],
    )
    return hidden * w * dynamic_scale[:, :1]


def run_split(hidden, dynamic_scale, group_list, resident_slots,
              decode_w, pool_w):
    mlp_in = MoEMlpComputeInput(
        hidden_states=hidden,
        group_list=group_list,
        group_list_type=1,
        dynamic_scale=dynamic_scale,
        topk_scales=None,
        weights=MoEWeights(w1=pool_w, w2=pool_w),
        quant=types.SimpleNamespace(is_quant=False),
        fusion=False,
    )
    ctx = PrefillSplitCtx(
        resident_slots=resident_slots,
        decode_weights=MoEWeights(w1=decode_w, w2=decode_w),
        num_slots=NSLOTS,
        layer_idx=0,
    )
    return prefill_split_apply(mlp_in, ctx, fake_apply_mlp)


@pytest.mark.parametrize("seed", list(range(8)))
def test_mixed_residency_matches_single_gmm(seed):
    torch.manual_seed(seed)
    resident_slots = torch.full((NTOTAL,), -1, dtype=torch.int64)
    resident_slots[torch.randperm(NTOTAL)[:NSLOTS]] = torch.randperm(NSLOTS)
    decode_w = 100.0 + torch.arange(NSLOTS, dtype=torch.float32).reshape(
        -1, 1).expand(NSLOTS, HIDDEN).contiguous()
    pool_w = torch.arange(NTOTAL, dtype=torch.float32).reshape(
        -1, 1).expand(NTOTAL, HIDDEN).contiguous()

    hidden, ds, gl, eids, _ = make_inputs(1024, resident_slots, seed)
    out, _ = run_split(hidden, ds, gl, resident_slots, decode_w, pool_w)
    expected = reference_output(hidden, ds, eids, resident_slots, decode_w,
                                pool_w)
    torch.testing.assert_close(out, expected)


def test_no_resident_rows_single_gmm_fast_path():
    resident_slots = torch.full((NTOTAL,), -1, dtype=torch.int64)
    decode_w = torch.zeros(NSLOTS, HIDDEN)
    pool_w = torch.arange(NTOTAL, dtype=torch.float32).reshape(
        -1, 1).expand(NTOTAL, HIDDEN).contiguous()
    hidden, ds, gl, eids, _ = make_inputs(64, resident_slots, 0)
    out, _ = run_split(hidden, ds, gl, resident_slots, decode_w, pool_w)
    expected = reference_output(hidden, ds, eids, resident_slots, decode_w,
                                pool_w)
    torch.testing.assert_close(out, expected)


def test_all_rows_resident():
    resident_slots = torch.full((NTOTAL,), -1, dtype=torch.int64)
    # route every row into slot-carrying experts 0..7; keep rows sorted by
    # expert id (dispatch order) after remapping
    resident_slots[:8] = torch.arange(8)
    decode_w = 100.0 + torch.arange(NSLOTS, dtype=torch.float32).reshape(
        -1, 1).expand(NSLOTS, HIDDEN).contiguous()
    pool_w = torch.arange(NTOTAL, dtype=torch.float32).reshape(
        -1, 1).expand(NTOTAL, HIDDEN).contiguous()
    hidden, ds, gl, eids, _ = make_inputs(128, resident_slots, 1)
    eids = torch.sort(eids % 8)[0]
    gl = _group_counts(eids, NTOTAL)
    out, _ = run_split(hidden, ds, gl, resident_slots, decode_w, pool_w)
    expected = reference_output(hidden, ds, eids, resident_slots, decode_w,
                                pool_w)
    torch.testing.assert_close(out, expected)


def test_row_count_mismatch_raises():
    """Dispatch rows must equal group_list.sum() (AllGather contract)."""
    resident_slots = torch.full((NTOTAL,), -1, dtype=torch.int64)
    resident_slots[torch.randperm(NTOTAL)[:NSLOTS]] = torch.randperm(NSLOTS)
    decode_w = 100.0 + torch.arange(NSLOTS, dtype=torch.float32).reshape(
        -1, 1).expand(NSLOTS, HIDDEN).contiguous()
    pool_w = torch.arange(NTOTAL, dtype=torch.float32).reshape(
        -1, 1).expand(NTOTAL, HIDDEN).contiguous()
    hidden, ds, gl, eids, _ = make_inputs(64, resident_slots, 2)
    n_pad = 8
    hidden = torch.cat([hidden, torch.full((n_pad, HIDDEN), 7.0)])
    ds = torch.cat([ds, torch.ones(n_pad, 2)])
    with pytest.raises(RuntimeError, match="group_list sum"):
        run_split(hidden, ds, gl, resident_slots, decode_w, pool_w)


def test_group_counts_matches_bincount():
    ids = torch.tensor([3, 3, 0, 7, 3], dtype=torch.int64)
    counts = _group_counts(ids, 8)
    assert counts.tolist() == torch.bincount(ids, minlength=8).tolist()
    assert counts.sum().item() == ids.numel()


def test_fp8_quantized_rows_match_reference():
    """W4A8 dispatch emits fp8 rows + fp8/e8m0 scales; NPU index_select has
    no fp8 kernel so the split must re-index through a uint8 bit view.

    Uses values exactly representable in e4m3/e8m0 (small ints, scale=1) so
    the fp8 path can be compared exactly against the fp32 reference.
    """
    torch.manual_seed(3)
    resident_slots = torch.full((NTOTAL,), -1, dtype=torch.int64)
    resident_slots[torch.randperm(NTOTAL)[:NSLOTS]] = torch.randperm(NSLOTS)
    decode_w = 100.0 + torch.arange(NSLOTS, dtype=torch.float32).reshape(
        -1, 1).expand(NSLOTS, HIDDEN).contiguous()
    pool_w = torch.arange(NTOTAL, dtype=torch.float32).reshape(
        -1, 1).expand(NTOTAL, HIDDEN).contiguous()

    hidden, ds, gl, eids, _ = make_inputs(256, resident_slots, 3)
    hidden = hidden % 12  # e4m3-exact integers
    ds = torch.ones_like(ds)
    hidden_fp8 = hidden.to(torch.float8_e4m3fn)
    ds_fp8 = ds.to(torch.float8_e8m0fnu)

    out_fp8, _ = run_split(hidden_fp8, ds_fp8, gl, resident_slots,
                           decode_w, pool_w)
    expected = reference_output(hidden, ds, eids, resident_slots, decode_w,
                                pool_w)
    torch.testing.assert_close(
        out_fp8.to(torch.float32), expected, rtol=0.0, atol=0.0)


def test_unsupported_group_list_type_raises():
    resident_slots = torch.full((NTOTAL,), -1, dtype=torch.int64)
    mlp_in = MoEMlpComputeInput(
        hidden_states=torch.zeros(4, HIDDEN),
        group_list=torch.zeros(NTOTAL, dtype=torch.int64),
        group_list_type=0,  # cumsum (MC2) -- not splittable
        dynamic_scale=None,
        topk_scales=None,
        weights=MoEWeights(w1=torch.zeros(NTOTAL, HIDDEN),
                           w2=torch.zeros(NTOTAL, HIDDEN)),
        quant=types.SimpleNamespace(is_quant=False),
        fusion=False,
    )
    ctx = PrefillSplitCtx(
        resident_slots=resident_slots,
        decode_weights=MoEWeights(w1=torch.zeros(NSLOTS, HIDDEN),
                                  w2=torch.zeros(NSLOTS, HIDDEN)),
        num_slots=NSLOTS,
    )
    with pytest.raises(RuntimeError, match="count-type group_list"):
        prefill_split_apply(mlp_in, ctx, fake_apply_mlp)
