# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Pluggable next-layer expert predictors for proactive prefetch.

DESIGN CHOICE (why this file exists)
------------------------------------
Proactive prefetch has exactly one swappable decision: *which experts will the
NEXT MoE layer need?*  Everything else — the background prefetch worker, the
H2D copy loop, LRC eviction (choose_victim), the log2phy update, the completion
events, and the graph-mode staging — is mechanism that does not care HOW the
prediction was made.  So we isolate only the prediction behind this abstraction
and leave all that mechanism untouched.

The default predictor implements the "Fate" cross-layer-gate method
(arXiv:2502.12224): take the current layer's hidden states as a cheap proxy for
the next layer's router input, run the next layer's gate, and top-k the result.

HARD OUTPUT CONTRACT (do not break)
-----------------------------------
`predict_from_device` / `predict_from_host` MUST return `set[int] | None`:
  * a set of predicted next-layer expert ids, or
  * None to skip prefetch this step (last layer, or prediction impossible).
The downstream `_do_prefetch` consumes a plain `set[int]`.  Returning a device
tensor (or anything else) would force changes to the eviction/copy path, which
must stay untouched.  A predictor that computes on NPU must do its own
`.tolist()` / `set(...)` at the boundary and return a CPU set.

HOW TO ADD A NEW PREDICTOR (future integration guide)
-----------------------------------------------------
1. Add a value to `ExpertPredictorType` (e.g. `NN = "nn"`).
2. Create a new file anywhere in the tree (e.g.
   `vllm_ascend/expert_offload/predictors_nn.py`), subclass `ExpertPredictor`,
   implement `predict_from_device` and `predict_from_host`, and decorate the
   class with `@register_predictor(ExpertPredictorType.NN)`.
3. Make sure that module is imported once so the decorator runs (registration
   is import-time).  Either add `from . import predictors_nn  # noqa: F401`
   at the bottom of THIS file, or import it where the manager is set up.
4. Select it via config: `expert_offload_config.expert_predictor = "nn"`.

What a predictor may read (via `self.mgr`, the ExpertOffloadManager back-ref):
  * `self.mgr.moe_layers`            — list of MoE layers (and its length).
  * `self.mgr.topk`                  — experts per token.
  * `self.mgr._gate_weights_cpu`     — fp32 CPU gate weights per layer (FATE).
  * `self.mgr._gate_weights_npu`     — fp32 NPU gate weights per layer (on-device variants).
  * `self.mgr.cache_policy.layer_states[idx]` — recent routing history / freq /
    ema, for activation-path or temporal predictors.
  * `self.mgr._npu_device`           — device for NPU-side predictors.
WEIGHT LOADING / PLACEMENT (per-predictor — this is the crux of register_from_model)
------------------------------------------------------------------------------------
`register_from_model(model)` is the predictor-agnostic load-time hook, called
once after the model's weights are loaded (from the manager's finalize path —
see Group 4). Each predictor decides WHAT parameters it needs and WHERE they
live; placement is a per-predictor decision, never a manager-wide assumption:
  * FATE keeps fp32 gate weights on CPU (its prediction runs on CPU). Its
    register_from_model delegates to the manager's FATE-specific
    register_gate_weights() helper, which populates self.mgr._gate_weights_cpu
    (+ an NPU mirror). A NON-FATE predictor does NOT trigger this, so switching
    away from FATE means no CPU gate registration happens at all.
  * An AI/NN predictor should load its OWN network weights onto the NPU (HBM),
    NOT CPU, since its prediction runs on-device. Resolve the device from the
    model (e.g. next(model.parameters()).device) or self.mgr._npu_device, move
    the net with .to(dev), and keep it on the predictor instance. It must still
    return a CPU set[int] (hard contract) — do the .tolist()/set() at the
    boundary. Skeleton (illustrative, NOT enabled):

      # in a new file, e.g. vllm_ascend/expert_offload/predictors_nn.py
      # @register_predictor(ExpertPredictorType.NN)
      # class NNPredictor(ExpertPredictor):
      #     def register_from_model(self, model):
      #         dev = next(model.parameters()).device       # NPU device (HBM)
      #         self.net = MyPredictorNet(...).to(dev).eval()  # weights in HBM
      #         # optionally torch.load(ckpt, map_location=dev) -> load_state_dict
      #     def predict_from_device(self, layer_idx, hidden_states):
      #         next_idx = layer_idx + 1
      #         if next_idx >= len(self.mgr.moe_layers):
      #             return None
      #         with torch.no_grad():                        # runs on NPU/HBM
      #             logits = self.net(hidden_states, layer_idx)
      #             ids = logits.topk(self.mgr.topk, dim=-1).indices
      #         return set(ids.flatten().tolist())           # hard contract: CPU set
      #     def predict_from_host(self, layer_idx, hs_cpu):
      #         # graph mode: move the staged CPU tensor to HBM, reuse on-device path
      #         dev = next(self.net.parameters()).device
      #         return self.predict_from_device(layer_idx, hs_cpu.to(dev))

CAVEATS for future predictors
-----------------------------
  * Graph mode: `trigger_next_layer_prefetch` currently stages ONLY
    `hidden_states` into a pinned buffer (because that is what FATE needs), and
    the graph host callback then calls `predict_from_host` over it.  A predictor
    that needs different staged inputs (e.g. routing tensors) will work in eager
    mode out of the box, but for graph mode it needs an additional staging hook
    in `trigger_next_layer_prefetch` (not provided here — extend when needed).
  * NPU-side prediction may run on the compute stream OR a separate stream —
    this is a performance trade-off, not a correctness requirement.  Running on
    the compute stream contends with the main forward pass (the same effect we
    measured for prefetch H2D vs. attention), so prefer a dedicated / the
    prefetch stream if that contention costs more than the prediction saves;
    either is allowed.  Regardless of stream choice, convert the result to a
    CPU set before returning (the hard output contract above).
  * Thread-safety: `predict_*` runs on the prefetch worker thread (eager) or the
    graph host-callback thread (graph) — never the main thread.  FATE is
    stateless so this is safe; a STATEFUL predictor must guard its own state and
    must not assume which thread it is on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    # Type-only import: avoids any runtime import cycle with the manager.
    from vllm_ascend.expert_offload.expert_offload_manager import (
        ExpertOffloadManager,
    )


class ExpertPredictorType(Enum):
    """Selectable predictor strategies. The config string maps to a member."""

    FATE = "fate"  # cross-layer gate (arXiv:2502.12224) — the default.


class ExpertPredictor(ABC):
    """Base class for next-layer expert predictors.

    See the module docstring for the output contract and the guide to adding
    new predictors. Subclasses get a read-only back-reference to the owning
    ExpertOffloadManager as `self.mgr`.
    """

    def __init__(self, mgr: "ExpertOffloadManager") -> None:
        self.mgr = mgr

    @abstractmethod
    def predict_from_device(
        self, layer_idx: int, hidden_states: torch.Tensor
    ) -> set[int] | None:
        """Predict next layer's experts from an on-device hidden_states tensor.

        Called on the prefetch worker thread (eager mode) after the current
        layer's GMM has completed, so a blocking .cpu() here is safe.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_from_host(
        self, layer_idx: int, hs_cpu: torch.Tensor
    ) -> set[int] | None:
        """Predict next layer's experts from an already-CPU hidden_states tensor.

        Called from the graph-mode host callback over the pre-staged pinned
        buffer — must not do a blocking D2H on a live graph tensor.
        """
        raise NotImplementedError

    def register_from_model(self, model) -> None:
        """Optional hook: snapshot parameters this predictor needs at weight-load
        time. Default is a no-op (FATE reads gate weights owned by the manager).
        Wired in Group 4 of the implementation guide.
        """
        return None


# enum member -> predictor class. Populated by @register_predictor at import.
_PREDICTOR_REGISTRY: dict[ExpertPredictorType, type[ExpertPredictor]] = {}


def register_predictor(ptype: ExpertPredictorType):
    """Class decorator: register a predictor implementation for `ptype`."""

    def _decorator(cls: type[ExpertPredictor]) -> type[ExpertPredictor]:
        _PREDICTOR_REGISTRY[ptype] = cls
        return cls

    return _decorator


def make_predictor(name: str, mgr: "ExpertOffloadManager") -> ExpertPredictor:
    """Construct the predictor selected by config string `name`.

    Raises ValueError on an unknown name or an unregistered (but valid) type,
    so a config typo fails fast at manager construction.
    """
    try:
        ptype = ExpertPredictorType(name)
    except ValueError:
        valid = ", ".join(repr(t.value) for t in ExpertPredictorType)
        raise ValueError(
            f"Unknown expert_predictor {name!r}; valid values: {valid}"
        )
    cls = _PREDICTOR_REGISTRY.get(ptype)
    if cls is None:
        raise ValueError(f"No predictor registered for {ptype}")
    return cls(mgr)


@register_predictor(ExpertPredictorType.FATE)
class FATEPredictor(ExpertPredictor):
    """Cross-layer-gate predictor (Fate, arXiv:2502.12224) — the default.

    Behavior is identical to the original ExpertOffloadManager methods
    predict_next_layer_experts() / _predict_next_layer_experts_cpu(): use the
    current layer's hidden_states as an approximation of the next layer's input,
    run the next layer's gate, and apply a simplified softmax + top-k (instead
    of the full grouped_topk) for speed. Misses are handled by the reactive
    fallback in update_weights(). Reads gate weights / topk / moe_layers from
    the manager (`self.mgr`); ownership of those stays on the manager so this
    refactor is behavior-preserving and minimal.
    """

    def predict_from_device(
        self, layer_idx: int, hidden_states: torch.Tensor
    ) -> set[int] | None:
        # NEW: this is the body of the original predict_next_layer_experts(),
        # relocated verbatim. Early-out BEFORE the .cpu() (last layer / missing
        # gate) is preserved so the last layer never pays an unnecessary D2H.
        next_idx = layer_idx + 1
        if next_idx >= len(self.mgr.moe_layers):
            return None  # last layer — nothing to prefetch
        if next_idx >= len(self.mgr._gate_weights_cpu):
            return None
        if self.mgr._gate_weights_cpu[next_idx] is None:
            return None
        hs_cpu = hidden_states.float().cpu()
        return self.predict_from_host(layer_idx, hs_cpu)

    def predict_from_host(
        self, layer_idx: int, hs_cpu: torch.Tensor
    ) -> set[int] | None:
        # NEW: body of the original _predict_next_layer_experts_cpu(), verbatim.
        next_idx = layer_idx + 1
        if next_idx >= len(self.mgr.moe_layers):
            return None  # last layer — nothing to prefetch
        if next_idx >= len(self.mgr._gate_weights_cpu):
            return None
        gate_w = self.mgr._gate_weights_cpu[next_idx]
        if gate_w is None:
            return None

        # hs_cpu is fp32 on CPU; gate_w is fp32 on CPU.
        router_logits = F.linear(hs_cpu, gate_w)  # [num_tokens, n_experts]
        # Simplified routing: softmax + topk (approximation).
        probs = router_logits.softmax(dim=-1)
        _, topk_ids = probs.topk(self.mgr.topk, dim=-1)
        return set(topk_ids.flatten().tolist())

    def register_from_model(self, model) -> None:
        # NEW: FATE's parameters are the per-layer gate weights, kept on CPU
        # (its prediction runs on CPU). This delegates to the manager's existing
        # FATE-specific helper register_gate_weights(), which populates
        # self.mgr._gate_weights_cpu (+ an NPU mirror) that predict_* read above.
        # IMPORTANT: this CPU gate registration only runs because FATE is the
        # active predictor — a different predictor's register_from_model would
        # load its own weights (e.g. an NN onto HBM) and would NOT call this, so
        # switching predictors no longer forces CPU gate registration. The
        # manager helper is left in place (not moved here) to keep the diff
        # minimal; it could be relocated into this class later.
        self.mgr.register_gate_weights(model)