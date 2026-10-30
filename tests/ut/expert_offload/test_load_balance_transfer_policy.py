"""Proportional decisions, eviction safety, and CPU/NPU route conservation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import torch

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend/expert_offload"
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
freq = load("balance_frequency", "frequency_transfer_policy.py")
with patch.dict(sys.modules, {"vllm_ascend.expert_offload.frequency_transfer_policy": freq}):
    module = load("balance_policy", "load_balance_transfer_policy.py")

def make(device="cpu", experts=32, slots=16):
    config = SimpleNamespace(balance_max_transfers=16, balance_cpu_expert_us=70.,
                             balance_cpu_extra_route_us=10., balance_copy_us=450.,
                             balance_npu_route_us=2., balance_min_gain_us=20.)
    return module.LoadBalanceTransferPolicy(experts, slots, device, config)

def test_small_work_stays_on_cpu():
    policy = make(); mapping = torch.tensor(list(range(16)) + [-1]*16)
    before = mapping.clone()
    _, _, active = policy.select(torch.arange(16, 22).reshape(1, -1), mapping)
    assert active.sum() == 0
    assert torch.equal(before, mapping)

def test_multiple_uploads_and_coldest_victims():
    policy = make(); mapping = torch.tensor(list(range(16)) + [-1]*16)
    policy.scores.copy_(torch.arange(16, 0, -1).repeat(2))
    candidates, slots, active = policy.select(torch.arange(16, 30).reshape(2, 7), mapping)
    assert active.sum() == 1
    assert set(slots[active].tolist()) == {15}
    assert len(set(slots[active].tolist())) == 1
    assert all(mapping[e] == slot for e, slot in zip(candidates[active], slots[active]))
    assert mapping[14] == 14 and mapping[15] == -1

def test_current_routes_protected_even_if_cold():
    policy = make(); mapping = torch.tensor(list(range(16)) + [-1]*16)
    original = mapping.clone()
    policy.select(torch.arange(30).reshape(5, 6), mapping)
    assert torch.equal(mapping, original)

def test_random_route_conservation_and_unique_slots():
    gen = torch.Generator().manual_seed(4242)
    policy = make(); mapping = torch.tensor(list(range(16)) + [-1]*16)
    for _ in range(100):
        ids = torch.randint(-2, 35, (8, 6), generator=gen)
        old = mapping.clone(); candidates, slots, active = policy.select(ids, mapping)
        valid = (ids >= 0) & (ids < len(mapping))
        routed = set(ids[valid].tolist())
        for expert in routed:
            if old[expert] >= 0:assert mapping[expert] == old[expert]
        actual = mapping[mapping >= 0].tolist()
        assert len(actual) == len(set(actual))
        for expert, slot in zip(candidates[active], slots[active]):
            assert old[expert] < 0 and int(expert) in routed
            assert mapping[expert] == slot
        cpu = valid & (mapping[ids.clamp(0, len(mapping)-1)] < 0)
        npu = valid & ~cpu
        assert torch.equal(cpu | npu, valid)
        assert not (cpu & npu).any()
