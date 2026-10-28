"""Semantic and optional NPU graph tests for the unintegrated candidate.

    python -m unittest discover -s work -p test_device_prefetch_candidate.py -v

The NPU test checks actual Triton outputs and active->inactive graph replay.
It uses dummy pointer numbers and does not launch the sparse-copy library.
Byte/layout copying still needs the separate remote NPU verification.
"""
from __future__ import annotations

import random
import unittest
from types import SimpleNamespace as NS

from vllm_ascend.expert_offload.device_prefetch import (
    DevicePrefetchCandidate,
    DevicePrefetchSelector,
    PrefetchDecision,
    frozen_victim_priority,
    select_prefetch_one_reference,
    triton,
    unsupported_reason,
)


def _config():
    cpu = NS(enabled=True, weight_format="gguf", transfer_one_expert=False,
             transfer_scheduler="npu", transfer_policy="load_balance",
             transfer_copy_library=__file__)
    config = NS(cpu_moe=cpu, expert_prefetch_enabled=True, expert_prefetch_num=1,
                enable_multi_card=False, cache_policy_enabled=True,
                cache_policy_type="lrc", expert_predictor="fate")
    manager = NS(offload_config=config, ep_size=1, native_mxfp4_enabled=False,
                 num_total_experts=6, prefetch_tokens=1, topk=6,
                 cache_policy=object())
    return manager, NS(log2phy=object(), _hybrid_expert_executor=object())


class SelectionSemantics(unittest.TestCase):
    def decide(self, ids, weights, mapping, heat=None, last=None, slots=2):
        e = len(mapping)
        priority = frozen_victim_priority(heat or [0.] * e, last or [-1.] * e)
        return select_prefetch_one_reference(ids, weights, mapping, priority, slots)

    def test_resident_top_score_does_not_hide_next_missing(self):
        self.assertEqual(self.decide([[0, 2, 1]], [[.9, .8, .7]], [0, -1, -1, 1]),
                         PrefetchDecision(2, 3, 1, True))

    def test_candidate_tie_is_first_occurrence_not_expert_id(self):
        self.assertEqual(self.decide([[2, 1]], [[.5, .5]], [-1, -1, -1, -1, 0, 1]),
                         PrefetchDecision(2, 4, 0, True))

    def test_candidate_uses_max_across_rows_and_stable_insertion(self):
        mapping = [-1, -1, -1, 0, 1]
        self.assertEqual(self.decide([[1, 2], [2, 1]], [[.9, .1], [.95, .8]], mapping).candidate, 2)
        self.assertEqual(self.decide([[1, 2], [2, 1]], [[.9, .1], [.9, .8]], mapping).candidate, 1)

    def test_no_miss_is_inactive(self):
        self.assertEqual(self.decide([[0, 1]], [[.9, .1]], [0, 1, -1]), PrefetchDecision())

    def test_all_predicted_residents_are_protected(self):
        # Predicted-but-not-selected expert 1 remains protected too.
        self.assertEqual(self.decide([[2, 0, 1]], [[.9, .1, .05]], [0, 1, -1]), PrefetchDecision())

    def test_empty_slot_does_not_change_existing_host_policy(self):
        self.assertEqual(self.decide([[1, 0]], [[.9, .1]], [0, -1, -1], slots=2), PrefetchDecision())

    def test_zero_weight_is_a_prediction_and_valid_candidate(self):
        self.assertEqual(self.decide([[2, 0]], [[0., 0.]], [0, 1, -1]),
                         PrefetchDecision(2, 1, 1, True))

    def test_invalid_ids_do_not_index_mapping(self):
        self.assertEqual(self.decide([[-1, 999, 2]], [[1., 1., .3]], [0, 1, -1]),
                         PrefetchDecision(2, 0, 0, True))
        self.assertEqual(self.decide([[-1, 999]], [[1., 1.]], [0, 1, -1]), PrefetchDecision())

    def test_victim_ties_follow_last_then_expert_not_slot(self):
        self.assertEqual(self.decide([[2]], [[.8]], [1, 0, -1], heat=[0., 0., 0.], last=[3., 2., -1.]),
                         PrefetchDecision(2, 1, 0, True))
        self.assertEqual(self.decide([[2]], [[.8]], [1, 0, -1]), PrefetchDecision(2, 0, 1, True))

    def test_frozen_priority_preserves_close_host_float_scores(self):
        self.assertEqual(frozen_victim_priority([1.00000001, 1.00000002], [0., 0.]), [0, 1])

    def test_nan_and_negative_infinity_do_not_insert_candidate(self):
        self.assertEqual(self.decide([[2, 3]], [[float("nan"), float("-inf")]], [0, 1, -1, -1]), PrefetchDecision())

    def test_no_weight_fallback_interleaves_rows_by_rank(self):
        self.assertEqual(self.decide([[0, 3], [2, 4]], None, [0, 1, -1, -1, -1]).candidate, 2)


class StaticFallbackGuards(unittest.TestCase):
    def test_supported_configuration(self):
        manager, layer = _config()
        self.assertIsNone(unsupported_reason(manager, layer))

    def test_unsupported_returns_before_transport_or_device_work(self):
        for field, value in [("expert_prefetch_num", 2), ("enable_multi_card", True),
                             ("cache_policy_type", "lru"), ("expert_predictor", "mode2_har")]:
            manager, layer = _config()
            setattr(manager.offload_config, field, value)
            self.assertIsNone(DevicePrefetchCandidate.try_create(manager, layer))
        manager, layer = _config()
        manager.offload_config.cpu_moe.transfer_one_expert = True
        self.assertIsNone(DevicePrefetchCandidate.try_create(manager, layer))
        manager, layer = _config()
        manager.offload_config.cpu_moe.transfer_copy_library = ""
        self.assertIsNone(DevicePrefetchCandidate.try_create(manager, layer))
        manager, layer = _config()
        layer._hybrid_expert_executor = None
        self.assertIsNone(DevicePrefetchCandidate.try_create(manager, layer))


def _npu_runtime():
    try:
        import torch
        import torch_npu  # noqa: F401
        if triton is not None and torch.npu.is_available():
            return torch
    except ImportError:
        pass
    return None


TORCH_NPU = _npu_runtime()


@unittest.skipUnless(TORCH_NPU is not None, "requires torch-npu and Triton on an NPU host")
class DeviceKernelAndReplay(unittest.TestCase):
    def setUp(self):
        self.torch = TORCH_NPU
        self.torch.npu.set_device(0)
        torch = self.torch
        self.table = torch.arange(24, device="npu", dtype=torch.int64).reshape(6, 4) * 100 + 1000
        self.bases = torch.tensor([10000, 20000, 30000, 40000], device="npu", dtype=torch.int64)
        self.strides = torch.tensor([64, 32, 8, 4], device="npu", dtype=torch.int64)
        self.lengths = self.strides.to(torch.int32)
        self.priority = torch.tensor([2, 0, 1, 3, 4, 5], device="npu", dtype=torch.int32)
        self.selector = DevicePrefetchSelector(self.table, self.bases, self.strides, self.lengths, self.priority, 2)

    def test_explicit_boundary_decisions_match_oracle(self):
        torch = self.torch
        cases = [
            # Tie goes by route insertion, rather than ascending expert ID.
            ([[3, 2]], [[.5, .5]], [0, 1, -1, -1, -1, -1]),
            # Zero is a prediction, not an inactive route sentinel.
            ([[2, 0]], [[0., 0.]], [0, 1, -1, -1, -1, -1]),
            # No miss and no eligible victim are different inactive cases.
            ([[0, 1]], [[1., .5]], [0, 1, -1, -1, -1, -1]),
            ([[0, 1, 2]], [[1., .5, .25]], [0, 1, -1, -1, -1, -1]),
            ([[-1, 99, 2]], [[1., 1., .5]], [0, 1, -1, -1, -1, -1]),
            ([[-1, 99]], [[1., 1.]], [0, 1, -1, -1, -1, -1]),
            ([[2, 3], [3, 2]], [[.5, .25], [.75, .5]], [0, 1, -1, -1, -1, -1]),
        ]
        for ids, weights, mapping in cases:
            with self.subTest(ids=ids, weights=weights):
                expected = select_prefetch_one_reference(ids, weights, mapping, self.priority.cpu().tolist(), 2)
                map_d = torch.tensor(mapping, device="npu", dtype=torch.int32)
                self.selector.prepare(torch.tensor(ids, device="npu", dtype=torch.int32), map_d,
                                      torch.tensor(weights, device="npu", dtype=torch.float32))
                torch.npu.synchronize()
                self.assertEqual(bool(self.selector.active.cpu().item()), expected.active)
                self.assertEqual(self.selector.copy_count.cpu().item(), 4 if expected.active else 0)
                if expected.active:
                    actual = (self.selector.candidate.cpu().item(), self.selector.victim.cpu().item(),
                              self.selector.slot.cpu().item())
                    self.assertEqual(actual, (expected.candidate, expected.victim, expected.slot))
                else:
                    self.assertEqual(self.selector.copy_lengths.cpu().tolist(), [0, 0, 0, 0])

    def test_random_decisions_and_descriptors_match_host_semantics(self):
        torch = self.torch
        rng = random.Random(20261025)
        for _ in range(100):
            owners = rng.sample(range(6), 2)
            mapping = [-1] * 6
            for slot, eid in enumerate(owners):
                mapping[eid] = slot
            ids = [[rng.choice([-1, 0, 1, 2, 3, 4, 5, 99]) for _ in range(3)] for _ in range(2)]
            weights = [[rng.choice([0., .25, .5, 1.]) for _ in range(3)] for _ in range(2)]
            expected = select_prefetch_one_reference(ids, weights, mapping, self.priority.cpu().tolist(), 2)
            map_d = torch.tensor(mapping, device="npu", dtype=torch.int32)
            self.selector.prepare(torch.tensor(ids, device="npu", dtype=torch.int32), map_d,
                                  torch.tensor(weights, device="npu", dtype=torch.float32))
            torch.npu.synchronize()
            self.assertEqual(bool(self.selector.active.cpu().item()), expected.active)
            self.assertEqual(self.selector.copy_count.cpu().item(), 4 if expected.active else 0)
            if expected.active:
                self.assertEqual(self.selector.candidate.cpu().item(), expected.candidate)
                self.assertEqual(self.selector.victim.cpu().item(), expected.victim)
                self.assertEqual(self.selector.slot.cpu().item(), expected.slot)
                self.assertEqual(self.selector.copy_sources.cpu().tolist(), self.table[expected.candidate].cpu().tolist())
                expected_dst = (self.bases + expected.slot * self.strides).cpu().tolist()
                self.assertEqual(self.selector.copy_destinations.cpu().tolist(), expected_dst)
            else:
                self.assertEqual(self.selector.copy_lengths.cpu().tolist(), [0, 0, 0, 0])
            # Selection must not publish residency before caller's copy.
            self.assertEqual(map_d.cpu().tolist(), mapping)
            self.selector.commit(map_d)
            torch.npu.synchronize()
            if expected.active:
                mapping[expected.victim] = -1
                mapping[expected.candidate] = expected.slot
            self.assertEqual(map_d.cpu().tolist(), mapping)

    def test_graph_active_to_inactive_clears_count_and_preserves_map(self):
        torch = self.torch
        ids = torch.tensor([[2, 3]], device="npu", dtype=torch.int32)
        weights = torch.tensor([[1., .5]], device="npu", dtype=torch.float32)
        mapping = torch.tensor([0, 1, -1, -1, -1, -1], device="npu", dtype=torch.int32)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            self.selector.prepare(ids, mapping, weights)  # Compile before capture.
            self.selector.commit(mapping)
        stream.synchronize()
        mapping.copy_(torch.tensor([0, 1, -1, -1, -1, -1], device="npu", dtype=torch.int32))
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            self.selector.prepare(ids, mapping, weights)
            self.selector.commit(mapping)
        with torch.npu.stream(stream):
            for _ in range(20):
                mapping.copy_(torch.tensor([0, 1, -1, -1, -1, -1], device="npu", dtype=torch.int32))
                ids.copy_(torch.tensor([[2, 3]], device="npu", dtype=torch.int32))
                graph.replay()
                stream.synchronize()
                self.assertEqual(self.selector.copy_count.cpu().item(), 4)
                # No miss: IDs now name the two resident experts. This checks
                # that Active/Count never retain a previous replay's decision.
                ids.copy_(torch.tensor([[0, 2]], device="npu", dtype=torch.int32))
                previous = mapping.cpu().tolist()
                graph.replay()
                stream.synchronize()
                self.assertEqual(self.selector.copy_count.cpu().item(), 0)
                self.assertEqual(self.selector.copy_lengths.cpu().tolist(), [0, 0, 0, 0])
                self.assertEqual(mapping.cpu().tolist(), previous)


if __name__ == "__main__":
    unittest.main()
