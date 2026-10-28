"""Policy tests runnable without torch/NPU or the vLLM package imports."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "frequency_transfer_policy",
    Path(__file__).resolve().parents[3]
    / "vllm_ascend/expert_offload/frequency_transfer_policy.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HostArray:
    def __init__(self, values):
        self.values = values

    def reshape(self, _):
        return self

    def tolist(self):
        return self.values


class FrequencyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = module.FrequencyTransferPolicy()

    def select(self, ids, mapping, slots=2):
        return self.policy.select(HostArray(ids), HostArray(mapping), slots)

    def test_cold_miss_requires_repeated_demand(self):
        self.assertIsNone(self.select([2], [0, 1, -1]))
        self.assertIsNone(self.select([2], [0, 1, -1]))
        self.assertEqual(self.select([2], [0, 1, -1]), (2, 0))

    def test_token_counts_choose_hottest_miss(self):
        self.assertEqual(self.select([2, 3, 3, 3], [0, 1, -1, -1]), (3, 0))

    def test_hits_are_counted_and_current_slots_protected(self):
        self.assertEqual(self.select([0, 0, 2, 2, 2], [0, 1, -1]), (2, 1))
        self.assertEqual(self.policy.scores[0], 2.0)
        self.assertIsNone(self.select([0, 1, 2, 2], [0, 1, -1]))

    def test_hot_resident_is_not_replaced_by_cold_miss(self):
        self.select([0] * 10 + [1] * 10, [0, 1, -1])
        self.assertIsNone(self.select([2] * 3, [0, 1, -1]))

    def test_lowest_score_victim_and_margin(self):
        self.policy.scores = [10.0, 3.0, 0.0]
        self.assertIsNone(self.select([2] * 3, [0, 1, -1]))
        self.assertEqual(self.select([2] * 2, [0, 1, -1]), (2, 1))

    def test_empty_slot_preferred(self):
        self.policy.scores = [100.0, 0.0]
        self.assertEqual(self.select([1] * 3, [0, -1]), (1, 1))

    def test_invalid_routes_ignored_and_scores_decay(self):
        self.policy.scores = [10.0, 0.0]
        self.assertIsNone(self.select([-1, 9], [0, -1]))
        self.assertAlmostEqual(self.policy.scores[0], 9.9)
        self.assertEqual(self.policy.scores[1], 0.0)

    def test_per_layer_state_is_independent(self):
        self.select([0] * 3, [-1], slots=1)
        other = module.FrequencyTransferPolicy()
        self.assertIsNone(other.select(HostArray([0]), HostArray([-1]), 1))


if __name__ == "__main__":
    unittest.main()
