"""CPU frequency-based admission for single-expert hybrid transfer."""

FREQUENCY_DECAY = 0.99
FREQUENCY_MIN_SCORE = 2.0
FREQUENCY_REPLACEMENT_RATIO = 1.2


class FrequencyTransferPolicy:
    """CPU-owned per-layer scores, updated once per runtime decision.

    Count every valid token route, including NPU hits. Decay once per batch;
    require repeated demand and a replacement margin to limit cache churn.
    """

    def __init__(self):
        self.scores = []

    def select(self, topk_ids, log2phy, slot_count):
        mapping = [int(slot) for slot in log2phy.tolist()]
        if not self.scores:
            self.scores = [0.0] * len(mapping)
        if len(self.scores) != len(mapping):
            raise ValueError("expert count changed during frequency tracking")
        self.scores = [score * FREQUENCY_DECAY for score in self.scores]
        routed = list(dict.fromkeys(
            int(e) for e in topk_ids.reshape(-1).tolist()
            if 0 <= int(e) < len(mapping)))
        for eid in topk_ids.reshape(-1).tolist():
            if 0 <= int(eid) < len(mapping):
                self.scores[int(eid)] += 1.0
        misses = [eid for eid in routed if mapping[eid] < 0]
        if not misses:
            return None
        candidate = max(misses, key=lambda eid: self.scores[eid])
        if self.scores[candidate] < FREQUENCY_MIN_SCORE:
            return None
        protected = {mapping[eid] for eid in routed if mapping[eid] >= 0}
        available = [slot for slot in range(slot_count) if slot not in protected]
        if not available:
            return None
        # Empty slots have score zero. Taking max also protects against an
        # inconsistent map with multiple logical owners of a physical slot.
        slot_scores = [0.0] * slot_count
        for eid, slot in enumerate(mapping):
            if 0 <= slot < slot_count:
                slot_scores[slot] = max(slot_scores[slot], self.scores[eid])
        victim_slot = min(available, key=lambda slot: slot_scores[slot])
        if self.scores[candidate] <= slot_scores[victim_slot] * FREQUENCY_REPLACEMENT_RATIO:
            return None
        return candidate, victim_slot
