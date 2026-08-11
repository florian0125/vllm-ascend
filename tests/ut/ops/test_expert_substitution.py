import torch

from vllm_ascend.ops.fused_moe.experts_selector import substitute_experts


def test_substitute_low_confidence_cache_miss():
    # Scores sum to one, so softmax(log(scores)) reproduces them exactly.
    router_logits = torch.log(torch.tensor([[0.40, 0.25, 0.20, 0.15]]))
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    # Experts 0 and 3 are resident; selected expert 1 is a cache miss.
    log2phy = torch.tensor([0, -1, -1, 1], dtype=torch.int32)

    ids = substitute_experts(
        router_logits,
        topk_ids,
        log2phy,
        expert_substitution_threshold=0.30,
    )

    assert ids.tolist() == [[0, 3]]


def test_substitution_keeps_high_confidence_cache_miss():
    router_logits = torch.log(torch.tensor([[0.70, 0.15, 0.10, 0.05]]))
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    log2phy = torch.tensor([-1, 0, -1, 1], dtype=torch.int32)

    ids = substitute_experts(
        router_logits,
        topk_ids,
        log2phy,
        expert_substitution_threshold=0.25,
    )

    assert torch.equal(ids, topk_ids)


def test_correction_bias_affects_candidate_selection():
    router_logits = torch.log(torch.tensor([[0.40, 0.25, 0.20, 0.15]]))
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    log2phy = torch.tensor([0, -1, -1, 1], dtype=torch.int32)
    correction_bias = torch.tensor([0.0, 0.0, 0.0, 0.04])

    ids = substitute_experts(
        router_logits,
        topk_ids,
        log2phy,
        expert_substitution_threshold=0.40,
        e_score_correction_bias=correction_bias,
    )

    assert ids.tolist() == [[0, 3]]


def test_substitution_supports_missing_correction_bias():
    router_logits = torch.log(torch.tensor([[0.40, 0.25, 0.20, 0.15]]))
    ids = substitute_experts(
        router_logits,
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([0, -1, -1, 1], dtype=torch.int32),
        expert_substitution_threshold=0.30,
        e_score_correction_bias=None,
    )

    assert ids.tolist() == [[0, 3]]


def test_substitution_keeps_all_references_when_one_is_high_confidence():
    router_logits = torch.log(torch.tensor([
        [0.40, 0.25, 0.20, 0.15],
        [0.15, 0.70, 0.10, 0.05],
    ]))
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    log2phy = torch.tensor([0, -1, -1, 1], dtype=torch.int32)

    ids = substitute_experts(
        router_logits,
        topk_ids,
        log2phy,
        expert_substitution_threshold=0.30,
    )

    assert torch.equal(ids, topk_ids)


def test_substitution_replaces_all_references_to_a_missing_expert():
    router_logits = torch.log(torch.tensor([
        [0.40, 0.25, 0.20, 0.15],
        [0.35, 0.25, 0.22, 0.18],
    ]))
    topk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    log2phy = torch.tensor([0, -1, -1, 1], dtype=torch.int32)

    ids = substitute_experts(
        router_logits,
        topk_ids,
        log2phy,
        expert_substitution_threshold=0.30,
    )

    assert ids.tolist() == [[0, 3], [0, 3]]


def test_substitution_rolls_back_group_when_one_reference_has_no_candidate():
    router_logits = torch.log(torch.tensor([
        [0.40, 0.25, 0.20, 0.15],
        [0.40, 0.25, 0.24, 0.11],
    ]))
    topk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    log2phy = torch.tensor([0, -1, -1, 1], dtype=torch.int32)

    ids = substitute_experts(
        router_logits,
        topk_ids,
        log2phy,
        expert_substitution_threshold=0.30,
    )

    assert torch.equal(ids, topk_ids)


def test_substitution_does_not_duplicate_candidate_within_token():
    router_logits = torch.log(
        torch.tensor([[0.25, 0.24, 0.20, 0.18, 0.13]]))
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    # Only expert 3 is both resident and inside the substitution score band.
    log2phy = torch.tensor([-1, -1, -1, 0, -1], dtype=torch.int32)

    ids = substitute_experts(
        router_logits,
        topk_ids,
        log2phy,
        expert_substitution_threshold=0.30,
    )

    assert ids.tolist() == [[0, 3]]
    assert len(set(ids[0].tolist())) == ids.shape[1]


def test_substitution_accepts_resident_candidate_at_boundary():
    router_logits = torch.log(torch.tensor([[0.40, 0.25, 0.20, 0.15]]))
    ids = substitute_experts(
        router_logits,
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([0, -1, 1, -1], dtype=torch.int32),
        expert_substitution_threshold=0.30,
    )

    assert ids.tolist() == [[0, 2]]
