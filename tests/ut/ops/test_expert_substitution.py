import torch

from vllm_ascend.ops.fused_moe.experts_selector import substitute_experts


def test_substitute_low_confidence_cache_miss():
    # Scores sum to one, so softmax(log(scores)) reproduces them exactly.
    router_logits = torch.log(torch.tensor([[0.40, 0.25, 0.20, 0.15]]))
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.40, 0.25]])
    # Experts 0 and 3 are resident; selected expert 1 is a cache miss.
    log2phy = torch.tensor([0, -1, -1, 1], dtype=torch.int32)

    weights, ids = substitute_experts(
        router_logits,
        topk_weights,
        topk_ids,
        log2phy,
        expert_substitution_threshold=0.30,
    )

    assert ids.tolist() == [[0, 3]]
    torch.testing.assert_close(weights, torch.tensor([[0.40, 0.15]]))


def test_substitution_keeps_high_confidence_cache_miss():
    router_logits = torch.log(torch.tensor([[0.70, 0.15, 0.10, 0.05]]))
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.70, 0.15]])
    log2phy = torch.tensor([-1, 0, -1, 1], dtype=torch.int32)

    weights, ids = substitute_experts(
        router_logits,
        topk_weights,
        topk_ids,
        log2phy,
        expert_substitution_threshold=0.25,
    )

    assert torch.equal(ids, topk_ids)
    assert torch.equal(weights, topk_weights)


def test_correction_bias_only_affects_selection_not_routing_weight():
    router_logits = torch.log(torch.tensor([[0.40, 0.25, 0.20, 0.15]]))
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.40, 0.25]])
    log2phy = torch.tensor([0, -1, -1, 1], dtype=torch.int32)
    correction_bias = torch.tensor([0.0, 0.0, 0.0, 0.04])

    weights, ids = substitute_experts(
        router_logits,
        topk_weights,
        topk_ids,
        log2phy,
        expert_substitution_threshold=0.40,
        e_score_correction_bias=correction_bias,
    )

    assert ids.tolist() == [[0, 3]]
    # The correction bias chooses candidates, but MoE combines with the
    # original (unbiased) router score.
    torch.testing.assert_close(weights, torch.tensor([[0.40, 0.15]]))


def test_substitution_supports_missing_correction_bias_and_renormalize():
    router_logits = torch.log(torch.tensor([[0.40, 0.25, 0.20, 0.15]]))
    weights, ids = substitute_experts(
        router_logits,
        torch.tensor([[0.40, 0.25]]),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([0, -1, -1, 1], dtype=torch.int32),
        renormalize=True,
        expert_substitution_threshold=0.30,
        e_score_correction_bias=None,
        routed_scaling_factor=2.0,
    )

    assert ids.tolist() == [[0, 3]]
    torch.testing.assert_close(weights.sum(dim=-1), torch.full((1,), 2.0))
