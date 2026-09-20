"""Expert pruning and substitution; base routing is provided by router/."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm_ascend.ascend_config import get_ascend_config

_prune_thr_npu = None


def dynamic_pruning_unsorted(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    thresholds: torch.Tensor,
    log2phy: torch.Tensor | None = None,
    debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Drop weak top-k experts. Invalid logical id is -1 (not Omni's 512)."""
    invalid_id = -1

    topk_weight_sorted, sorted_indices = torch.sort(topk_weights, dim=1, descending=True)
    topk_id_sorted = torch.gather(topk_ids, dim=1, index=sorted_indices)
    topk_weight_sum = torch.sum(topk_weight_sorted, dim=1, keepdim=True)
    thr = thresholds.to(device=topk_weight_sorted.device, dtype=topk_weight_sorted.dtype)
    if thr.numel() != topk_weights.shape[1]:
        raise ValueError(f"experts_pruning_threshold length {thr.numel()} != top_k {topk_weights.shape[1]}")

    weak_mask = topk_weight_sorted < topk_weight_sum * thr

    if log2phy is not None:
        valid_mask = topk_id_sorted >= 0
        safe_ids = topk_id_sorted.clamp(min=0).long()

        physical_ids = log2phy[safe_ids]

        miss_mask = valid_mask & (physical_ids < 0)
        prune_mask = weak_mask & miss_mask
    else:
        prune_mask = weak_mask

    debug_tensor = None
    if debug and log2phy is not None:
        invalid_ids = torch.full_like(topk_id_sorted, -1)
        remaining_miss_mask = miss_mask & ~prune_mask
        debug_tensor = torch.stack(
            [
                # [0] 剪枝前所有miss路由
                torch.where(
                    miss_mask,
                    topk_id_sorted,
                    invalid_ids,
                ),
                # [1] 被剪掉的路由
                torch.where(
                    prune_mask,
                    topk_id_sorted,
                    invalid_ids,
                ),
                # [2] 剪枝后仍保留、仍需H2D的miss路由
                torch.where(
                    remaining_miss_mask,
                    topk_id_sorted,
                    invalid_ids,
                ),
            ],
            dim=0,
        )

    new_topk_weights = topk_weight_sorted.masked_fill(prune_mask, 0)
    new_topk_ids = topk_id_sorted.masked_fill(prune_mask, invalid_id)
    return new_topk_weights, new_topk_ids.to(topk_ids.dtype), debug_tensor


def maybe_prune_topk_experts(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    log2phy: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    try:
        cfg = get_ascend_config().expert_offload_config
    except RuntimeError:
        return topk_weights, topk_ids, None

    if not cfg.experts_pruning_enabled:
        return topk_weights, topk_ids, None

    global _prune_thr_npu
    if (
        _prune_thr_npu is None
        or _prune_thr_npu.device != topk_weights.device
        or _prune_thr_npu.dtype != topk_weights.dtype
        or _prune_thr_npu.numel() != len(cfg.experts_pruning_threshold)
    ):
        _prune_thr_npu = torch.tensor(
            cfg.experts_pruning_threshold,
            device=topk_weights.device,
            dtype=topk_weights.dtype,
        )
    return dynamic_pruning_unsorted(
        topk_weights,
        topk_ids,
        _prune_thr_npu,
        log2phy=log2phy,
        debug=cfg.experts_pruning_debug,
    )


@dataclass
class ExpertSubstitutionPlan:
    """A per-source-expert, all-or-nothing substitution plan."""

    replacements: dict[int, list[tuple[int, int, int]]]
    referenced: torch.Tensor
    blocked: torch.Tensor


def _expert_routing_scores(
    router_logits: torch.Tensor,
    scoring_func: str,
) -> torch.Tensor:
    if scoring_func == "softmax":
        return router_logits.softmax(dim=-1)
    if scoring_func == "sigmoid":
        return router_logits.sigmoid()
    if scoring_func == "sqrtsoftplus":
        return F.softplus(router_logits).sqrt()
    raise ValueError(f"Unsupported scoring function: {scoring_func}")


def plan_expert_substitutions(
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    log2phy: torch.Tensor,
    expert_substitution_threshold: float = 0.25,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> ExpertSubstitutionPlan:
    """Plan atomic substitutions grouped by each non-resident source expert.

    All inputs are expected to be CPU tensors. Selection confidence follows
    the router's scoring function and optional correction bias. A source
    expert is blocked when any of its token references is high-confidence or
    cannot be assigned a distinct resident candidate. No route is mutated here.
    """
    num_tokens, top_k = topk_ids.shape
    num_experts = router_logits.shape[-1]
    routing_scores = _expert_routing_scores(router_logits, scoring_func)
    referenced = torch.zeros(num_experts, dtype=torch.bool)
    blocked = torch.zeros(num_experts, dtype=torch.bool)
    empty_plan = ExpertSubstitutionPlan({}, referenced, blocked)
    if num_tokens == 0 or top_k == 0 or top_k >= num_experts:
        return empty_plan

    selection_scores = routing_scores
    if e_score_correction_bias is not None:
        selection_scores = routing_scores + e_score_correction_bias.unsqueeze(0)

    sorted_scores = selection_scores.sort(dim=-1, descending=True).values
    boundary = sorted_scores[:, top_k].clamp(min=0)
    upper = boundary * (1 + expert_substitution_threshold)
    lower = boundary * (1 - expert_substitution_threshold)

    # 剪枝后的-1不是有效专家，不能参与gather、log2phy索引和替换。
    valid_routes = (topk_ids >= 0) & (topk_ids < num_experts)

    # gather不接受-1；log2phy[-1]还会错误读取最后一个专家。
    safe_ids = topk_ids.clamp(
        min=0,
        max=num_experts - 1,
    ).long()

    selected_scores = selection_scores.gather(1, safe_ids)
    low_confidence = (selected_scores < upper.unsqueeze(1)) & valid_routes
    in_cache = valid_routes & (log2phy[safe_ids] >= 0)

    cached_experts = set(torch.where(log2phy >= 0)[0].tolist())
    references: dict[int, list[tuple[int, int]]] = {}
    for token_idx, position in (valid_routes & ~in_cache).nonzero(as_tuple=False).tolist():
        source_id = int(topk_ids[token_idx, position])
        referenced[source_id] = True
        references.setdefault(source_id, []).append((token_idx, position))

    # Build plans deterministically. Earlier committed plans reserve their
    # candidate IDs within a token; a failed source group changes nothing.
    planned_ids = topk_ids.clone()
    replacements: dict[int, list[tuple[int, int, int]]] = {}
    source_order = sorted(
        references,
        key=lambda source_id: (
            max(selected_scores[token_idx, position].item() for token_idx, position in references[source_id]),
            -len(references[source_id]),
            source_id,
        ),
    )
    for source_id in source_order:
        source_references = sorted(
            references[source_id],
            key=lambda item: (selected_scores[item[0], item[1]].item(), item[0], item[1]),
        )
        if any(not bool(low_confidence[token_idx, position]) for token_idx, position in source_references):
            blocked[source_id] = True
            continue

        tentative: list[tuple[int, int, int]] = []
        tentative_ids = planned_ids.clone()
        for token_idx, position in source_references:
            token_scores = selection_scores[token_idx]
            in_range = (token_scores >= lower[token_idx]) & (token_scores <= boundary[token_idx])
            candidates = set(torch.where(in_range)[0].tolist()) & cached_experts
            candidates.difference_update(tentative_ids[token_idx].tolist())
            if not candidates:
                blocked[source_id] = True
                break
            substitute_id = max(
                candidates,
                key=lambda expert_id: (token_scores[expert_id].item(), -expert_id),
            )
            tentative_ids[token_idx, position] = substitute_id
            tentative.append((token_idx, position, substitute_id))

        if not bool(blocked[source_id]):
            planned_ids = tentative_ids
            replacements[source_id] = tentative

    return ExpertSubstitutionPlan(replacements, referenced, blocked)


def commit_expert_substitutions(
    plan: ExpertSubstitutionPlan,
    allowed_experts: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Atomically apply allowed groups without changing routing weights."""
    out_ids = topk_ids.clone()
    for source_id in sorted(plan.replacements):
        if not bool(allowed_experts[source_id]):
            continue
        for token_idx, position, substitute_id in plan.replacements[source_id]:
            out_ids[token_idx, position] = substitute_id
    return out_ids


def substitute_experts(
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    log2phy: torch.Tensor,
    expert_substitution_threshold: float = 0.25,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace IDs atomically while preserving the router's original weights."""
    plan = plan_expert_substitutions(
        router_logits,
        topk_ids,
        log2phy,
        expert_substitution_threshold=expert_substitution_threshold,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias,
    )
    allowed = plan.referenced & ~plan.blocked
    return commit_expert_substitutions(
        plan,
        allowed,
        topk_ids,
    )


def substitute_experts_device(
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    log2phy: torch.Tensor,
    expert_substitution_threshold: float = 0.25,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Device-resident mirror of ``substitute_experts``.

    Same rule as the host planner — replace a non-resident, low-confidence
    routed expert with a resident one whose score falls inside the importance
    band below the top-k boundary (SMoE, arXiv 2508.18983) — expressed as
    static-shape NPU ops so it runs inside a captured graph and never touches
    the host. Routing weights are NOT modified, matching the host version.

    All tensors are NPU tensors: ``router_logits`` [n, E] in the model dtype,
    ``topk_ids`` [n, k] int32, ``log2phy`` [E] int32 (>= 0 means resident),
    ``e_score_correction_bias`` [E] or None. Returns a NEW [n, k] tensor in
    ``topk_ids``' dtype; the caller decides whether to write it back.

    Verified against plan_expert_substitutions + commit_expert_substitutions
    on 400 random decode steps per cell at E=256, k=6, 3 MoE rows,
    sqrtsoftplus + bias: BIT-IDENTICAL at threshold 0.02 for 24/36/60 resident
    experts; 97.5% identical and +0.09% experts still needing H2D at 0.05;
    93.2% / +0.56% at 0.10; 78.0% / +1.28% at 0.25. Zero violations of the
    three invariants (no duplicate id in a row, no substitution to a
    non-resident expert, no partial substitution of a source expert).

    Preserved exactly from the host planner:
      * boundary = the (k+1)-th best biased score per row, clamped at 0;
      * a reference is substitutable only if its own score is below
        ``boundary * (1 + threshold)`` (the low-confidence test);
      * candidates must be resident, inside ``[lower, boundary]``, and not
        already selected by that token;
      * scarce candidates go to the LOWEST-scoring references first, which is
        what the host's ascending-max-score ``source_order`` does and what
        SMoE means by substituting the least important experts;
      * ATOMICITY: a source expert is substituted for ALL of its references or
        for none. Load-bearing, not cosmetic — if even one reference to expert
        e survives, e must still be paged in, so substituting the others buys
        zero transfer and costs accuracy.

    Deliberately different: the host consumes candidates group by group, so a
    source expert doomed by a reference in ANOTHER row can free up a candidate
    that a later group then uses. That is inherently sequential. The
    ``eligible`` prepass below recovers the order-independent part of it (a
    source with any high-confidence reference is doomed no matter what the
    candidates are); the rest is the 0-2% divergence in the table above.
    """
    n_tokens, top_k = topk_ids.shape
    num_experts = router_logits.shape[-1]
    if n_tokens == 0 or top_k == 0 or top_k >= num_experts:
        return topk_ids

    scores = _expert_routing_scores(router_logits.to(torch.float32), scoring_func)
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.to(torch.float32).unsqueeze(0)

    # topk(k+1) instead of a full sort. The host planner sorts all E
    # columns to read one element; only the (k+1)-th largest is ever used.
    boundary = scores.topk(top_k + 1, dim=-1).values[:, top_k]
    boundary = boundary.clamp(min=0).unsqueeze(1)  # [n, 1]
    upper = boundary * (1.0 + expert_substitution_threshold)
    lower = boundary * (1.0 - expert_substitution_threshold)

    # Pruning uses -1 as its invalid route ID. Clamp before gather/scatter and
    # keep a separate mask so invalid routes never participate in substitution.
    valid_routes = (topk_ids >= 0) & (topk_ids < num_experts)
    safe_ids = topk_ids.clamp(min=0, max=num_experts - 1).long()
    ids_flat = safe_ids.reshape(-1)
    selected = scores.gather(1, safe_ids)  # [n, k]
    low_confidence = (selected < upper) & valid_routes  # [n, k]
    miss = valid_routes & (log2phy.gather(0, ids_flat).reshape(n_tokens, top_k) < 0)

    def _any_per_source(flag: torch.Tensor) -> torch.Tensor:
        """[n, k] bool -> [n, k] bool: True where this position's SOURCE
        expert has the flag set at any of its positions. Dense [E]
        accumulator rather than a boolean mask over topk_ids[flag], so the
        shape stays static under graph capture."""
        acc = torch.zeros(num_experts, dtype=torch.float32, device=scores.device)
        acc.scatter_add_(0, ids_flat, flag.to(torch.float32).reshape(-1))
        return acc.gather(0, ids_flat).reshape(n_tokens, top_k) > 0

    # the host blocks a source expert outright if ANY of its references is
    # high-confidence, BEFORE looking for candidates
    eligible = miss & low_confidence & ~_any_per_source(miss & ~low_confidence)

    resident = (log2phy >= 0).unsqueeze(0)  # [1, E]
    chosen_count = torch.zeros_like(scores, dtype=torch.float32)
    chosen_count.scatter_add_(1, safe_ids, valid_routes.to(torch.float32))
    chosen = chosen_count > 0
    in_band = (scores >= lower) & (scores <= boundary)
    candidate = in_band & resident & ~chosen  # [n, E]
    candidate_scores = scores.masked_fill(~candidate, float("-inf"))
    cand_vals, cand_ids = candidate_scores.topk(top_k, dim=-1)  # [n, k]

    # rank the eligible references in each row by ASCENDING selected
    # score, so the r-th least important reference takes the r-th best candidate
    order_key = selected.masked_fill(~eligible, float("inf"))
    position = torch.arange(top_k, device=scores.device)
    strictly_lower = order_key.unsqueeze(1) < order_key.unsqueeze(2)
    tie_break = (order_key.unsqueeze(1) == order_key.unsqueeze(2)) & (position.view(1, 1, -1) < position.view(1, -1, 1))
    rank = (strictly_lower | tie_break).sum(dim=2).clamp(max=top_k - 1)

    substitute_id = cand_ids.gather(1, rank)  # [n, k]
    has_candidate = torch.isfinite(cand_vals.gather(1, rank))

    ok = eligible & has_candidate
    blocked = _any_per_source(miss & ~ok)
    return torch.where(ok & ~blocked, substitute_id.to(topk_ids.dtype), topk_ids)
