"""Expert substitution helpers; base routing is provided by router/."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

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

    selected_scores = selection_scores.gather(1, topk_ids.long())
    low_confidence = selected_scores < upper.unsqueeze(1)
    in_cache = log2phy[topk_ids.long()] >= 0
    cached_experts = set(torch.where(log2phy >= 0)[0].tolist())
    references: dict[int, list[tuple[int, int]]] = {}
    for token_idx, position in (~in_cache).nonzero(as_tuple=False).tolist():
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
            max(selected_scores[token_idx, position].item()
                for token_idx, position in references[source_id]),
            -len(references[source_id]),
            source_id,
        ),
    )
    for source_id in source_order:
        source_references = sorted(
            references[source_id],
            key=lambda item: (
                selected_scores[item[0], item[1]].item(), item[0], item[1]),
        )
        if any(not bool(low_confidence[token_idx, position])
               for token_idx, position in source_references):
            blocked[source_id] = True
            continue

        tentative: list[tuple[int, int, int]] = []
        tentative_ids = planned_ids.clone()
        for token_idx, position in source_references:
            token_scores = selection_scores[token_idx]
            in_range = ((token_scores >= lower[token_idx])
                        & (token_scores <= boundary[token_idx]))
            candidates = (set(torch.where(in_range)[0].tolist())
                          & cached_experts)
            candidates.difference_update(
                tentative_ids[token_idx].tolist())
            if not candidates:
                blocked[source_id] = True
                break
            substitute_id = max(
                candidates,
                key=lambda expert_id: (
                    token_scores[expert_id].item(), -expert_id),
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
