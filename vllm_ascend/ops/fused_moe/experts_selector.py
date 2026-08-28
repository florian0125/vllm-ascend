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


def substitute_experts_device(
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    log2phy: torch.Tensor,
    expert_substitution_threshold: float = 0.25,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """ device-resident mirror of ``substitute_experts``.

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

    scores = _expert_routing_scores(router_logits.to(torch.float32),
                                    scoring_func)
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.to(
            torch.float32).unsqueeze(0)

    # topk(k+1) instead of a full sort. The host planner sorts all E
    # columns to read one element; only the (k+1)-th largest is ever used.
    boundary = scores.topk(top_k + 1, dim=-1).values[:, top_k]
    boundary = boundary.clamp(min=0).unsqueeze(1)                  # [n, 1]
    upper = boundary * (1.0 + expert_substitution_threshold)
    lower = boundary * (1.0 - expert_substitution_threshold)

    ids_long = topk_ids.long()
    ids_flat = ids_long.reshape(-1)
    selected = scores.gather(1, ids_long)                          # [n, k]
    low_confidence = selected < upper                              # [n, k]
    miss = log2phy.gather(0, ids_flat).reshape(n_tokens, top_k) < 0

    def _any_per_source(flag: torch.Tensor) -> torch.Tensor:
        """[n, k] bool -> [n, k] bool: True where this position's SOURCE
        expert has the flag set at any of its positions. Dense [E]
        accumulator rather than a boolean mask over topk_ids[flag], so the
        shape stays static under graph capture."""
        acc = torch.zeros(num_experts, dtype=torch.float32,
                          device=scores.device)
        acc.scatter_add_(0, ids_flat, flag.to(torch.float32).reshape(-1))
        return acc.gather(0, ids_flat).reshape(n_tokens, top_k) > 0

    # the host blocks a source expert outright if ANY of its references is
    # high-confidence, BEFORE looking for candidates
    eligible = miss & low_confidence & ~_any_per_source(miss & ~low_confidence)

    resident = (log2phy >= 0).unsqueeze(0)                         # [1, E]
    chosen = torch.zeros_like(scores, dtype=torch.bool)
    chosen.scatter_(1, ids_long,
                    torch.ones_like(ids_long, dtype=torch.bool))
    in_band = (scores >= lower) & (scores <= boundary)
    candidate = in_band & resident & ~chosen                       # [n, E]
    candidate_scores = scores.masked_fill(~candidate, float("-inf"))
    cand_vals, cand_ids = candidate_scores.topk(top_k, dim=-1)     # [n, k]

    # rank the eligible references in each row by ASCENDING selected
    # score, so the r-th least important reference takes the r-th best candidate
    order_key = selected.masked_fill(~eligible, float("inf"))
    position = torch.arange(top_k, device=scores.device)
    strictly_lower = order_key.unsqueeze(1) < order_key.unsqueeze(2)
    tie_break = ((order_key.unsqueeze(1) == order_key.unsqueeze(2))
                 & (position.view(1, 1, -1) < position.view(1, -1, 1)))
    rank = (strictly_lower | tie_break).sum(dim=2).clamp(max=top_k - 1)

    substitute_id = cand_ids.gather(1, rank)                       # [n, k]
    has_candidate = torch.isfinite(cand_vals.gather(1, rank))

    ok = eligible & has_candidate
    blocked = _any_per_source(miss & ~ok)
    return torch.where(ok & ~blocked,
                       substitute_id.to(topk_ids.dtype), topk_ids)
