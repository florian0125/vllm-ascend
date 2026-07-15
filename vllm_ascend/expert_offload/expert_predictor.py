"""Pluggable next-layer expert predictors for the MoE offload manager.

This module owns the *shared* predictor machinery only:
  * the ``ExpertPredictor`` base class and the output contract,
  * a string-keyed registry (``register_predictor`` / ``make_predictor`` /
    ``valid_predictor_names``),
  * the default ``FATEPredictor`` (cross-layer gate),
  * the generic ``LearnedNPUPredictor`` base for trained, on-NPU predictors,
    plus the per-architecture heads (``_HEAD_BUILDERS``) and the driver context
    (``AIPredictCtx``).

Concrete *learned / AI* predictors do NOT live here. Each one is a single file
in the ``vllm_ascend.ai_predictors`` package and registers itself with
``@register_predictor("<name>")``. That package is imported lazily (see
``_ensure_predictors_loaded``) so adding a predictor never touches this file and
never creates an import cycle. See the worked example below ``LearnedNPUPredictor``.

The default predictor implements the "Fate" cross-layer-gate method
(arXiv:2502.12224): take the current layer's hidden states as a cheap proxy for
the next layer's router input, run the next layer's gate, and top-k the result.

HARD OUTPUT CONTRACT (FATE-style predictors — do not break)
-----------------------------------------------------------
``predict_from_device`` / ``predict_from_host`` MUST return ``set[int] | None``:
  * a set of predicted next-layer expert ids, or
  * None to skip prefetch this step (last layer, or prediction impossible).
The downstream ``_do_prefetch`` consumes a plain ``set[int]``. A predictor that
computes on NPU must do its own ``.tolist()`` / ``set(...)`` at the boundary and
return a CPU set. (Learned NPU predictors are driven differently — see below —
and leave these two methods inert.)

TWO PREDICTOR FAMILIES
----------------------
1. FATE-style (``ExpertPredictor`` subclass): implements
   ``predict_from_device`` / ``predict_from_host``; driven by the manager's
   ``trigger_next_layer_prefetch`` worker. ``FATEPredictor`` is the example.
2. Learned on-NPU (``LearnedNPUPredictor`` subclass, in ``ai_predictors/``):
   weights live on HBM; prediction runs on the prefetch stream from tensors the
   model-forward hooks capture (``AIPredictCtx``). Flagged by
   ``uses_model_forward_driver = True`` so the manager runs the AI driver and
   skips the FATE trigger; ``predict_from_device/host`` stay inert.

What a predictor may read (via ``self.mgr``, the manager back-reference):
  * ``self.mgr.moe_layers``        — list of MoE layers (and its length).
  * ``self.mgr.topk``              — experts per token.
  * ``self.mgr._gate_weights_cpu`` — fp32 CPU gate weights per layer (FATE).
  * ``self.mgr.offload_config``    — e.g. ``expert_predictor_ckpt`` for learned nets.
  * ``self.mgr._npu_device``       — device for NPU-side predictors.

CAVEATS (FATE-style predictors)
-------------------------------
  * Graph mode stages only ``hidden_states`` for ``predict_from_host``; a
    predictor needing other staged inputs works in eager mode but needs an extra
    staging hook for graph mode.
  * ``predict_*`` runs on the prefetch worker / graph host-callback thread, never
    the main thread; a STATEFUL predictor must guard its own state.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

if TYPE_CHECKING:
    # Type-only import: avoids any runtime import cycle with the manager.
    from vllm_ascend.expert_offload.expert_offload_manager import (
        ExpertOffloadManager,
    )

logger = init_logger(__name__)

# Package holding the concrete learned/AI predictors (one module per predictor).
# Imported lazily by _ensure_predictors_loaded() — NEVER at module import time —
# because those modules import base classes from THIS module.
_AI_PREDICTORS_PACKAGE = "vllm_ascend.ai_predictors"


# ---------------------------------------------------------------------------
# Base class + registry
# ---------------------------------------------------------------------------
class ExpertPredictor(ABC):
    """Base class for next-layer expert predictors.

    See the module docstring for the output contract and the guide to adding
    new predictors. Subclasses get a read-only back-reference to the owning
    ExpertOffloadManager as ``self.mgr``.
    """

    # Learned predictors set this True so the manager runs the model-forward AI
    # driver (and gates the FATE trigger off). FATE-style predictors leave it False.
    uses_model_forward_driver = False

    def __init__(self, mgr: "ExpertOffloadManager") -> None:
        self.mgr = mgr

    @abstractmethod
    def predict_from_device(
        self, layer_idx: int, hidden_states: torch.Tensor
    ) -> set[int] | None:
        """Predict next layer's experts from an on-device hidden_states tensor.

        Called on the prefetch worker thread (eager mode) after the current
        layer's GMM has completed, so a blocking ``.cpu()`` here is safe.
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
        Called once from the manager's finalize path.
        """
        return None


# predictor name (the `expert_predictor` config string) -> predictor class.
# Populated by @register_predictor at import. FATE (below) registers eagerly;
# learned predictors register when the ai_predictors package is lazily imported.
_PREDICTOR_REGISTRY: dict[str, type[ExpertPredictor]] = {}

_ai_predictors_loaded = False


def register_predictor(name: str):
    """Class decorator: register a predictor implementation under config `name`.

    `name` is exactly the string put in the ``expert_predictor`` config. Adding a
    predictor is just this decorator on a new class in ``ai_predictors/`` — no
    central enum to edit.
    """

    def _decorator(cls: type[ExpertPredictor]) -> type[ExpertPredictor]:
        existing = _PREDICTOR_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"duplicate expert_predictor name {name!r}: "
                f"{existing.__name__} vs {cls.__name__}")
        _PREDICTOR_REGISTRY[name] = cls
        return cls

    return _decorator


def _ensure_predictors_loaded() -> None:
    """Import the ai_predictors package once so every concrete predictor's
    ``@register_predictor`` runs. Lazy (not at module import) to avoid a cycle:
    those modules import this module's base classes. Idempotent."""
    global _ai_predictors_loaded
    if _ai_predictors_loaded:
        return
    try:
        importlib.import_module(_AI_PREDICTORS_PACKAGE)
    except ModuleNotFoundError as exc:
        # Only swallow "the package itself is absent" (FATE-only deployment).
        # A genuine missing import inside a predictor module must propagate.
        if exc.name != _AI_PREDICTORS_PACKAGE:
            raise
        logger.debug("[expert-predictor] %s not present; built-in predictors only",
                     _AI_PREDICTORS_PACKAGE)
    _ai_predictors_loaded = True


def make_predictor(name: str, mgr: "ExpertOffloadManager") -> ExpertPredictor:
    """Construct the predictor selected by config string `name`.

    Raises ValueError on an unknown name so a config typo fails fast at manager
    construction. Triggers lazy registration of the ai_predictors package first.
    """
    _ensure_predictors_loaded()
    cls = _PREDICTOR_REGISTRY.get(name)
    if cls is None:
        valid = ", ".join(repr(n) for n in valid_predictor_names())
        raise ValueError(
            f"Unknown expert_predictor {name!r}; valid values: {valid}")
    return cls(mgr)


def valid_predictor_names() -> list[str]:
    """The registered predictor names accepted by the ``expert_predictor`` config.
    Single source of truth for config validation + its error message — register a
    predictor with ``@register_predictor`` and it shows up here automatically."""
    _ensure_predictors_loaded()
    return sorted(_PREDICTOR_REGISTRY)


# ---------------------------------------------------------------------------
# Default predictor: FATE cross-layer gate
# ---------------------------------------------------------------------------
@register_predictor("fate")
class FATEPredictor(ExpertPredictor):
    """Cross-layer-gate predictor (Fate, arXiv:2502.12224) — the default.

    Uses the current layer's hidden_states as an approximation of the next
    layer's input, runs the next layer's gate, and applies a simplified
    softmax + top-k (instead of the full grouped_topk) for speed. Misses are
    handled by the reactive fallback in ``update_weights()``. Reads gate weights /
    topk / moe_layers from the manager; ownership of those stays on the manager so
    this stays behavior-preserving and minimal.
    """

    def predict_from_device(
        self, layer_idx: int, hidden_states: torch.Tensor
    ) -> list[int] | None:                     # ordered list (was set[int])
        # Early-out BEFORE the .cpu() (last layer / missing gate) so the last
        # layer never pays an unnecessary D2H.
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
    ) -> list[int] | None:                     # CHANGE: ordered list (was set[int])
        next_idx = layer_idx + 1
        if next_idx >= len(self.mgr.moe_layers):
            return None  # last layer — nothing to prefetch
        if next_idx >= len(self.mgr._gate_weights_cpu):
            return None
        gate_w = self.mgr._gate_weights_cpu[next_idx]
        if gate_w is None:
            return None

        # INTENTIONAL SIMPLIFICATION. ... (existing note unchanged) ...
        # hs_cpu is fp32 on CPU; gate_w is fp32 on CPU.
        router_logits = F.linear(hs_cpu, gate_w)  # [num_tokens, n_experts]
        # Simplified routing: softmax + topk (approximation).
        probs = router_logits.softmax(dim=-1)
        _, topk_ids = probs.topk(self.mgr.topk, dim=-1)
        # return predicted experts ORDERED by DESC aggregate score
        uniq = torch.unique(topk_ids)
        agg = probs.amax(dim=0)                                      # [E] best prob per expert
        order = uniq[torch.argsort(agg[uniq], descending=True)]
        return order.tolist()

    def register_from_model(self, model) -> None:
        # FATE's parameters are the per-layer gate weights, kept on CPU (its
        # prediction runs on CPU). Delegates to the manager's FATE-specific helper
        # register_gate_weights(), which populates self.mgr._gate_weights_cpu (+ an
        # NPU mirror) that predict_* read above. This CPU gate registration only
        # runs because FATE is active — a different predictor's register_from_model
        # loads its own weights (e.g. an NN onto HBM) and never calls this.
        self.mgr.register_gate_weights(model)


# ---------------------------------------------------------------------------
# Learned-predictor infrastructure (shared by every ai_predictors/ module)
# ---------------------------------------------------------------------------
class _PredictorHead:
    """Per-layer forward for ONE trained architecture. Owns its params (fp32,
    HBM). Params carry a leading L dim; forward() slices head_idx."""

    def load(self, params: dict, dev, dtype) -> None:
        raise NotImplementedError

    def forward(self, head_idx: int, x_n: torch.Tensor) -> torch.Tensor:
        """x_n: [n, in_dim] (z-scored). Returns logits [n, E]."""
        raise NotImplementedError


class _LowRankHead(_PredictorHead):
    """LowRankProbe: in -> r -> width -> E, LN + GELU (dropout is eval no-op).
    params: Wd[L,in,r], W1[L,r,w], b1[L,w], Wo[L,w,E], bo[L,E]."""

    def load(self, params, dev, dtype):
        g = lambda k: params[k].to(device=dev, dtype=dtype).contiguous()
        self.Wd = g("Wd"); self.W1 = g("W1"); self.b1 = g("b1")
        self.Wo = g("Wo"); self.bo = g("bo")
        self.width = self.W1.shape[-1]

    def forward(self, head_idx, x_n):
        z = x_n @ self.Wd[head_idx]                          # [n, r]
        h = z @ self.W1[head_idx] + self.b1[head_idx]         # [n, w]
        h = F.layer_norm(h, (self.width,))
        h = F.gelu(h)
        return h @ self.Wo[head_idx] + self.bo[head_idx]      # [n, E]
    
    
class _TwoTowerHead(_PredictorHead):
    """TwoTower / retrieval head: per-layer query projection q = LN(x·P + bp)
    and learned per-layer expert embeddings Emb[E,d]; score(e) = q · Emb_e.
    Mirrors moe_predictor_study_mode2_pa.TwoTowerProbe.forward (study lines
    718-737), sliced per head_idx. params: P[L,in,d], bp[L,d], Emb[L,E,d], bo[L,E]."""

    def load(self, params, dev, dtype):
        g = lambda k: params[k].to(device=dev, dtype=dtype).contiguous()
        self.P = g("P"); self.bp = g("bp")
        self.Emb = g("Emb"); self.bo = g("bo")
        self.d = self.P.shape[-1]

    def forward(self, head_idx, x_n):
        q = x_n @ self.P[head_idx] + self.bp[head_idx]        # [n, d]
        q = F.layer_norm(q, (self.d,))
        # In study: einsum("nld,led->nle"); per-head: q[n,d] @ Emb[h][E,d]^T -> [n, E]
        return q @ self.Emb[head_idx].t() + self.bo[head_idx]  # [n, E]


# arch name (checkpoint meta["arch"]) -> head class. Add new archs HERE only.
_HEAD_BUILDERS: dict[str, type[_PredictorHead]] = {
    "lowrank": _LowRankHead,
    "twotower": _TwoTowerHead,
    # "mlp": _MLPHead, "resid": _ResidHead, "twotower": _TwoTowerHead, ...
}


def _build_head(arch: str) -> _PredictorHead:
    if arch not in _HEAD_BUILDERS:
        raise ValueError(
            f"unknown predictor arch {arch!r}; have {list(_HEAD_BUILDERS)}")
    return _HEAD_BUILDERS[arch]()


@dataclass
class AIPredictCtx:
    """Tensors the model-forward driver captures and passes to a learned
    predictor. Filled per the predictor's ``INPUT_SPEC``; unused fields stay None.
    Add a field here (+ one capture in the manager's ``ai_predict_start``) when a
    composition needs more than pre_attn[ℓ] + router_input[ℓ-1]."""

    layer_idx: int                                  # predictor head index
    n_tokens: int
    pre_attn: torch.Tensor | None = None            # pre_attn[ℓ]        [n, H]
    router_input_prev: torch.Tensor | None = None   # router_input[ℓ-1] [n, H] | None@head0
    router_input: torch.Tensor | None = None
    # the PREVIOUS decode token's router logits at THIS layer, i.e. router_logits[t-1, ℓ] (ATOM_SPEC["ptlg"] = ("lg", False,
    # True, True): logit stream, NO layer shift, one token back)
    router_logits_prev: torch.Tensor | None = None  # router_logits[t-1, ℓ] [n, E]


class LearnedNPUPredictor(ExpertPredictor):
    """Generic base for trained predictors that run on the NPU (weights on HBM).

    Owns what every learned predictor shares: loading {params, mu, sd, meta}
    onto HBM, selecting the per-arch head (meta["arch"] -> _HEAD_BUILDERS), input
    z-scoring, and the MoE-index -> head-index offset mapping. A subclass defines
    ONLY the input composition: ``IN_FEATURES``, ``INPUT_SPEC`` (which
    ``AIPredictCtx`` fields it needs, so the driver knows what to capture), and
    ``assemble()`` (the exact training concat order). Learned predictors are
    driven by the model-forward hooks, not the FATE trigger — hence
    ``uses_model_forward_driver = True`` and predict_from_device/host are inert.
    """

    IN_FEATURES: int = 1                       # H-blocks in the concatenated input
    E_FEATURES: int = 0                        # E-dim (router-logit) blocks in the concatenated input.
    INPUT_SPEC: frozenset[str] = frozenset()   # AIPredictCtx fields consumed
    uses_model_forward_driver = True           # manager flag: run the AI driver
    # when True the manager runs the *next-layer* driver (capture pre_attn
    # before attn; predict n+1 + prefetch from fused_moe after on-demand load).
    # Default False = the original *current-layer* driver (predict ℓ before attn).
    predicts_next_layer = False

    def __init__(self, mgr):
        super().__init__(mgr)
        self.ready = False
        self.L = self.E = self.in_dim = self.H = 0
        self.top_k = 0
        self._dev = None
        self._head: _PredictorHead | None = None
        self.mu = self.sd = None
        self.layer_offset = 0          # leading MoE layers this checkpoint doesn't cover

    @property
    def input_spec(self) -> frozenset:
        return self.INPUT_SPEC

    def register_from_model(self, model) -> None:
        ckpt_path = getattr(self.mgr.offload_config, "expert_predictor_ckpt", None)
        if not ckpt_path:
            raise ValueError(f"{type(self).__name__} requires expert_predictor_ckpt")
        dev = next(model.parameters()).device
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        meta = ckpt["meta"]
        self.L = int(meta["L"]); self.E = int(meta["E"])
        self.in_dim = int(meta["in_dim"])
        self.top_k = int(meta["top_k"])
        
        self.H = self.mgr.moe_layers[0].hidden_size
        expect = self.IN_FEATURES * self.H + self.E_FEATURES * self.E
        if self.in_dim != expect:
            raise ValueError(
                f"{type(self).__name__}: checkpoint in_dim={self.in_dim}, but this "
                f"composition needs IN_FEATURES({self.IN_FEATURES})*H({self.H}) + "
                f"E_FEATURES({self.E_FEATURES})*E({self.E}) = {expect}. Either "
                f"IN_FEATURES/E_FEATURES don't describe the checkpoint's atom list, "
                f"or it was trained on a model with a different hidden_size.")
    
        self._dev = dev
        self._head = _build_head(meta["arch"])
        self._head.load(ckpt["params"], dev, torch.float32)
        self.mu = ckpt["mu"].to(device=dev, dtype=torch.float32)   # [1, L, in_dim]
        self.sd = ckpt["sd"].to(device=dev, dtype=torch.float32)
        # The checkpoint covers the LAST L MoE layers (front dense/"hash" layers
        # were dropped from the training dump). The model may have MORE MoE layers
        # than L (e.g. 43 vs 40), so map the manager's MoE index -> head via this
        # offset; the leading `layer_offset` layers get no AI prefetch.
        n_moe = len(self.mgr.moe_layers)
        self.layer_offset = n_moe - self.L
        if self.layer_offset < 0:
            raise ValueError(
                f"{type(self).__name__}: checkpoint L={self.L} exceeds model MoE "
                f"layers {n_moe}")
            
        # the head's output width E = meta["E"] is the expert-id space this
        # predictor selects over (topk in _ai_select_topk). If it doesn't match the
        # model's physical expert count, predicted ids can fall outside the per-layer
        # CPU expert buffers and _do_prefetch would IndexError (now filtered there).
        nt = self.mgr.num_total_experts
        if nt is not None and self.E != nt:
            logger.warning(
                "[AI-PRED] %s checkpoint E=%d != model num_total_experts=%d; "
                "predicted ids outside [0,%d) will be dropped by _do_prefetch. If "
                "unexpected, this checkpoint was trained for a different expert "
                "space and its predictions won't be meaningful for this model.",
                type(self).__name__, self.E, nt, nt)    
            
        self.ready = True
        logger.info("[AI-PRED] %s arch=%s in_dim=%d (H=%d) L=%d E=%d top_k=%d "
                    "covers MoE layers [%d,%d) on %s",
                    type(self).__name__, meta["arch"], self.in_dim, self.H,
                    self.L, self.E, self.top_k, self.layer_offset, n_moe, dev)

    def head_index(self, moe_idx: int) -> int | None:
        """Manager MoE-layer index -> this predictor's head index, or None if the
        layer isn't covered (the leading ``layer_offset`` dense/hash layers)."""
        head = moe_idx - self.layer_offset
        return head if 0 <= head < self.L else None

    def predict_logits_npu(self, head_idx: int, ctx: AIPredictCtx) -> torch.Tensor:
        # head_idx is the predictor's own layer index (0..L-1), already mapped
        # from the manager's MoE index by the driver via head_index().
        x = self.assemble(head_idx, ctx)                       # [n, in_dim]
        x_n = (x - self.mu[0, head_idx]) / self.sd[0, head_idx]
        return self._head.forward(head_idx, x_n)               # [n, E]

    @abstractmethod
    def assemble(self, layer_idx: int, ctx: AIPredictCtx) -> torch.Tensor:
        """Build the (pre-z-score) input x [n, in_dim] from ctx, in the EXACT
        order the checkpoint was trained with."""

    def predict_from_device(self, layer_idx, hidden_states):   # inert (not FATE)
        return None

    def predict_from_host(self, layer_idx, hs_cpu):            # inert (not FATE)
        return None


# ===========================================================================
# HOW TO ADD A NEW AI PREDICTOR  (keep concrete predictors in the
# vllm_ascend/ai_predictors/ package — ONE FILE PER PREDICTOR, never here)
# ===========================================================================
# 1. Drop a new file in vllm_ascend/ai_predictors/, e.g. mode3_triplet.py. The
#    package __init__ auto-imports every module in the folder, so the
#    @register_predictor decorator runs with NO other edit (no central enum).
# 2. Subclass LearnedNPUPredictor and define ONLY the input composition:
#       IN_FEATURES  — number of H-blocks concatenated into the input,
#       INPUT_SPEC   — which AIPredictCtx fields the driver must capture,
#       assemble()   — the exact training concat order -> [n, in_dim].
#    Everything else (ckpt load to HBM, head dispatch, z-scoring, the
#    layer-offset mapping, the on-NPU driver) is inherited.
# 3. Select it at runtime: expert_predictor = "<your name>"  (+ the checkpoint
#    path in expert_predictor_ckpt). valid_predictor_names() lists it for free.
#
#    # vllm_ascend/ai_predictors/mode3_triplet.py
#    import torch
#    from vllm_ascend.expert_offload.expert_predictor import (
#        AIPredictCtx, LearnedNPUPredictor, register_predictor,
#    )
#
#    @register_predictor("mode3_triplet")
#    class Mode3TripletPredictor(LearnedNPUPredictor):
#        IN_FEATURES = 3
#        INPUT_SPEC = frozenset({"pre_attn", "router_input_prev"})  # + any new fields
#        def assemble(self, layer_idx, ctx):
#            ...                          # cat in the trained order -> [n, in_dim]
#
#  * Need a tensor beyond pre_attn[ℓ] / router_input[ℓ-1]?  Add ONE field to
#    AIPredictCtx (above) and ONE capture in the manager's ai_predict_start, then
#    name it in INPUT_SPEC so the driver fills it.
#  * Need new head math (checkpoint meta["arch"] != "lowrank")?  Add a
#    _PredictorHead subclass + one line in _HEAD_BUILDERS — no predictor change.
#  * FATE-style (CPU, gate-based) predictors instead subclass ExpertPredictor and
#    implement predict_from_device/host (see FATEPredictor); they may also live in
#    ai_predictors/ and register the same way.
# ===========================================================================