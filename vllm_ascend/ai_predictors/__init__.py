"""Concrete learned / AI expert predictors — ONE module per predictor.

Each module subclasses
``vllm_ascend.expert_offload.expert_predictor.LearnedNPUPredictor`` (or, for a
CPU/gate-based predictor, ``ExpertPredictor``) and registers itself with
``@register_predictor("<name>")``.

This ``__init__`` auto-imports every module in the package, so dropping a new
file is enough to register a new predictor — no edits here, no central enum.

The package is imported lazily by ``expert_predictor.make_predictor()`` /
``valid_predictor_names()`` (never at ``expert_predictor`` import time), so the
predictor modules — which import base classes from ``expert_predictor`` — do not
create an import cycle.
"""

from __future__ import annotations

import importlib
import pkgutil

# Auto-discover and import every predictor module in this package. Each import
# runs that module's @register_predictor decorator. Modules whose names start
# with "_" are treated as private helpers and skipped.
for _module in pkgutil.iter_modules(__path__):
    if not _module.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module.name}")

del importlib, pkgutil