#!/usr/bin/env python3
"""
download_benchmark.py — seed the HF datasets cache for lm-eval, ONE TIME, over a
TLS-intercepting corporate proxy.

Run with the cache location pointed at your dir:
    HF_HOME=/home/keyi/llms/benchmarks/keyi python download_benchmark.py
  (or, if your eval script uses HF_DATASETS_CACHE, set that instead/too:
    HF_DATASETS_CACHE=/home/keyi/llms/benchmarks/keyi/hf_cache python download_benchmark.py)

After this succeeds, v4_eval.sh (OFFLINE) reads every task from the cache
with no network.

SSL: the proxy MITMs TLS with a self-signed root that's in no bundle you have,
and huggingface_hub uses httpx (which builds its context via
ssl.create_default_context). We patch THAT below — verified to be the lever that
actually disables httpx verification (ssl._create_default_https_context and
HF_HUB_DISABLE_SSL_VERIFICATION do NOT work for this httpx path).

If you later get the proxy's root CA as a .pem, delete the SSL block and set
SSL_CERT_FILE to a bundle that includes it — that's the clean, verify-on fix.
"""
import os, ssl

# ── SSL bypass (must run BEFORE huggingface_hub is imported) ─────────────────
for _v in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(_v, None)          # an empty/bad bundle crashes httpx before verify is skipped
ssl.create_default_context = ssl._create_unverified_context
# If the mirror itself is the problem (not SSL), uncomment to go direct:
# os.environ.pop("HF_ENDPOINT", None)
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")  # harmless; helps any script-based repo

from datasets import load_dataset

# MMLU is seeded PER SUBJECT. lm-eval's `mmlu` group instantiates 57 separate
# load_dataset("cais/mmlu", "<subject>") calls — config "all" does NOT satisfy
# them (you get: ValueError: Couldn't find cache for cais/mmlu for config
# 'abstract_algebra'). List taken verbatim from lm-eval 0.4.12 tasks/mmlu/default.
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
]

# (task label, repo_id, config-or-None) — repo/config match lm-eval 0.4.12 EXACTLY,
# so the cache keys line up and lm-eval won't re-download.
DATASETS = [
    ("gsm8k",          "openai/gsm8k",              "main"),
    # mmlu handled per-subject below (see MMLU_SUBJECTS)
    ("mmlu_pro",       "TIGER-Lab/MMLU-Pro",        None),            # lm-eval tasks: mmlu_pro, mmlu_pro_<subject>
    ("gpqa_main",      "Idavidrein/gpqa",           "gpqa_main"),     # GATED on HF — needs HF_TOKEN, see note below
    ("gpqa_diamond",   "Idavidrein/gpqa",           "gpqa_diamond"),  # GATED on HF
    ("hellaswag",      "Rowan/hellaswag",           None),
    ("arc_easy",       "allenai/ai2_arc",           "ARC-Easy"),
    ("arc_challenge",  "allenai/ai2_arc",           "ARC-Challenge"),
    ("winogrande",     "allenai/winogrande",        "winogrande_xl"),
    ("piqa",           "baber/piqa",                None),            # lm-eval uses baber/piqa, not ybisk/piqa
    ("boolq",          "aps/super_glue",            "boolq"),         # lm-eval uses super_glue's boolq
    ("openbookqa",     "allenai/openbookqa",        "main"),
    ("truthfulqa",     "truthfulqa/truthful_qa",    "multiple_choice"),
    ("lambada_openai", "EleutherAI/lambada_openai", "default"),
    ("sciq",           "allenai/sciq",              None),
    ("humaneval",      "openai/openai_humaneval",   None),
]

ok, fail = [], []

def seed(label, repo, cfg):
    try:
        load_dataset(repo, cfg) if cfg else load_dataset(repo)
        print(f"[ok]   {label:22s} {repo}" + (f" [{cfg}]" if cfg else ""))
        ok.append(label)
    except Exception as e:
        print(f"[FAIL] {label:22s} {repo}: {type(e).__name__}: {str(e)[:90]}")
        fail.append(label)

# ── MMLU: 57 configs from one repo (repo downloads once, then each config builds) ──
print(f"--- mmlu: seeding {len(MMLU_SUBJECTS)} per-subject configs from cais/mmlu ---")
for i, subj in enumerate(MMLU_SUBJECTS, 1):
    seed(f"mmlu:{subj}", "cais/mmlu", subj)
    if i % 10 == 0:
        print(f"    ...{i}/{len(MMLU_SUBJECTS)}")

# ── everything else ─────────────────────────────────────────────────────────
print("--- other benchmarks ---")
for label, repo, cfg in DATASETS:
    seed(label, repo, cfg)

total = len(MMLU_SUBJECTS) + len(DATASETS)
cache = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_HOME", "~/.cache/huggingface")
print(f"\n{len(ok)}/{total} seeded into {cache}")
if fail:
    print("failed:", ", ".join(fail))
    print("-> re-run after fixing; already-seeded ones are cached and skip fast.")
    if any(f.startswith("gpqa") for f in fail):
        print("-> GPQA is a GATED dataset: accept the terms at")
        print("   https://huggingface.co/datasets/Idavidrein/gpqa  then set HF_TOKEN=<your token>")
        print("   (login once with `huggingface-cli login`, or export HF_TOKEN) and re-run.")

# HumanEval also needs the code_eval METRIC module (separate from the dataset).
# Seed it here too so offline humaneval runs find it under <cache>/modules.
try:
    import evaluate
    evaluate.load("code_eval")
    print("[ok]   code_eval metric module cached (for humaneval)")
except Exception as e:
    print(f"[warn] code_eval metric not cached ({type(e).__name__}); humaneval offline will need it. "
          f"pip install evaluate, then re-run.")