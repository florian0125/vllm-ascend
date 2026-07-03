#!/usr/bin/env python3
"""
download_benchmark.py — seed the HF datasets cache for lm-eval, ONE TIME, over a
TLS-intercepting corporate proxy.

Run with HF_HOME pointed at your cache:
    HF_HOME=/home/keyi/llms/benchmarks/keyi python download_benchmark.py

After this succeeds, serve_eval.sh (OFFLINE=1) reads every task from the cache
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

# (task label, repo_id, config-or-None) — repo/config match lm-eval 0.4.12 EXACTLY,
# so the cache keys line up and lm-eval won't re-download.
DATASETS = [
    ("gsm8k",          "openai/gsm8k",              "main"),
    ("mmlu",           "cais/mmlu",                 "all"),           # all 57 subjects in one build
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
for label, repo, cfg in DATASETS:
    try:
        load_dataset(repo, cfg) if cfg else load_dataset(repo)
        print(f"[ok]   {label:16s} {repo}" + (f" [{cfg}]" if cfg else ""))
        ok.append(label)
    except Exception as e:
        print(f"[FAIL] {label:16s} {repo}: {type(e).__name__}: {str(e)[:80]}")
        fail.append(label)

print(f"\n{len(ok)}/{len(DATASETS)} seeded into HF_HOME={os.environ.get('HF_HOME','~/.cache/huggingface')}")
if fail:
    print("failed:", ", ".join(fail), "— fix and re-run; seeded ones are cached and skip fast.")

# HumanEval also needs the code_eval METRIC module (separate from the dataset).
# Seed it here too so offline humaneval runs find it under HF_HOME/modules.
try:
    import evaluate
    evaluate.load("code_eval")
    print("[ok]   code_eval metric module cached (for humaneval)")
except Exception as e:
    print(f"[warn] code_eval metric not cached ({type(e).__name__}); humaneval offline will need it. "
          f"pip install evaluate, then re-run.")

# import os
# # must be set BEFORE huggingface_hub is imported (it builds its httpx client once)
# os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
# os.environ.pop("SSL_CERT_FILE", None)        # a bad/empty bundle crashes httpx before verify is skipped
# os.environ.pop("REQUESTS_CA_BUNDLE", None)
# os.environ.pop("CURL_CA_BUNDLE", None)

# # belt-and-suspenders: also disable at the ssl level, in case a code path
# # constructs its own default context
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# from datasets import load_dataset
# load_dataset("openai/gsm8k", "main")
# load_dataset("cais/mmlu", "all")
# print("seeded gsm8k + mmlu into", os.environ.get("HF_HOME"))