#!/usr/bin/env python3
# =============================================================================
#  demo_infer.py  —  offline sanity check for the moe_offload_v2.0 branch
#
#  Runs prompts through the model with the SAME engine + offload config the
#  benchmark uses, and prints prompt -> output so you can eyeball correctness.
#
#  WHY THE ENGINE KNOBS BELOW MATTER:
#    The W8A8 fused-MoE swiglu-quant kernel (npu_grouped_matmul_swiglu_quant,
#    NZ weights) picks its tiling from the shapes seen at init. If max_model_len
#    is left at the model's native max, the warmup/profiling run hits that kernel
#    with a huge shape and dies with "Parse dynamic kernel config fail /
#    AclOpKernelInit failed" on A5. The benchmark works because it CAPS
#    max_model_len / max_num_batched_tokens. This demo now does the same, so it
#    takes the identical engine path the benchmark does.
#
#  Prereq:  conda activate vllm-ascend && source env_a5.sh
#
#  Examples:
#    # W8A8 model, offload ON, quantization ON (mirrors the benchmark)
#    python demo_infer.py --card 5 --quant ascend
#
#    # your own prompts
#    python demo_infer.py --card 5 --quant ascend \
#        --prompt "Explain MoE in one sentence." --prompt "The capital of France is"
#
#    # real prompts from ShareGPT
#    python demo_infer.py --card 5 --quant ascend \
#        --sharegpt /data/ShareGPT_V3_unfiltered_cleaned_split.json -n 5
#
#    # A/B: prove offload doesn't corrupt output (outputs must match)
#    python demo_infer.py --card 5 --quant ascend                # offload on
#    python demo_infer.py --card 5 --quant ascend --no-offload   # baseline
# =============================================================================
import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description="Offline inference sanity check (offload-aware).")
    # what / where
    p.add_argument("--model", default="vllm-ascend/DeepSeek-V2-Lite-W8A8",
                   help="model path or ModelScope id")
    p.add_argument("--card", default="5", help="NPU id (sets ASCEND_RT_VISIBLE_DEVICES)")
    p.add_argument("--quant", default="", help="quantization: empty = OFF; 'ascend' = W8A8")
    # sampling
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (best for a sanity check)")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1024)
    # engine knobs (defaults mirror the working benchmark)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=1, help="1 exercises the decode/LRC offload path")
    p.add_argument("--max-model-len", type=int, default=2048,
                   help="CAP the context; leaving it at the model native max can break the swiglu-quant kernel")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048,
                   help="cap the warmup/prefill shape (the benchmark sets this)")
    p.add_argument("--expert-parallel", action=argparse.BooleanOptionalAction, default=False,
                   help="match your working serve command if it uses --enable-expert-parallel")
    p.add_argument("--eager", action=argparse.BooleanOptionalAction, default=True)
    # offload feature (defaults mirror bench_official.sh)
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-device-experts", type=int, default=24)
    p.add_argument("--num-device-layers", type=int, default=1)
    p.add_argument("--cache-policy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache-debug", action=argparse.BooleanOptionalAction, default=False,
                   help="per-paging [UPDATE-W] logs (noisy; off for a clean demo)")
    p.add_argument("--cpu-binding", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wprefetch", action=argparse.BooleanOptionalAction, default=False)
    # prompts (priority: --sharegpt > --prompt-file > --prompt > built-in)
    p.add_argument("--prompt", action="append", default=[], help="repeatable; one prompt each")
    p.add_argument("--prompt-file", default=None, help="plain text file, one prompt per line")
    p.add_argument("--sharegpt", default=None, help="ShareGPT_V3_*.json path; samples first human turns")
    p.add_argument("-n", "--num-prompts", type=int, default=4, help="how many prompts to run")
    p.add_argument("--max-prompt-chars", type=int, default=2000, help="skip longer ShareGPT prompts")
    return p.parse_args()


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Q: What is a mixture-of-experts model? A:",
    "Write one sentence about the ocean.",
    "List three prime numbers:",
]


def load_prompts(args):
    if args.sharegpt:
        with open(args.sharegpt, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = []
        for entry in data:
            for turn in entry.get("conversations", []):
                if turn.get("from") in ("human", "user"):
                    text = (turn.get("value") or "").strip()
                    if text and len(text) <= args.max_prompt_chars:
                        prompts.append(text)
                    break
            if len(prompts) >= args.num_prompts:
                break
        if not prompts:
            raise SystemExit(f"No usable prompts found in {args.sharegpt}")
        return prompts
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
        return prompts[: args.num_prompts]
    if args.prompt:
        return args.prompt
    return DEFAULT_PROMPTS[: args.num_prompts]


def build_additional_config(args):
    cfg = {}
    if args.offload:
        cfg["enable_cpu_binding"] = args.cpu_binding
        cfg["expert_offload_config"] = {
            "expert_offload": True,
            "num_device_experts": args.num_device_experts,
            "num_device_layers": args.num_device_layers,
            "cache_policy_enabled": args.cache_policy,
            "cache_debug_log_updates": args.cache_debug,
        }
    if args.wprefetch:
        cfg["weight_prefetch_config"] = {"enabled": True}
    return cfg or None


def main():
    args = parse_args()

    # Must be set BEFORE importing vllm/torch_npu (device visibility + loader).
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.card)
    os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams  # imported after env is set

    prompts = load_prompts(args)
    additional_config = build_additional_config(args)
    offload_on = bool(additional_config and "expert_offload_config" in additional_config)

    if not args.quant and "w8a8" in args.model.lower():
        print("WARN: model name looks W8A8-quantized but --quant is off — did you mean --quant ascend?")

    print("=" * 70)
    print(f"model            : {args.model}")
    print(f"card={args.card}  quant={args.quant or 'none'}  tp={args.tensor_parallel_size}  "
          f"util={args.gpu_mem_util}  ep={args.expert_parallel}  eager={args.eager}")
    print(f"max_num_seqs={args.max_num_seqs}  max_model_len={args.max_model_len}  "
          f"max_num_batched_tokens={args.max_num_batched_tokens}")
    print(f"offload          : {offload_on}")
    if additional_config:
        print(f"additional_config: {json.dumps(additional_config)}")
    print(f"prompts          : {len(prompts)}")
    print("=" * 70)

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        enforce_eager=args.eager,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
    )
    if args.quant:
        llm_kwargs["quantization"] = args.quant
    if args.expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if additional_config is not None:
        llm_kwargs["additional_config"] = additional_config

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )

    outputs = llm.generate(prompts, sampling)

    print("\n" + "#" * 70)
    print("# RESULTS  (eyeball these — coherent continuation = model is working)")
    print("#" * 70)
    for i, out in enumerate(outputs):
        prompt = out.prompt if out.prompt is not None else prompts[i]
        shown = prompt if len(prompt) <= 200 else prompt[:200] + " …"
        print(f"\n[{i}] PROMPT : {shown!r}")
        print(f"[{i}] OUTPUT : {out.outputs[0].text!r}")
    print("\n" + "#" * 70)
    if offload_on:
        print("# offload was ON — the [EXPERT_OFFLOAD HOOK] lines at startup confirm it engaged.")
        print("# tip: rerun with --no-offload and check the outputs match (offload must not change them).")
    print("#" * 70)


if __name__ == "__main__":
    main()