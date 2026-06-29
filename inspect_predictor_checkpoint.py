"""Diagnose the mode2 pa+prevhs checkpoint vs the deploy assumptions.

Run on the box that has the .ckpt (CPU is fine):

    python inspect_predictor_ckpt.py \
        /home/keyi/code/moe_offload/hq_v5/mode2_pa/learned_predictor_study_mode2_pa.html.ckpts_260619/pa+prevhs_prompt_std_dist_lowrank_w=2048.pt

It answers three questions:
  (1) Are the weights / mu / sd shaped the way the deploy code expects?
  (2) Is `pre_attn` a LayerNorm OUTPUT (capture input_layernorm out, deepseek_v4
      line 861 — current) or the PRE-norm residual (capture `residual`, line 859)?
      We use the prevhs half as a built-in reference: it is router_input =
      post_attention_layernorm OUTPUT (a known LayerNorm output). If the pre_attn
      half has a similar O(1) per-feature std and is roughly flat across layers,
      pre_attn is also a LayerNorm output (current capture is correct). If the
      pre_attn half std is much larger and GROWS with layer depth, pre_attn is the
      pre-norm residual stream -> the current capture feeds the wrong tensor.
  (3) What loss/standardize/top_k the checkpoint was trained with.
"""

import sys
import torch


def main(path: str) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    print("=== meta ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    params = ck["params"]
    print("\n=== params (state_dict) ===")
    for k, v in params.items():
        print(f"  {k:>4}: {tuple(v.shape)}  {v.dtype}")
    expected = {"Wd", "W1", "b1", "Wo", "bo"}  # lowrank head keys the deploy reads
    missing = expected - set(params)
    print(f"  lowrank keys present: {sorted(expected & set(params))}"
          + (f"  MISSING {sorted(missing)}" if missing else "  (all present)"))

    mu = ck["mu"].float()      # [1, L, in_dim]
    sd = ck["sd"].float()
    print(f"\n=== mu/sd ===\n  mu {tuple(mu.shape)}  sd {tuple(sd.shape)}")
    if mu.dim() != 3 or mu.shape[1] < 2:
        print("  (input_standardize was False -> mu/sd are scalars; "
              "the deploy must NOT index mu[0, head] in that case)")
        return

    L, in_dim = mu.shape[1], mu.shape[2]
    H = in_dim // 2            # pa+prevhs -> in_dim = 2H
    pa_sd = sd[0, :, :H]       # pre_attn half  (first H)
    prev_sd = sd[0, :, H:]     # router_input[l-1] half (second H) -> KNOWN post-LN output

    def line(name, t):
        per_layer = t.mean(dim=1)             # [L] mean per-feature std at each layer
        lo, mid, hi = per_layer[0], per_layer[L // 2], per_layer[-1]
        print(f"  {name:<14} mean={t.mean():8.3f}  "
              f"per-layer std@[0,mid,-1]=[{lo:7.3f},{mid:7.3f},{hi:7.3f}]  "
              f"growth(hi/lo)={(hi / lo):.2f}x")

    print("\n=== std scale by half (the pre_attn capture-point test) ===")
    line("prevhs (ref)", prev_sd)   # router_input = post-LN output: the reference scale
    line("pre_attn", pa_sd)
    ratio = float(pa_sd.mean() / prev_sd.mean())
    growth = float(pa_sd.mean(dim=1)[-1] / pa_sd.mean(dim=1)[0])
    print(f"\n  pre_attn/prevhs mean-std ratio = {ratio:.2f}")
    if ratio < 3.0 and growth < 3.0:
        print("  => pre_attn looks like a LayerNorm OUTPUT (similar scale to the "
              "post-LN prevhs, roughly flat). Current capture at deepseek_v4:861 "
              "(input_layernorm output) is CORRECT.")
    else:
        print("  => pre_attn half is much larger / grows with depth -> it is the "
              "PRE-norm residual stream. Capture `residual` (deepseek_v4:859), "
              "NOT the input_layernorm output, for ai_predict_start.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python inspect_predictor_ckpt.py <checkpoint.pt>")
    main(sys.argv[1])