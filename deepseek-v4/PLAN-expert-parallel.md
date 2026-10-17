# Plan: expert parallelism for DeepSeek V4 Flash (parked)

Status: **planned, parked** (2026-10-16). No code yet and no gain promised until step 1 is done.

## Idea

Bring Strata-style expert parallelism to DeepSeek V4 Flash in the llama.cpp SYCL fork.

- Strata decodes Qwen3.8-Flash-Next at 108.2 tok/s on 2 cards ([../strata-windows-port/](../strata-windows-port/)).
  Every card computes its share of each layer's experts for every token, so both cards work all the time.
- The fork splits DeepSeek V4 by layers over 4 cards. During single-token decode only one card works at a time, so
  each card sits idle up to ~75% of a decode step.
- The fork's cross-card pipeline overlaps cards for prompt reading (many tokens in flight), not for single-token
  decode.

## Where DeepSeek V4 Flash is now

UD-Q2_K_XL, 4x B70, fork r52 + cross-card pipeline, DSpark draft (details in [README.md](README.md)):

| Measure | Now |
|---|---|
| Decode on long-document work (DSpark) | ~30 tok/s |
| Decode at 59k context | 21-24 tok/s |
| Prompt read | ~450 tok/s at 59k, ~400 at 102k |

## Step 1: measure the ceiling (about an hour; decides whether to continue)

Profile one decode step per op (`GGML_SYCL_PROFILE_OPS=1`) and measure:

- the share of step time in the routed-expert matmuls (`MUL_MAT_ID`), and
- the idle time per card.

Only expert work splits across cards, so the expert share bounds the possible gain. If it is small, stop here.

## Step 2: expert-parallel MoE in the fork (multi-day)

- Each card holds a share of every layer's experts and computes them for every token; partial results are combined
  per layer.
- On Windows there is no IPC or peer access between B70s, so the card-to-card exchange goes through host memory
  inside one shared SYCL context, as in the Strata Windows port (patch `0002`). Strata shows this works at speed on
  these cards.
- Validate like every other change here: `test-backend-ops`, perplexity equal to the layer-split build, fixed-answer
  prompts, then decode A/B on the same harness.

## Caveats

- Part of Strata's speed may come from its smaller quant (IQ3_XXS, ~27% fewer bytes than UD-Q4_K_XL) and from
  model-specific fused kernels, not only from parallelism.
- DeepSeek V4's attention (compressed sparse attention + indexer) and the DSpark draft stay as they are; only the
  MoE part changes.
- No number is promised until step 1.
