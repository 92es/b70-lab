# Rejected ideas (with numbers)

Things that were built or configured, measured, and dropped. Machine: 4x B70 (EPYC host) unless it says 2x B70
(earlier Ryzen host). "Fork rNN" = my llama.cpp fork runtime, see [../llama.cpp-sycl/](../llama.cpp-sycl/).

## Execution model

| Idea | Result |
|---|---|
| SYCL graph replay (record the decode graph once, replay) | DeepSeek V4: 21.9 -> 7.1 tok/s (graph re-recorded every step). Flash-Next with MTP: graphs ran after skipping big flash-attention calls (`patches/r53-to-r55-graph-fa-skip.diff`) but decode was ~3x slower (10-23 vs ~54), from re-finalizing with MTP's changing batch shapes and growing KV. Mistral dense on 1 card: -6% (19.4 -> 18.3). |
| Broad small-op fusion | A launch costs ~1-3 us on this stack. Removing ~1,000 launches per token gave +2% (Flash-Next); ~530 gave +1% (GLM); ~340 gave ~1 ms (DeepSeek V4). Worth doing for specific chains, not as a strategy. |
| GLM `hc` gate fusion | +1%, and perplexity moved 5.5565 -> 5.6031 although the fused kernel's output matched; re-running the skipped intermediates restored it. Reverted, cause not found. |
| `-sm tensor` for GLM-5.3 | runs after three meta-backend fixes, 4.8 tok/s: ~1,000 tiny per-card graphs per token, sync-bound. Maya-M also does not fit. |
| `-sm tensor` on the EPYC host (2 cards, gpt-oss-120b) | ~58 -> 28.7 tok/s vs the Ryzen host; host round trips ~4x slower. Layer split wins on this host. |
| Two server slots (`-np 2`) for Flash-Next | no throughput gain (2 x ~11.5 vs 1 x ~23 tok/s); later device OOM on first request. |
| Level Zero v1 adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=0`) | DeepSeek V4 decode 19.2 -> 16.0. `UR_L0_USE_IMMEDIATE_COMMANDLISTS` 0/1/2: no effect. |
| Host flags `--prio 2/3`, `-t 4`, `--cpu-strict` + `-C` mask | no effect on DeepSeek V4. |
| Expert-only parallelism for DeepSeek V4 Flash 0731 (Strata-style, experts split over 4 cards) | A0 profile, fork r52, 7k context: expert matmuls 14% of decode op time (plain, 102 ms step) / 22% (DSpark verify, 156 ms); dense matmuls ~28-29%; small ops ~50%. Ceiling with zero exchange cost 1.12x / 1.19x (2 cards: 1.08x / 1.12x), below the 1.4x go line. Still open: dense + expert tensor parallel (ceiling ~1.6x, all-reduce cost unmeasured). See [../deepseek-v4/PLAN-expert-parallel.md](../deepseek-v4/PLAN-expert-parallel.md). |

## Kernels

| Idea | Result |
|---|---|
| Grouped expert verification: one work-group per distinct expert in an MTP verify window, to read each expert's weights once (`patches/moe-grouped-verify-attempt.py`) | Flash-Next -5 to -10%. The cache already absorbed the repeated reads; the owner work-group still unpacked weights per token and lost parallelism. Measured overlap: Flash-Next 5-token window 50 assignments / 29.1 distinct experts (~42% redundant), GLM 40 / 28.1 (~30%), DeepSeek V4 4-token 24 / 16.3 (~32%). |
| Shared-weight Q6_K dot (r65) | +0.35%. |
| 16-byte / aligned loads for reordered Q4_0 and Q6_K (like the Q8_0 win) | Qwen3.8 Q6_K end-to-end unchanged; Q4_0 n=8 -10%. The compiler already merged those loads. |
| Unrolling / interleaving rows in the DeepSeek IQ MoE kernel | no gain; the kernel is bound by local-memory loads. |
| IQ2_S codebook in local memory (GLM) | slower than reading it from global memory. |
| MoE rows-per-work-group sweep 1..32 (MXFP4) | no effect. |
| Grouped single-launch Q8_0 mat-vec for DeepSeek `attn_output_a` (plain layout) | same 310 us; the plain Q8_0 kernel was the cost, not the launches. The per-slice reorder that replaced it gave +12%. |
| Chunked dequant in the multi-token MoE prompt path | no further gain (285 / 405 vs 283 / 394 tok/s). |
| Fixing only the `nsm = CU / 16` estimate for Xe2 | the flash-attention split heuristic then picks 32 parallel blocks, which is slower than the forced 48. |
| `ext_oneapi_memcpy2d` for `set_tensor_2d` | 227 s vs 17 s for the gather approach: it copies row by row on Level Zero. Pinned staging slots: 9 s. |

## Speculative decoding

| Idea | Result |
|---|---|
| DSpark on DeepSeek V4 with the generic MoE kernels | no gain (17.2 long / 16.1 short vs ~17 plain): a 4-token verify cost ~3.4x one step. Became a +30% win once the IQ kernel made multi-token steps cheap. |
| DSpark depth > 4, p-min 0.7 | no gain / worse. |
| n-gram lookup chained in front of DSpark, real document-checking work | 19-23 tok/s vs 27-30 for DSpark alone (draft acceptance 30-41% vs 88-96%), despite 4-7x on synthetic copy tests. |
| Adaptive back-off for the n-gram drafter (`patches/NOT-ADOPTED-ngram-adaptive.patch`) | costs 10-18% on copy tasks, no chat benefit; a simpler cap/cooldown rule recovered most of the real-work loss (28.0 vs 29.6 tok/s) but was not needed once the lookup entry was dropped for that work. |
| n-gram lookup in front of GLM's MTP | MTP already accepts 98-99% on copy-heavy output; n-gram hurt code edits. |
| Flash-Next MTP n-max 6, p-min 0.7 | worse than n 4 / p 0.1. |
| Upstream sparse flash attention on Flash-Next | -6% at 6k context, 0 at 50k. |

## Configuration

| Idea | Result |
|---|---|
| `-ub 2048` (Flash-Next) | prompt slower than `-ub 1024` (154 / 202 vs 208 / 304 tok/s). DeepSeek V4: `-ub 2048` +7% prompt, -4% decode, +1.7 GB per card; `-ub 4096` worse. |
| `-ub 1024` for GLM | does not fit card 0; prompts 20x slower. |
| `GGML_SYCL_MKL_FA_*` for GLM | no gain; 25-30 GB extra host RAM at load. |
| Maya-S instead of Maya-M (GLM) | more context headroom, +5% perplexity, 20% top-token disagreement. |
| Teamblobfish DeepSeek V4 Q2_K-XL (true K-quant experts) instead of Unsloth's IQ experts | decode same (14.2-14.7), prompt +18%, max 262k context; not worth switching. |
| `GGML_SYCL_SCALE_SILU=1` on DeepSeek V4 | no effect. |
| `--checkpoint-min-step` to fix prompt-cache misses on recurrent / compressed-state models | no help (tested at 512). The fix that worked is a priming request; see [../hardware-and-windows-notes.md](../hardware-and-windows-notes.md). |
| Hugging Face `hf` / Xet downloads on the Windows box | hang mid-download; a `curl -C -` resume loop works. |
