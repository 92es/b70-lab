# gpt-oss-120b on B70s

OpenAI gpt-oss-120b (MXFP4 experts, attention sinks). Status: **live** on fork runtime `r52`, 3 cards, 131k context,
f16 KV, `-sm layer`, n-gram speculative decoding, cross-card pipeline on.

## MXFP4: why it was slow, and the fix (now upstream)

On upstream b11190 (2x B70, Ryzen host), gpt-oss-120b decoded at ~24.6 tok/s with layer split. Two causes:

1. `ggml_backend_sycl_buffer_init_tensor` created the reorder "extra" only for Q4_0 / Q8_0 / Q2_K..Q6_K, so MXFP4
   experts never got the reordered layout and always ran the generic MoE mat-vec. Adding MXFP4 to that list took the
   MoE op from 138 to 36.7 us per call (1 token).
2. E2M1 nibble decode used 8 byte-permutes per int, each emulated with 64-bit shifts. A SWAR decode replaced it
   (bit-exact).

With both, decode went 24.6 -> 69.6 tok/s (2 cards, layer split, `-ts 17,19`, Ryzen host). This became
[ggml-org/llama.cpp#29809](https://github.com/ggml-org/llama.cpp/pull/29809), merged 2026-10-10 (measured on the EPYC host: gpt-oss-20b decode 55.8 -> 106.6, gpt-oss-120b 33.3 -> 67.1). A follow-up found that prompt
processing only got the reordered layout after the first single-token step had run; calling the reorder from the
multi-token path too gave prompt +20-27% in both flash-attention modes.

Note for anyone measuring MoE kernels: `test-backend-ops perf` reuses the same few experts every iteration, so
they sit in L2 and the numbers come out inflated (Q4_K showed "631 GB/s" against a ~608 GB/s peak). It misled my first
estimate here. Trust end-to-end numbers.

## Card count, split mode, host CPU

| Setup (gpt-oss-120b) | Decode tok/s | Notes |
|---|---|---|
| 2 cards, layer, default split (Ryzen) | 24.6 | card 0 over-subscribed: WDDM paged its experts to system RAM |
| 2 cards, layer, `-ts 17,19` (Ryzen) | 33.7 -> 69.6 with MXFP4 fixes | |
| 2 cards, `-sm tensor` (Ryzen) | 40.5 -> 58.2 with MXFP4 fixes | tensor split loses once MoE is cheap |
| 2 cards, `-sm tensor` (EPYC) | ~28.7 | EPYC host round trips ~4x slower; layer split wins on this host |
| 3 cards, layer, 131k f16 KV (EPYC) | ~61 via the server | live layout, 21.3 / 22.1 / 23.8 GB |

2 cards at 80k+ context in layer mode did not fit: the fit step aborted, WDDM spilled ~24 GB to host RAM and decode
fell from 50 to 26 as context grew. Grep the server log for `fit_params` before trusting a fit.

## Cross-card pipeline

r52, 3 cards, `GGML_SYCL_PIPELINE=1 GGML_SYCL_PIPELINE_MAX_QUEUED=4`, `-b 8192`, driver 8805: 46.7k-token prompt
444 -> ~1030 tok/s, 80k 332 -> ~806. Perplexity equal on/off, soak clean. Smoke after promotion: 1275 tok/s.

## Loading with `-sm tensor`

Tensor split used to take ~6 minutes to load gpt-oss-120b (vs ~33 s layer split): the SYCL buffer had no
`set_tensor_2d`, so every row of every split expert tensor was its own blocking copy. Implementing it took the load
from 355 s to 45.5 s on the Ryzen host (later ~265 s -> ~115 s re-measured on a later master). This is
[ggml-org/llama.cpp#30358](https://github.com/ggml-org/llama.cpp/pull/30358), open.

## Things to know

- Do not test 131k f16 KV configurations that do not fit back to back on a machine with little RAM: four such loads
  in a row hard-hung the box when it had 31 GB of RAM (see
  [../hardware-and-windows-notes.md](../hardware-and-windows-notes.md)).
- gpt-oss attention at 32k cost ~54 ms per token on the old b11190 runtime (Qwen 27B: ~25 ms), likely a slow
  flash-attention path for head size 64 + sinks + f16 KV. Not investigated further.
