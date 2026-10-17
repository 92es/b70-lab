# Strata on native Windows

[Strata](https://github.com/hotschmoe/Strata) is a SYCL inference engine with B70-specific work (two-card expert
parallel, MTP, int8 KV). It targets Linux. These three patches make it build and run natively on Windows 11 with
Intel's `icx` compiler, including two-card expert parallel and int8 KV.

- Base: `hotschmoe/Strata` branch `b70-night-20261016` @ `1c360526` (from SergiioB/b70-launcher PR #4).
- Model: ISTA-DASLab GSQ-RCO quants of Qwen3.8-Flash-Next, **IQ3_S** (84 GB, current) and IQ3_XXS (76 GB, earlier),
  sha256 checked; packs + MTP built on the same machine. Cards: two B70s (`level_zero:2,3`).
- Status (2026-10-17): **best setup = IQ3_S, 1 slot x 200k context, int8 KV, prefill chunk 8192, JIT build**
  (patch 0003): decode 97.6 tok/s, prompt read 1,415 tok/s at 33.6k, answers correct. The AOT build with oneAPI 2026.0 miscompiles the int8 attention (see below).
  Not done: soak test, concurrent requests on both slots, AOT with oneAPI 2026.1.

Patches (prepared with AI assistance; Strata is MIT, notice in
[../LICENSES/Strata-LICENSE](../LICENSES/Strata-LICENSE)):

| Patch | What |
|---|---|
| `0001` native Windows build (icx) | CMake: `/fp:precise`, `/Qmkl:sequential`, C-safe warning flags; Level Zero SDK lookup; positioned reads with `ReadFile`; drop a CUDA-only shared-memory query; undo `<windows.h>` macros (`small`, `IN`, `OUT`, `near`, `far`) that clash with engine names; `memmem` fallback |
| `0002` two-card expert parallel through host memory | every GPU shares one `sycl::context`; the shared expert-parallel regions are host USM in that context (one address for both cards). `STRATA_WIN_SHARED_CONTEXT=0` restores one context per device |
| `0003` JIT build option for Xe2 | `STRATA_SYCL_XE2_JIT=ON` makes a JIT build define `STRATA_XMX_SG16` like the bmg AOT build does; `STRATA_SYCL_AOT_OPTIONS` passes extra ocloc options to an AOT build; `setenv` shim for two parity tests |

`configs/` has the engine configs used below (paths replaced with placeholders): `strata-iq3s-200k1-pf8k.json` (the
current best setup), `strata-ep2-jit.json` (IQ3_XXS, int8, 2 x 200k), `strata-ep2-int8-32k.json`, `strata-ep2-fp16-96k.json`.

## Build (JIT)

```
cmake -DSTRATA_SYCL_AOT= -DSTRATA_SYCL_XE2_JIT=ON ...
```

With an empty `STRATA_SYCL_AOT` the device code is compiled by the GPU driver at first use. Here (driver
32.0.101.8976) the engine was ready in under a minute and no startup penalty was seen afterwards.
oneAPI 2026.0 `icx` with the MSVC toolchain environment; the oneAPI runtime DLLs must be next to `strata.exe` or on
`PATH`.

## Results

Same 7-prompt set ([../bench/sbench.py](../bench/)), temperature 0, thinking off, 4x B70 machine, driver 8976.
Strata rows: 2 cards, MTP speculative decoding depth 4.

| Setup | Answers | Decode tok/s | Prompt read | Decode after long prompt |
|---|---|---|---|---|
| llama.cpp fork r62, 4 cards, UD-Q4_K_XL | correct | 60.6 | 748 tok/s @ 33.6k | 28.6 |
| **Strata IQ3_S, int8 KV, 1 x 200k, prefill 8192, JIT** | **correct** | **97.6 (+61%)** | **1,415 tok/s @ 33.6k (1.9x)** | **64.9 (2.3x)** |
| Strata IQ3_S, int8 KV, 2 x 112k, prefill 4096, JIT | correct | 100.8 | 881 tok/s @ 33.6k | 60.7 |
| Strata IQ3_XXS, int8 KV, 2 x 200k, prefill 12288, JIT (superseded by IQ3_S) | correct (identical to llama.cpp on 4 check prompts) | 108.2 (97.7 on a rerun) | 599-678 tok/s @ 33.6k | 58.4-58.7 |
| Strata, fp16 KV, 32k (superseded) | correct | 67.4 (+11%) | 570 tok/s @ 26.8k | 46.4 |
| Strata, int8 KV, 200k, AOT build (oneAPI 2026.0) | **wrong** (garbage) | 96.6 (invalid) | 682 @ 33.6k | 66.6 (invalid) |
| Strata, 1 card, int8 KV, AOT build | wrong | 12-20 | - | - |

The quants differ: Strata runs GSQ-RCO IQ3_S / IQ3_XXS, llama.cpp runs UD-Q4_K_XL. On the 4 check prompts
(`bench/squal.py`) the answers matched llama.cpp.

## Quant quality (IQ3_S vs IQ3_XXS)

Measured in llama.cpp (fork r62), 30 x 512-token chunks, KL divergence against UD-Q4_K_XL as the reference:

| Quant | Perplexity vs Q4 | Mean KLD | Median KLD | Same top token |
|---|---|---|---|---|
| GSQ-RCO IQ3_XXS (76 GB) | +3.5% +- 0.9% | 0.211 | 0.046 | 85.3% |
| **GSQ-RCO IQ3_S (84 GB)** | **+1.0% +- 0.7%** | **0.131** | **0.025** | **88.4%** |

The reference is itself a quant from a different family, so these KLD values overstate the distance to full
precision. IQ3_S is closer to the reference, reads prompts faster in Strata, and decodes ~7-10% slower than IQ3_XXS;
it replaced IQ3_XXS.

## Strata settings that matter

- `STRATA_NIGHT_PREFILL_GROUPED` must be **0 for IQ3_S**: the grouped prefill kernel does not support it (engine
  error "card 0 gate/up type 23", then a restart). IQ3_XXS needs it **on** (off: prompt read 239 tok/s).
- Prompt chunk size matters a lot: IQ3_S with 4k chunks 881 tok/s, with 8k chunks 1,415 tok/s.
- IQ3_S at 2 x 200k and 2 x 131k does not fit on two 32 GB cards; 1 x 200k and 2 x 112k do.

## Prompt chunk size sweep (IQ3_S)

IQ3_S, 1 x 200k, int8 KV, JIT build, 2 cards, driver 8976; `--prefill` = prompt chunk in tokens; same 7-prompt set,
one round each. "Free" = what the engine reported free on the fuller card after loading.

| `--prefill` | Fits | Decode tok/s | Prompt read @ 33.6k | Decode after it |
|---|---|---|---|---|
| 6144 | yes (1,851 MiB free) | 102.3 | 1,247 | 70.7 |
| 7168 | yes (1,835 MiB) | 98.4 | 1,435 | 54.1 |
| **8192 (kept)** | yes (1,815 MiB) | 99.0 | 1,417 | 65.7 |
| 10240 | barely (1,778 MiB) | 35.8 | 517 | 26.2 |
| 12288 | no (card 1 needs 30.46 of 30.16 GiB) | - | - | - |

At 10240 card 1 sits at its memory limit and everything slows ~3x. Decode-after numbers vary about +-8 tok/s run to
run.

Long prompt with 8192: **112,142 tokens read in 78.3 s = 1,433 tok/s**, decode after it 72.2 tok/s. For comparison,
the llama.cpp fork (Flash-Next MTP, 4 cards, cross-card pipeline) read an ~87k prompt at ~685 tok/s.

**Lesson:** leave memory headroom. A config that "fits" with a card at its limit can be 3x slower than one step
smaller.
fp16 KV fits 96k context on two cards (1.6 GB free); 131k does not (card 1 ~0.85 GB short).

**Lesson:** a speed number only counts after an answer check. The broken AOT build looked fast (96.6 tok/s) because
garbage output is easy for the MTP draft to predict, so acceptance went up and inflated tok/s.

## The int8 KV bug: an AOT miscompile (solved)

With the AOT build (oneAPI 2026.0 `ocloc`, device `bmg-g31`), int8 KV gave wrong answers while fp16 KV was correct
with everything else equal. Ruled out on the way: the tokenizer (token ids identical to llama.cpp), the download
(sha256 match), PLE file I/O, the prompt-attention kernel (`STRATA_PROMPT_ATTN_OLD=1` still wrong), the experimental
"night" switches (minimal env still wrong, output "!!!!"). The `kv_q8`, `qsa` and `kv_steps` parity tests passed.

Cause: the AOT build miscompiles the **batched decode attention over int8 KV** used by the MTP verify step.

- `qsa_verify_batch_parity`, "one vs row": half the outputs wrong (including `inf`), 14 failures.
- `kv_hybrid_parity`: batched attention FAIL.
- The same sources JIT-compiled by the Windows driver (32.0.101.8976) pass **bitwise**, and the engine answers
  correctly.
- Adding `-ze-opt-greater-than-4GB-buffer-required` to the AOT build does not help.

Not tested: AOT with oneAPI 2026.1 (the version the upstream B70 work used).

Parity tests with the AOT build: 21 of 23 pass. Failing: `gr_parity` (fused GR read + q8_1, `STRATA_QFUSE` path:
"not written") and `s2_expert_grouped_parity` (the Q2_0 experts it covers are the MTP draft layer only). Not yet
re-run under JIT; they may be the same compiler bug.

## Windows / Level Zero findings

Measured with [../reproducers/ep-probe/](../reproducers/) on two B70s:

- No IPC: `zeMemGetIpcHandle` returns `0x78000003` (unsupported feature).
- `zeDeviceCanAccessPeer` = 0 between B70s.
- A kernel on one card touching the other card's device memory: `DEVICE_LOST`.
- Driver-imported host pointers used from another context: also `DEVICE_LOST`.
- Host USM allocated inside **one** context shared by both cards works for both cards. Patch `0002` is built on that.

So on Windows, card-to-card traffic has to go through host memory. Linux behaviour was not tested here.

## Other notes

- With default settings the engine hangs: verify waits on a host doorbell and hits a timeout. The PR's
  `STRATA_VERIFY_DEVICE_PLAN=1` + `STRATA_VERIFY_NO_HOST=1` avoid it (set in all configs).
