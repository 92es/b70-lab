# Strata on native Windows

[Strata](https://github.com/hotschmoe/Strata) is a SYCL inference engine with B70-specific work (two-card expert
parallel, MTP, int8 KV). It targets Linux. These three patches make it build and run natively on Windows 11 with
Intel's `icx` compiler, including two-card expert parallel and int8 KV.

- Base: `hotschmoe/Strata` branch `b70-night-20261016` @ `1c360526` (from SergiioB/b70-launcher PR #4).
- Model: ISTA-DASLab GSQ-RCO IQ3_XXS quant of Qwen3.8-Flash-Next (sha256 checked); pack + MTP built on the same
  machine. Cards: two B70s (`level_zero:2,3`).
- Status (2026-10-16): **works with int8 KV, 200k context x 2 slots, 2 cards, when built JIT** (patch 0003).
  108.2 tok/s decode, answers correct. The AOT build with oneAPI 2026.0 miscompiles the int8 attention (see below).
  Not done: soak test, concurrent requests on both slots, AOT with oneAPI 2026.1.

Patches (prepared with AI assistance; Strata is MIT, notice in
[../LICENSES/Strata-LICENSE](../LICENSES/Strata-LICENSE)):

| Patch | What |
|---|---|
| `0001` native Windows build (icx) | CMake: `/fp:precise`, `/Qmkl:sequential`, C-safe warning flags; Level Zero SDK lookup; positioned reads with `ReadFile`; drop a CUDA-only shared-memory query; undo `<windows.h>` macros (`small`, `IN`, `OUT`, `near`, `far`) that clash with engine names; `memmem` fallback |
| `0002` two-card expert parallel through host memory | every GPU shares one `sycl::context`; the shared expert-parallel regions are host USM in that context (one address for both cards). `STRATA_WIN_SHARED_CONTEXT=0` restores one context per device |
| `0003` JIT build option for Xe2 | `STRATA_SYCL_XE2_JIT=ON` makes a JIT build define `STRATA_XMX_SG16` like the bmg AOT build does; `STRATA_SYCL_AOT_OPTIONS` passes extra ocloc options to an AOT build; `setenv` shim for two parity tests |

`configs/` has the engine configs used below (paths replaced with placeholders): `strata-ep2-jit.json` (the working
int8 setup), `strata-ep2-int8-32k.json`, `strata-ep2-fp16-96k.json`.

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
| **Strata, int8 KV, 200k x 2 slots, JIT build** | **correct** (identical to llama.cpp on 4 check prompts) | **108.2 (+79%)** | 599 tok/s @ 33.6k (-20%) | **58.7 (2.0x)** |
| Strata, fp16 KV, 32k (superseded) | correct | 67.4 (+11%) | 570 tok/s @ 26.8k | 46.4 |
| Strata, int8 KV, 200k, AOT build (oneAPI 2026.0) | **wrong** (garbage) | 96.6 (invalid) | 682 @ 33.6k | 66.6 (invalid) |
| Strata, 1 card, int8 KV, AOT build | wrong | 12-20 | - | - |

The quants differ: IQ3_XXS (Strata) is smaller than UD-Q4_K_XL (llama.cpp). On the 4 check prompts
(`bench/squal.py`) the answers matched llama.cpp.
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
