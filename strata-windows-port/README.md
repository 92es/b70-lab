# Strata on native Windows

[Strata](https://github.com/hotschmoe/Strata) is a SYCL inference engine with B70-specific work (two-card expert
parallel, MTP, int8 KV). It targets Linux. These two patches make it build and run natively on Windows 11 with
Intel's `icx` compiler, including two-card expert parallel.

- Base: `hotschmoe/Strata` branch `b70-night-20261016` @ `1c360526` (from SergiioB/b70-launcher PR #4).
- Model: ISTA-DASLab GSQ-RCO IQ3_XXS quant of Qwen3.8-Flash-Next (sha256 checked); pack + MTP built on the same
  machine. Cards: two B70s (`level_zero:2,3`).
- Status (2026-10-16): **works with fp16 KV, 2 cards.** int8 KV gives **wrong answers** on Windows (open bug).
  Not done: soak test, concurrent requests.

Patches (prepared with AI assistance; Strata is MIT, notice in
[../LICENSES/Strata-LICENSE](../LICENSES/Strata-LICENSE)):

| Patch | What |
|---|---|
| `0001` native Windows build (icx) | CMake: `/fp:precise`, `/Qmkl:sequential`, C-safe warning flags; Level Zero SDK lookup; positioned reads with `ReadFile`; drop a CUDA-only shared-memory query; undo `<windows.h>` macros (`small`, `IN`, `OUT`, `near`, `far`) that clash with engine names; `memmem` fallback |
| `0002` two-card expert parallel through host memory | every GPU shares one `sycl::context`; the shared expert-parallel regions are host USM in that context (one address for both cards). `STRATA_WIN_SHARED_CONTEXT=0` restores one context per device |

`configs/` has the two engine configs used below (paths replaced with placeholders).

## Results

Same 7-prompt set ([../bench/sbench.py](../bench/)), temperature 0, thinking off, 4x B70 machine, driver 8976:

| Setup | Answers | Decode tok/s | Prompt read | Decode after long prompt |
|---|---|---|---|---|
| llama.cpp fork r62, 4 cards, UD-Q4_K_XL | correct | 60.6 | 748 tok/s @ 33.6k | 28.6 |
| Strata, 2 cards, **fp16 KV**, 32k ctx | **correct** (same as llama.cpp) | **67.4** (+11%) | 570 tok/s @ 26.8k | 46.4 |
| Strata, 2 cards, int8 KV (upstream recipe, 200k) | **wrong** (garbage) | 96.6 (invalid) | 682 @ 33.6k | 66.6 (invalid) |
| Strata, 1 card, int8 KV | wrong | 12-20 | - | - |

fp16 KV fits 96k context on two cards (1.6 GB free); 131k does not (card 1 ~0.85 GB short).
IQ3_XXS is a smaller quant than the UD-Q4_K_XL used with llama.cpp; on the 4 check prompts (`bench/squal.py`) the
answers matched llama.cpp on the same file.

Lesson: run fixed-answer checks before trusting any speed number. The int8 run was the fastest and useless.

## Windows / Level Zero findings

Measured with [../reproducers/ep-probe/](../reproducers/) on two B70s:

- No IPC: `zeMemGetIpcHandle` returns `0x78000003` (unsupported feature).
- `zeDeviceCanAccessPeer` = 0 between B70s.
- A kernel on one card touching the other card's device memory: `DEVICE_LOST`.
- Driver-imported host pointers used from another context: also `DEVICE_LOST`.
- Host USM allocated inside **one** context shared by both cards works for both cards. Patch `0002` is built on that.

So on Windows, card-to-card traffic has to go through host memory. Linux behaviour was not tested here.

## Open: int8 KV wrong on Windows

fp16 KV gives correct answers with everything else equal. Ruled out:

- the tokenizer (token ids identical to llama.cpp),
- the download (sha256 match),
- PLE file I/O,
- the prompt-attention kernel (`STRATA_PROMPT_ATTN_OLD=1` still wrong),
- the experimental "night" switches (minimal env still wrong, output "!!!!").

The `kv_q8`, `qsa` and `kv_steps` parity tests pass, so the bug is in a path they do not cover.

Parity tests on Windows: 21 of 23 pass. Failing: `gr_parity` (fused GR read + q8_1, `STRATA_QFUSE` path: "not
written") and `s2_expert_grouped_parity` (grouped kernel differs from the previous one; the Q2_0 experts it covers are
the MTP draft layer only).

## Other notes

- With default settings the engine hangs: verify waits on a host doorbell and hits a timeout. The PR's
  `STRATA_VERIFY_DEVICE_PLAN=1` + `STRATA_VERIFY_NO_HOST=1` avoid it (set in both configs).
- Build: oneAPI 2026.0 `icx` with the MSVC toolchain environment; the oneAPI runtime DLLs must be next to
  `strata.exe` or on `PATH`.
