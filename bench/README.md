# Benchmarks and how to compare fairly

## Scripts

Both talk to any OpenAI-compatible server (llama-server, Strata, ...) and need only Python 3.

- `sbench.py URL MODEL [--n 512] [--rounds 2] [--long 30000] [--tag name] [--doc file]`
  Seven prompts (code explain, code write, summary, fiction, JSON extraction, continuity edit, arithmetic), streamed,
  temperature 0, thinking off. Decode tok/s = (completion tokens - 1) / (last token time - first token time).
  Round 1 is warm-up and is left out of the mean. `--long` adds one long prompt read (about N tokens of `--doc`) and
  reports prompt tok/s and the decode speed right after it.
- `squal.py URL MODEL`: four prompts with checkable answers (JSON extraction, arithmetic, five capitals, count to
  30). Run it before trusting any speed number.

Note: the published `sbench.py` uses a different fiction prompt than the one behind the 2026-10-16 numbers in this
repo (the original prompt was replaced). The other six prompts are the same; expect the fiction number to move a bit.

## Rules I learned the hard way

- **One harness per comparison.** The same Flash-Next server reads ~45 tok/s on one harness and ~61 on another.
  Prompt mix, output length and draft acceptance all move the number.
- **Same-session baseline.** Re-run the baseline next to the candidate (A B B A order), not from yesterday's notes.
  Driver updates, other GPU users and even the order of tests changed results here (the MXFP4 prompt speed depended on
  whether a decode step had already run).
- **Enough samples.** Run-to-run noise on a single prompt was +-1.5 tok/s for Flash-Next; a single-prompt "+17%" did
  not replicate on a varied set. The Flash-Next decode tables use 28-42 samples per arm over 7 prompts.
- **A speed number only counts after an answer check.** A miscompiled Strata build produced garbage at 96.6 "tok/s":
  the MTP draft predicted the garbage well, so acceptance and tok/s went up. Run `squal.py` first.
- **Check answers, not just speed.** Fixed-answer prompts (`squal.py`), `test-backend-ops` against the CPU backend for
  kernels, and perplexity / KL divergence for anything that changes numerics.
- **Batch-1 perplexity is the exactness test.** `llama-perplexity -ub 8` is deterministic on this backend; batched
  prompt reads are not bit-reproducible run to run on any runtime I tried, so comparing output text after a long
  prompt does not prove anything.
- **Test short prompts too.** A layout bug that only hit 2-8 token batches passed every long-prompt benchmark and
  poisoned the model's recurrent state for later requests. A probe that asks for a first token after a 5-token and a
  45-token prompt (expecting " Paris") caught it.
- **Draft and no-draft configs behave differently.** Test both when changing the backend.
- **Check for running jobs before every GPU action on a shared box.** If scheduled jobs or other clients use the same
  GPU server, check what is running (and when jobs are scheduled) before each load, unload or benchmark; tests here
  broke a few of their in-flight requests.
- **Keep other clients off the GPUs.** A model swap by another client wipes the prompt cache and changes timings;
  check the server's running list and recent requests before starting.
- **Environment variables leak between arms** if you run A and B from one PowerShell session. Start each arm clean.
  (PowerShell variable names are also case-insensitive: `$t` and `$T` are the same variable.)
- **Profile the slow case against a fast one before blaming hardware.** Strata's decode dropped from ~98 to ~70 tok/s;
  heat, clocks, CPU, RAM, card pair, inter-card latency, timer resolution and a reboot were all ruled out before the
  cause turned up: per-token random reads from the SSD (`--ple-io direct`). Check what a decode step reads from disk.
- **Leave memory headroom.** A config that "fits" with a card at its memory limit can be ~3x slower than the next
  smaller one (Strata prompt chunk 10240 vs 8192).
- **Watch host RAM during loads.** A config that does not fit can silently spill GPU allocations into system RAM
  (see [../hardware-and-windows-notes.md](../hardware-and-windows-notes.md)). I run every test server under a guard
  that kills it when available RAM drops below ~10 GB.
- **Microbenchmarks lie for MoE.** `test-backend-ops perf` reuses the same experts, which then sit in L2; trust
  end-to-end numbers.
- **Per-op profiles over-count small ops.** `GGML_SYCL_PROFILE_OPS=1` serialises everything; a small kernel that shows
  ~13 us there costs ~1-3 us in a normal async run.
