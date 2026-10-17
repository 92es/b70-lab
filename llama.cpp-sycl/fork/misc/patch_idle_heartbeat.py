# patch_idle_heartbeat.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Idle heartbeat for the SYCL backend (worktree llama.cpp-dsv4).
On Windows the Intel Arc driver starts demoting a process's VRAM to system RAM after ~70 s without GPU work from
that process (measured 2026-10-01: Flash-Next on 4 B70s, ~1 GB/s into host RAM until the RAM watchdog kills the
server). Activity from another process does not help; a request every 45 s does. This adds an opt-in background
thread (GGML_SYCL_IDLE_HEARTBEAT_SEC=N, default 0 = off) that submits a trivial kernel to every queue this process
has computed on whenever no graph has run for N seconds."""
import os

import sys
root = sys.argv[1] if len(sys.argv) > 1 else "."
p = os.path.join(root, "ggml", "src", "ggml-sycl", "ggml-sycl.cpp")
raw = open(p, "rb").read().decode("utf-8")
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")
assert "ggml_sycl_idle_heartbeat" not in s, "already patched"

CODE = r'''#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

// ---- idle heartbeat (GGML_SYCL_IDLE_HEARTBEAT_SEC, default off) ---------------------------------------------
// Keeps the driver from treating a loaded-but-idle process as inactive and demoting its VRAM to system RAM.
static std::atomic<int64_t> g_sycl_heartbeat_last_us{0};
static std::mutex           g_sycl_heartbeat_mutex;
static std::vector<std::pair<sycl::queue *, uint32_t *>> g_sycl_heartbeat_queues;

static void ggml_sycl_idle_heartbeat_note(sycl::queue * q) {
    static const int period_s = ggml_sycl_get_env("GGML_SYCL_IDLE_HEARTBEAT_SEC", 0);
    if (period_s <= 0) {
        return;
    }
    g_sycl_heartbeat_last_us.store(ggml_time_us());
    {
        std::lock_guard<std::mutex> lock(g_sycl_heartbeat_mutex);
        bool known = false;
        for (const auto & e : g_sycl_heartbeat_queues) {
            known = known || e.first == q;
        }
        if (!known) {
            uint32_t * scratch = sycl::malloc_device<uint32_t>(64, *q);
            if (scratch != nullptr) {
                g_sycl_heartbeat_queues.emplace_back(q, scratch);
            }
        }
    }
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread([] {
            const int64_t period_us = (int64_t) period_s * 1000000;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(period_s));
                if (ggml_time_us() - g_sycl_heartbeat_last_us.load() < period_us) {
                    continue;
                }
                std::vector<std::pair<sycl::queue *, uint32_t *>> queues;
                {
                    std::lock_guard<std::mutex> lock(g_sycl_heartbeat_mutex);
                    queues = g_sycl_heartbeat_queues;
                }
                for (const auto & e : queues) {
                    try {
                        uint32_t * scratch = e.second;
                        e.first->parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { scratch[i[0]] += 1u; }).wait();
                    } catch (const sycl::exception &) {
                        // best effort
                    }
                }
            }
        }).detach();
    });
}

'''
a1 = "static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {\n    ggml_sycl_set_main_device(sycl_ctx->device);\n"
assert s.count(a1) == 1
s = s.replace(a1, CODE + a1 + "    ggml_sycl_idle_heartbeat_note(sycl_ctx->stream());\n")
if crlf:
    s = s.replace("\n", "\r\n")
open(p, "wb").write(s.encode("utf-8"))
print("patched")
