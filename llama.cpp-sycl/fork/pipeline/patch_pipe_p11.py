# patch_pipe_p11.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Stage 5 of the cross-card pipeline: bounded queues.

Without a draft model nothing made the host wait once the scheduler/llama waits were removed (r51): it queued tens of
thousands of tokens of work in seconds, and the host copies kept for queued work exhausted memory ("bad allocation" on
a 102k-token read of the plain DeepSeek entry). Now the host waits before handing a card another batch graph while that
card already has GGML_SYCL_PIPELINE_MAX_QUEUED (default 2) batch graphs unfinished. The cards are the bottleneck, so
this costs no throughput; it bounds host memory and keeps progress/cancel meaningful.

Input: ggml-sycl.pipe12.cpp. Output: ggml-sycl.pipe13.cpp
"""
b = open('ggml-sycl.pipe12.cpp', 'rb').read().decode('utf-8')
crlf = '\r\n' in b
s = b.replace('\r\n', '\n')

def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:80], s.count(a))
    s = s.replace(a, n)

rep('''    bool                               busy    = false;
    bool                               started = false;
};
''', '''    bool                               busy    = false;
    bool                               started = false;
    int                                graphs  = 0;   // batch graphs queued or running (guarded by m)
};
''')

# every finished job wakes waiters (drain checks its own condition; the throttle checks the graph count)
rep('''            std::lock_guard<std::mutex> lock(w.m);
            w.busy = false;
            if (w.q.empty()) {
                w.cv_idle.notify_all();
            }''', '''            std::lock_guard<std::mutex> lock(w.m);
            w.busy = false;
            w.cv_idle.notify_all();''')

rep('''// Host copies of writes that are still queued, by target address.''', '''// Bounded queues: wait (without the submission lock) while this card already has max_queued() batch graphs unfinished,
// then count the one about to be queued.
static int max_queued() {
    static const int v = std::max(1, ggml_sycl_get_env("GGML_SYCL_PIPELINE_MAX_QUEUED", 2));
    return v;
}

static void graph_admit(int device) {
    worker & w = g_workers[device];
    ggml_sycl_pipe_wait_scope ws_;
    std::unique_lock<std::mutex> lock(w.m);
    const int limit = max_queued();
    if (w.graphs >= limit && trace()) {
        fprintf(stderr, "[SYCL-PIPE] host holds back: device %d has %d batch graphs unfinished t=%lld\\n", device, w.graphs, (long long) ggml_time_us());
    }
    w.cv_idle.wait(lock, [&] { return w.graphs < limit; });
    w.graphs++;
}

static void graph_finished(int device) {
    worker & w = g_workers[device];
    std::lock_guard<std::mutex> lock(w.m);
    w.graphs--;
}

// Host copies of writes that are still queued, by target address.''')

rep('''            sycl_pipe::ensure_started();
            auto copy = std::make_shared<sycl_pipe::graph_copy>(cgraph);''', '''            sycl_pipe::ensure_started();
            const bool counted = sycl_pipe::is_batch_graph(cgraph);
            if (counted) {
                sycl_pipe::graph_admit(sycl_ctx->device);
            }
            auto copy = std::make_shared<sycl_pipe::graph_copy>(cgraph);''')

rep('''            sycl_pipe::enqueue(sycl_ctx->device, [sycl_ctx, copy] {
                const int64_t tq = ggml_time_us();''', '''            sycl_pipe::enqueue(sycl_ctx->device, [sycl_ctx, copy, counted] {
                const int64_t tq = ggml_time_us();''')

rep('''                            (long long) (sycl_pipe::t_lock_wait_us / 1000), (long long) (sycl_pipe::t_unlocked_us / 1000), (long long) sycl_pipe::t_releases);
                }
            }, "graph");''', '''                            (long long) (sycl_pipe::t_lock_wait_us / 1000), (long long) (sycl_pipe::t_unlocked_us / 1000), (long long) sycl_pipe::t_releases);
                }
                if (counted) {
                    sycl_pipe::graph_finished(sycl_ctx->device);
                }
            }, "graph");''')

open('ggml-sycl.pipe13.cpp', 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print('p11 patched')
