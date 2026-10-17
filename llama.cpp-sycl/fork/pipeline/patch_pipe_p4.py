# patch_pipe_p4.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Diagnostics for the submission lock: per queued graph, how long the worker waited to get the lock and how long it ran with the lock released."""
b = open('ggml-sycl.pipe5.cpp', 'rb').read().decode('utf-8')
crlf = '\r\n' in b
s = b.replace('\r\n', '\n')
def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:80], s.count(a))
    s = s.replace(a, n)

rep('''static thread_local bool t_submit_held  = false;
''', '''static thread_local bool t_submit_held  = false;
// diagnostics (GGML_SYCL_PIPELINE_TRACE): time spent waiting for the lock / running without it, and number of releases
static thread_local int64_t t_lock_wait_us = 0;
static thread_local int64_t t_unlocked_us  = 0;
static thread_local int64_t t_unlock_at    = 0;
static thread_local int64_t t_releases     = 0;
''')
rep('''        if (t_submit_depth++ == 0 && g_started.load(std::memory_order_acquire)) {
            g_submit_mutex.lock();
            t_submit_held = true;''', '''        if (t_submit_depth++ == 0 && g_started.load(std::memory_order_acquire)) {
            const int64_t w0 = ggml_time_us();
            g_submit_mutex.lock();
            t_lock_wait_us += ggml_time_us() - w0;
            t_submit_held = true;''')
rep('''    sycl_pipe::t_submit_held = false;
    sycl_pipe::g_submit_mutex.unlock();
    return true;
}

void ggml_sycl_pipe_relock() {
    sycl_pipe::g_submit_mutex.lock();
    sycl_pipe::t_submit_held = true;
}''', '''    sycl_pipe::t_submit_held = false;
    sycl_pipe::t_unlock_at   = ggml_time_us();
    sycl_pipe::t_releases++;
    sycl_pipe::g_submit_mutex.unlock();
    return true;
}

void ggml_sycl_pipe_relock() {
    const int64_t w0 = ggml_time_us();
    sycl_pipe::t_unlocked_us += w0 - sycl_pipe::t_unlock_at;
    sycl_pipe::g_submit_mutex.lock();
    sycl_pipe::t_lock_wait_us += ggml_time_us() - w0;
    sycl_pipe::t_submit_held = true;
}''')
rep('''                sycl_pipe::submit_scope ss_;
                const int64_t t0 = ggml_time_us();
                ggml_backend_sycl_graph_compute_impl(sycl_ctx, &copy->g);''', '''                const int64_t tq = ggml_time_us();
                sycl_pipe::t_lock_wait_us = 0; sycl_pipe::t_unlocked_us = 0; sycl_pipe::t_releases = 0;
                sycl_pipe::submit_scope ss_;
                const int64_t t0 = ggml_time_us();
                ggml_backend_sycl_graph_compute_impl(sycl_ctx, &copy->g);''')
rep('''start=%lld end=%lld\\n", sycl_ctx->device, (long long) t0, (long long) ggml_time_us());''',
    '''start=%lld end=%lld lockwait_ms=%lld unlocked_ms=%lld releases=%lld\\n", sycl_ctx->device, (long long) tq, (long long) ggml_time_us(),
                            (long long) (sycl_pipe::t_lock_wait_us / 1000), (long long) (sycl_pipe::t_unlocked_us / 1000), (long long) sycl_pipe::t_releases);''')
open('ggml-sycl.pipe6.cpp', 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print('p4 patched')
