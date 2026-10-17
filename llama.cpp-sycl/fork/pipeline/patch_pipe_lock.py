# patch_pipe_lock.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Stage 1b of the cross-card pipeline: one submission lock.

The Level Zero GPU driver deadlocks when two host threads append kernel launches at the same time (stacks in
stacks-pipe.txt: two threads inside zeCommandListAppendLaunchKernelWithArguments waiting on driver SRW locks).
So only one thread may talk to the runtime at a time; a thread gives the lock up while it WAITS for its card,
which is where prompt batches spend their time.

Inputs: ggml-sycl.pipe2.cpp (stage 1), common.hpp, fattn-mkl.cpp, lightning-indexer.cpp, fattn-sparse.cpp, set_rows.cpp
"""
import re, sys

def load(fn):
    b = open(fn, 'rb').read().decode('utf-8')
    return ('\r\n' in b), b.replace('\r\n', '\n')

def save(fn, crlf, s):
    open(fn, 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))

PURE_WAIT = re.compile(
    r'^(?P<ind>\s*)(?P<stmt>(?:SYCL_CHECK\(CHECK_TRY_ERROR\()?\(?\*?[A-Za-z_][\w\.]*(?:->[\w]+(?:\(\))?)*\)?(?:->|\.)wait(?:_and_throw)?\(\)(?:\)\))?;)(?P<tail>\s*(?://.*)?)$')

def wrap_waits(s, skip=lambda line: False):
    out, n = [], 0
    for line in s.split('\n'):
        m = PURE_WAIT.match(line)
        if m and 'cv' not in line and 'memcpy' not in line and 'memset' not in line and not skip(line):
            out.append(f"{m.group('ind')}{{ ggml_sycl_pipe_wait_scope ws_; {m.group('stmt')} }}{m.group('tail')}")
            n += 1
        else:
            out.append(line)
    return '\n'.join(out), n

# ---------------------------------------------------------------------------------------------------- common.hpp
crlf, s = load('common.r29.hpp')
a = 'const ggml_sycl_device_info & ggml_sycl_info();\n'
assert s.count(a) == 1
s = s.replace(a, a + '''
// Cross-card pipeline (GGML_SYCL_PIPELINE=1): the submission lock. The GPU driver deadlocks when two host threads
// append kernel launches concurrently, so a thread holds this lock while it issues work and gives it up while it
// waits for its card. Both calls are no-ops on threads that do not hold the lock (i.e. always, with the switch off).
bool ggml_sycl_pipe_unlock();   // returns true if the calling thread held the lock and released it
void ggml_sycl_pipe_relock();
struct ggml_sycl_pipe_wait_scope {
    bool was;
    ggml_sycl_pipe_wait_scope() : was(ggml_sycl_pipe_unlock()) {}
    ~ggml_sycl_pipe_wait_scope() { if (was) { ggml_sycl_pipe_relock(); } }
    ggml_sycl_pipe_wait_scope(const ggml_sycl_pipe_wait_scope &) = delete;
    ggml_sycl_pipe_wait_scope & operator=(const ggml_sycl_pipe_wait_scope &) = delete;
};
''')
save('common.pipe.hpp', crlf, s)

# ------------------------------------------------------------------------------------------------- ggml-sycl.cpp
crlf, s = load('ggml-sycl.pipe2.cpp')

def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:80], s.count(a))
    s = s.replace(a, n)

# the lock itself + scope, inside the namespace (after the worker table)
rep('''static thread_local int  t_worker_device = -1;   // device whose FIFO this thread serves, -1 on other threads
''', '''static thread_local int  t_worker_device = -1;   // device whose FIFO this thread serves, -1 on other threads

// the submission lock (see common.hpp)
static std::mutex        g_submit_mutex;
static thread_local int  t_submit_depth = 0;
static thread_local bool t_submit_held  = false;

// held by every backend entry point and every queued job while it issues work
struct submit_scope {
    submit_scope() {
        if (t_submit_depth++ == 0 && g_started.load(std::memory_order_acquire)) {
            g_submit_mutex.lock();
            t_submit_held = true;
        }
    }
    ~submit_scope() {
        if (--t_submit_depth == 0 && t_submit_held) {
            t_submit_held = false;
            g_submit_mutex.unlock();
        }
    }
    submit_scope(const submit_scope &) = delete;
    submit_scope & operator=(const submit_scope &) = delete;
};
''')
rep('''} // namespace sycl_pipe
''', '''} // namespace sycl_pipe

bool ggml_sycl_pipe_unlock() {
    if (!sycl_pipe::t_submit_held) {
        return false;
    }
    sycl_pipe::t_submit_held = false;
    sycl_pipe::g_submit_mutex.unlock();
    return true;
}

void ggml_sycl_pipe_relock() {
    sycl_pipe::g_submit_mutex.lock();
    sycl_pipe::t_submit_held = true;
}
''')

# drain: never wait for another thread while holding the submission lock
rep('''    worker & w = g_workers[device];
    std::unique_lock<std::mutex> lock(w.m);
    while (!w.cv_idle.wait_for(''', '''    worker & w = g_workers[device];
    ggml_sycl_pipe_wait_scope ws_;
    std::unique_lock<std::mutex> lock(w.m);
    while (!w.cv_idle.wait_for(''')
# event wait: same
rep('''    std::unique_lock<std::mutex> lock(st->m);
    st->cv.wait(lock, [&] { return st->done; });
}''', '''    ggml_sycl_pipe_wait_scope ws_;
    std::unique_lock<std::mutex> lock(st->m);
    st->cv.wait(lock, [&] { return st->done; });
}''')
# worker registration touches the runtime
rep('''        std::lock_guard<std::mutex> lock(g_start_mutex);
        dpct::select_device(device);
    }
    g_registered.fetch_add(1);''', '''        std::lock_guard<std::mutex> lock(g_start_mutex);
        dpct::select_device(device);
    }
    g_registered.fetch_add(1);
    // (jobs take the submission lock themselves)''')

# queued jobs
rep('''        sycl_pipe::enqueue(device, [held, target, device] {
            auto q = &(dpct::dev_mgr::instance().get_device(device).default_queue());''',
    '''        sycl_pipe::enqueue(device, [held, target, device] {
            sycl_pipe::submit_scope ss_;
            auto q = &(dpct::dev_mgr::instance().get_device(device).default_queue());''')
rep('''                sycl_pipe::enqueue(device, [held, target, device] {
                    auto q = &(dpct::dev_mgr::instance().get_device(device).default_queue());''',
    '''                sycl_pipe::enqueue(device, [held, target, device] {
                    sycl_pipe::submit_scope ss_;
                    auto q = &(dpct::dev_mgr::instance().get_device(device).default_queue());''')
rep('''        sycl_pipe::enqueue(sycl_ctx->device, [held, target, stream] {
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));''',
    '''        sycl_pipe::enqueue(sycl_ctx->device, [held, target, stream] {
            sycl_pipe::submit_scope ss_;
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));''')
rep('''        sycl_pipe::enqueue(sycl_ctx->device, [data, source, size, stream] {
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));''',
    '''        sycl_pipe::enqueue(sycl_ctx->device, [data, source, size, stream] {
            sycl_pipe::submit_scope ss_;
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));''')
rep('''            sycl_pipe::enqueue(sycl_ctx->device, [sycl_ctx, copy] {
                const int64_t t0 = ggml_time_us();''', '''            sycl_pipe::enqueue(sycl_ctx->device, [sycl_ctx, copy] {
                sycl_pipe::submit_scope ss_;
                const int64_t t0 = ggml_time_us();''')

# backend / buffer entry points: take the lock for the duration of the call
ENTRY = [
    ('ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) try {\n', None),
    ('ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,\n                                     ggml_tensor *tensor) try {\n', None),
    ('static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,\n                                                ggml_tensor *tensor,\n                                                const void *data, size_t offset,\n                                                size_t size) try {\n', None),
    ('                                                   size_t n_copies, size_t stride_tensor,\n                                                   size_t stride_data) try {\n', None),
    ('static void ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,\n                                                const ggml_tensor *tensor,\n                                                void *data, size_t offset,\n                                                size_t size) try {\n', None),
    ('ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,\n                                    const ggml_tensor *src,\n                                    ggml_tensor *dst) try {\n', None),
    ('static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,\n                                           uint8_t value) try {\n', None),
    ('static void ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,\n                                                   size_t offset, size_t size) {\n', None),
    ('static void ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,\n                                               ggml_tensor *tensor,\n                                               const void *data, size_t offset,\n                                               size_t size) try {\n', None),
    ('static void ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,\n                                               const ggml_tensor *tensor,\n                                               void *data, size_t offset,\n                                               size_t size) try {\n', None),
    ('static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {\n', None),
    ('static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {\n', None),
    ('static void ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)\ntry\n{\n', None),
    ('static void ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) try {\n', None),
]
for head, _ in ENTRY:
    rep(head, head + '    sycl_pipe::submit_scope pipe_ss_;\n')

# the idle heartbeat also submits kernels
rep('''                for (const auto & e : queues) {
                    try {
                        uint32_t * scratch = e.second;''', '''                for (const auto & e : queues) {
                    try {
                        sycl_pipe::submit_scope ss_;
                        uint32_t * scratch = e.second;''')

s, n_main = wrap_waits(s)
save('ggml-sycl.pipe3.cpp', crlf, s)

# ------------------------------------------------------------------------------------------------- other files
crlf, s = load('fattn-mkl.cpp')
s, n1 = wrap_waits(s)
c = s.count('try { ev.wait_and_throw(); }')
s = s.replace('try { ev.wait_and_throw(); }', 'try { ggml_sycl_pipe_wait_scope ws_; ev.wait_and_throw(); }')
save('fattn-mkl.pipe.cpp', crlf, s)

crlf, s = load('lightning-indexer.new2.cpp')
s, n2 = wrap_waits(s)
save('lightning-indexer.pipe.cpp', crlf, s)

print('wrapped waits: ggml-sycl.cpp', n_main, 'fattn-mkl.cpp', n1, '+', c, 'lightning-indexer.cpp', n2)
