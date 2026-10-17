# patch_pipe_p3.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Stage 3 of the cross-card pipeline: the scheduler's event waits no longer block the host thread.

With a draft model attached (DSpark) the target graph has one small extra split per card (the layer-input tap). The
scheduler waits for the card's event before it copies that split's inputs, i.e. for the card's main split of the SAME
batch, which puts the cards back in lockstep. Every write to a busy card already goes through that card's FIFO (set_tensor,
card-to-card copies), so the order the wait protects is kept by the queue itself.

Input: ggml-sycl.pipe4.cpp. Output: ggml-sycl.pipe5.cpp
"""
b = open('ggml-sycl.pipe4.cpp', 'rb').read().decode('utf-8')
crlf = '\r\n' in b
s = b.replace('\r\n', '\n')

def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:80], s.count(a))
    s = s.replace(a, n)

rep('''static void event_wait(ggml_backend_event_t event) {
    std::shared_ptr<event_state> st;
''', '''static void event_wait(ggml_backend_event_t event) {
    // GGML_SYCL_PIPELINE_EVENT_WAIT=1 restores the blocking wait (stage 1 behaviour)
    static const bool blocking = ggml_sycl_get_env("GGML_SYCL_PIPELINE_EVENT_WAIT", 0) != 0;
    if (!blocking) {
        return;
    }
    std::shared_ptr<event_state> st;
''')

rep('''    worker & w = g_workers[device];
    ggml_sycl_pipe_wait_scope ws_;
    std::unique_lock<std::mutex> lock(w.m);
    while (!w.cv_idle.wait_for(''', '''    worker & w = g_workers[device];
    if (trace() && t_worker_device < 0 && pending(device)) {
        fprintf(stderr, "[SYCL-PIPE] host waits for device %d t=%lld\\n", device, (long long) ggml_time_us());
    }
    ggml_sycl_pipe_wait_scope ws_;
    std::unique_lock<std::mutex> lock(w.m);
    while (!w.cv_idle.wait_for(''')

# a small split (for example the layer-input tap of a draft model) on a card that still has queued work follows the
# queue instead of making the host wait for that card
rep('''        if (sycl_pipe::is_batch_graph(cgraph)) {
            sycl_pipe::ensure_started();''', '''        static const bool follow = ggml_sycl_get_env("GGML_SYCL_PIPELINE_FOLLOW", 1) != 0;
        if (sycl_pipe::is_batch_graph(cgraph) || (follow && sycl_pipe::pending(sycl_ctx->device))) {
            sycl_pipe::ensure_started();''')

open('ggml-sycl.pipe5.cpp', 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print('p3 patched')
