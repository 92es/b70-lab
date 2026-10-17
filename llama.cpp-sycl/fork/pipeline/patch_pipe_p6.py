# patch_pipe_p6.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Stage 4 of the cross-card pipeline: a card-to-card copy of a tensor whose value is a still-queued host write takes
the bytes from that host copy instead of waiting for the source card.

With a draft model attached, a per-batch input of the target graph lives on card 1 and card 0 needs a copy of it. The
write to card 1 is queued behind card 1's previous batch, so the copy (and with it card 0's next batch) waited for card 1.

Also: the queued writes/reads release the submission lock while they wait for their memcpy.

Input: ggml-sycl.pipe7.cpp. Output: ggml-sycl.pipe8.cpp
"""
b = open('ggml-sycl.pipe7.cpp', 'rb').read().decode('utf-8')
crlf = '\r\n' in b
s = b.replace('\r\n', '\n')

def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:80], s.count(a))
    s = s.replace(a, n)

# ---- registry of queued host writes -------------------------------------------------------------------------------
rep('''// Copy device memory from one card to another without blocking the caller.''', '''// Host copies of writes that are still queued, by target address. A later card-to-card copy of the same tensor can take
// the bytes from here: they are what the tensor will hold once the write has run. Entries of a card are dropped when a
// graph is handed to that card (the graph may reuse the memory for something else).
struct held_write {
    std::shared_ptr<std::vector<char>> data;
    int                                device;
};
static std::mutex                                     g_held_mutex;
static std::unordered_map<const char *, held_write>   g_held;

static bool held_enabled() {
    static const bool v = ggml_sycl_get_env("GGML_SYCL_PIPELINE_HELD", 1) != 0;
    return v;
}

static void held_add(int device, const char * base, size_t offset, const std::shared_ptr<std::vector<char>> & data) {
    std::lock_guard<std::mutex> lock(g_held_mutex);
    if (offset != 0 || !held_enabled()) {
        g_held.erase(base);   // a partial write: the earlier full copy no longer describes the tensor
        return;
    }
    g_held[base] = { data, device };
}

static void held_done(const char * target, const std::shared_ptr<std::vector<char>> & data) {
    std::lock_guard<std::mutex> lock(g_held_mutex);
    auto it = g_held.find(target);
    if (it != g_held.end() && it->second.data == data) {
        g_held.erase(it);
    }
}

static void held_forget_device(int device) {
    std::lock_guard<std::mutex> lock(g_held_mutex);
    for (auto it = g_held.begin(); it != g_held.end();) {
        it = it->second.device == device ? g_held.erase(it) : std::next(it);
    }
}

static std::shared_ptr<std::vector<char>> held_find(int device, const char * src, size_t nbytes) {
    std::lock_guard<std::mutex> lock(g_held_mutex);
    auto it = g_held.find(src);
    if (it == g_held.end() || it->second.device != device || it->second.data->size() != nbytes) {
        return nullptr;
    }
    return it->second.data;
}

// Copy device memory from one card to another without blocking the caller.''')

rep('''static void copy_async(int dev_src, const char * src, int dev_dst, char * dst, size_t nbytes) {
    auto st = std::make_shared<copy_state>();''', '''static void copy_async(int dev_src, const char * src, int dev_dst, char * dst, size_t nbytes) {
    if (auto held = held_find(dev_src, src, nbytes)) {
        // the source is a host write that card dev_src has not run yet: no need to wait for that card
        auto write_held = [held, dev_dst, dst] {
            submit_scope ss_;
            auto q = &(dpct::dev_mgr::instance().get_device(dev_dst).default_queue());
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(q->wait())); }
            sycl::event ev;
            SYCL_CHECK(CHECK_TRY_ERROR(ev = q->memcpy(dst, held->data(), held->size())));
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }
        };
        if (pending(dev_dst)) {
            enqueue(dev_dst, write_held, "cpy-held");
        } else {
            write_held();
        }
        return;
    }
    auto st = std::make_shared<copy_state>();''')

# ---- the two queued host writes register themselves ---------------------------------------------------------------
rep('''        char * target = (char *) tensor->data + offset;
        const int device = ctx->device;
        sycl_pipe::enqueue(device, [held, target, device] {
            sycl_pipe::submit_scope ss_;
            auto q = &(dpct::dev_mgr::instance().get_device(device).default_queue());
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(q->wait())); }
            SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(target, held->data(), held->size()).wait()));
        }, "buf-set");''', '''        char * target = (char *) tensor->data + offset;
        const int device = ctx->device;
        sycl_pipe::held_add(device, (const char *) tensor->data, offset, held);
        sycl_pipe::enqueue(device, [held, target, device] {
            {
                sycl_pipe::submit_scope ss_;
                auto q = &(dpct::dev_mgr::instance().get_device(device).default_queue());
                { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(q->wait())); }
                sycl::event ev;
                SYCL_CHECK(CHECK_TRY_ERROR(ev = q->memcpy(target, held->data(), held->size())));
                { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }
            }
            sycl_pipe::held_done(target, held);
        }, "buf-set");''')

rep('''        char * target = (char *) tensor->data + offset;
        sycl_pipe::enqueue(sycl_ctx->device, [held, target, stream] {
            sycl_pipe::submit_scope ss_;
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(target, held->data(), held->size()).wait()));
        }, "set");''', '''        char * target = (char *) tensor->data + offset;
        sycl_pipe::held_add(sycl_ctx->device, (const char *) tensor->data, offset, held);
        sycl_pipe::enqueue(sycl_ctx->device, [held, target, stream] {
            {
                sycl_pipe::submit_scope ss_;
                { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
                sycl::event ev;
                SYCL_CHECK(CHECK_TRY_ERROR(ev = stream->memcpy(target, held->data(), held->size())));
                { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }
            }
            sycl_pipe::held_done(target, held);
        }, "set");''')

# a write that is NOT queued (card idle) also changes the tensor: drop any stale entry. Cheap: done at the top of both.
rep('''    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    if (sycl_pipe::pending(ctx->device)) {
        // the card is still working on queued batches: this write must land after them, with its own copy of the data''',
    '''    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    if (sycl_pipe::pending(ctx->device)) {
        // the card is still working on queued batches: this write must land after them, with its own copy of the data''')

# ---- a graph handed to a card invalidates that card's entries ------------------------------------------------------
rep('''    if (sycl_pipe::enabled() && sycl_pipe::t_worker_device < 0) {
        static const bool follow''', '''    if (sycl_pipe::enabled() && sycl_pipe::t_worker_device < 0) {
        sycl_pipe::held_forget_device(sycl_ctx->device);
        static const bool follow''')

# ---- queued read: release the lock while the memcpy runs ------------------------------------------------------------
rep('''            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(data, source, size).wait()));''',
    '''            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); }
            sycl::event ev;
            SYCL_CHECK(CHECK_TRY_ERROR(ev = stream->memcpy(data, source, size)));
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }''')

open('ggml-sycl.pipe8.cpp', 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print('p6 patched')
