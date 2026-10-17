# patch_pipe_p2.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Stage 2 of the cross-card pipeline: card-to-card copies that do not block the host thread.

Input: ggml-sycl.pipe3.cpp (stage 1 + submission lock). Output: ggml-sycl.pipe4.cpp
"""
b = open('ggml-sycl.pipe3.cpp', 'rb').read().decode('utf-8')
crlf = '\r\n' in b
s = b.replace('\r\n', '\n')

def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:80], s.count(a))
    s = s.replace(a, n)

# ---- helper inside the namespace ---------------------------------------------------------------------------------
rep('''} // namespace sycl_pipe

bool ggml_sycl_pipe_unlock() {''', '''// Copy device memory from one card to another without blocking the caller. The read is queued on the source card
// (behind the work that produces the data), the write on the destination card (ahead of the graph that consumes it);
// the write waits for the read through `st`. The bytes travel through host memory, as the blocking copy does.
struct copy_state {
    std::mutex              m;
    std::condition_variable cv;
    bool                    ready = false;
    std::vector<char>       data;
};

static void copy_async(int dev_src, const char * src, int dev_dst, char * dst, size_t nbytes) {
    auto st = std::make_shared<copy_state>();
    auto read = [st, dev_src, src, nbytes] {
        {
            submit_scope ss_;
            auto q = &(dpct::dev_mgr::instance().get_device(dev_src).default_queue());
            st->data.resize(nbytes);
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(q->wait())); }
            sycl::event ev;
            SYCL_CHECK(CHECK_TRY_ERROR(ev = q->memcpy(st->data.data(), src, nbytes)));
            { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }
        }
        { std::lock_guard<std::mutex> lock(st->m); st->ready = true; }
        st->cv.notify_all();
    };
    auto write = [st, dev_dst, dst, nbytes] {
        {
            std::unique_lock<std::mutex> lock(st->m);
            st->cv.wait(lock, [&] { return st->ready; });
        }
        submit_scope ss_;
        auto q = &(dpct::dev_mgr::instance().get_device(dev_dst).default_queue());
        { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(q->wait())); }
        sycl::event ev;
        SYCL_CHECK(CHECK_TRY_ERROR(ev = q->memcpy(dst, st->data.data(), nbytes)));
        { ggml_sycl_pipe_wait_scope ws_; SYCL_CHECK(CHECK_TRY_ERROR(ev.wait())); }
    };

    const bool src_busy = pending(dev_src);
    if (src_busy) {
        enqueue(dev_src, read);
    } else {
        read();   // the caller already holds the submission lock
    }
    if (src_busy || pending(dev_dst)) {
        enqueue(dev_dst, write);
    } else {
        write();
    }
}

} // namespace sycl_pipe

bool ggml_sycl_pipe_unlock() {''')

# ---- buffer cpy_tensor: use it instead of draining the source ----------------------------------------------------
old_start = '''        if (sycl_pipe::active(dst_ctx->device)) {
            // the source must be complete now (the caller may let that card overwrite it next); the destination may
            // still be busy with earlier batches, so its write goes through its FIFO
            sycl_pipe::drain(src_ctx->device);'''
i = s.index(old_start)
j = s.index('''        ggml_sycl_set_device(src_ctx->device);''', i)
s = s[:i] + '''        if (sycl_pipe::active(dst_ctx->device) && sycl_pipe::t_worker_device < 0 &&
            (sycl_pipe::pending(src_ctx->device) || sycl_pipe::pending(dst_ctx->device))) {
            // queued behind whatever both cards still have to do; the caller is not blocked
            sycl_pipe::copy_async(src_ctx->device, (const char *) src->data, dst_ctx->device, (char *) dst->data, ggml_nbytes(src));
            return true;
        }

''' + s[j:]

# ---- backend cpy_tensor_async -------------------------------------------------------------------------------------
rep('''static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {''', '''// asynchronous copy between two cards: only with the pipeline running (otherwise the scheduler's blocking path is used)
static bool ggml_backend_sycl_cpy_tensor_async_pipe(ggml_backend_t backend_src, ggml_backend_t backend_dst,
                                                    const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_is_sycl(backend_src) || !ggml_backend_is_sycl(backend_dst)) {
        return false;
    }
    ggml_backend_sycl_context * ctx_src = (ggml_backend_sycl_context *) backend_src->context;
    ggml_backend_sycl_context * ctx_dst = (ggml_backend_sycl_context *) backend_dst->context;
    if (!sycl_pipe::active(ctx_dst->device) || sycl_pipe::t_worker_device >= 0) {
        return false;
    }
    // nothing in flight (decode): the scheduler's direct card-to-card copy is faster than going through host memory
    if (!sycl_pipe::pending(ctx_src->device) && !sycl_pipe::pending(ctx_dst->device)) {
        return false;
    }
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (buf_src->buft != ggml_backend_sycl_buffer_type(ctx_src->device) || buf_dst->buft != ggml_backend_sycl_buffer_type(ctx_dst->device)) {
        return false;
    }
    sycl_pipe::copy_async(ctx_src->device, (const char *) src->data, ctx_dst->device, (char *) dst->data, ggml_nbytes(src));
    return true;
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {''')

rep('''    /* .cpy_tensor_async        = */ NULL, // ggml_backend_sycl_cpy_tensor_async,''',
    '''    /* .cpy_tensor_async        = */ ggml_backend_sycl_cpy_tensor_async_pipe, // (old: ggml_backend_sycl_cpy_tensor_async)''')

open('ggml-sycl.pipe4.cpp', 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print('p2 patched')
