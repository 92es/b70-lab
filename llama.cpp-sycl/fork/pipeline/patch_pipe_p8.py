# patch_pipe_p8.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Diagnostics: name the tensors of card-to-card copies that could not use a queued host write."""
b = open('ggml-sycl.pipe9.cpp', 'rb').read().decode('utf-8')
crlf = '\r\n' in b
s = b.replace('\r\n', '\n')
BS = chr(92)
def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:80], s.count(a))
    s = s.replace(a, n)

rep('''static std::shared_ptr<std::vector<char>> held_find(int device, const char * src, size_t nbytes) {
    std::lock_guard<std::mutex> lock(g_held_mutex);
    auto it = g_held.find(src);''', '''static void held_report(const char * what, const ggml_tensor * src, int dev_src, int dev_dst) {
    if (!trace()) {
        return;
    }
    size_t held_size = 0;
    int    held_dev  = -1;
    {
        std::lock_guard<std::mutex> lock(g_held_mutex);
        auto it = g_held.find((const char *) src->data);
        if (it != g_held.end()) {
            held_size = it->second.data->size();
            held_dev  = it->second.device;
        }
    }
    fprintf(stderr, "[SYCL-PIPE] copy %s '%s' op=%s view=%d dev%d->dev%d nbytes=%zu held_size=%zu held_dev=%d src_pending=%d" "''' + BS + '''n", what, src->name,
            ggml_op_name(src->op), (int) (src->view_src != nullptr), dev_src, dev_dst, ggml_nbytes(src), held_size, held_dev, (int) pending(dev_src));
}

static std::shared_ptr<std::vector<char>> held_find(int device, const char * src, size_t nbytes) {
    std::lock_guard<std::mutex> lock(g_held_mutex);
    auto it = g_held.find(src);''')

rep('''            sycl_pipe::copy_async(src_ctx->device, (const char *) src->data, dst_ctx->device, (char *) dst->data, ggml_nbytes(src));
            return true;''', '''            sycl_pipe::held_report("buf", src, src_ctx->device, dst_ctx->device);
            sycl_pipe::copy_async(src_ctx->device, (const char *) src->data, dst_ctx->device, (char *) dst->data, ggml_nbytes(src));
            return true;''')
rep('''    sycl_pipe::copy_async(ctx_src->device, (const char *) src->data, ctx_dst->device, (char *) dst->data, ggml_nbytes(src));
    return true;''', '''    sycl_pipe::held_report("async", src, ctx_src->device, ctx_dst->device);
    sycl_pipe::copy_async(ctx_src->device, (const char *) src->data, ctx_dst->device, (char *) dst->data, ggml_nbytes(src));
    return true;''')
open('ggml-sycl.pipe10.cpp', 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print('p8 patched')
