# patch_pipe_p7.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Diagnostics: with the trace on, list the nodes of small graphs handed to the pipeline (and of small inline graphs)."""
b = open('ggml-sycl.pipe8.cpp', 'rb').read().decode('utf-8')
crlf = '\r\n' in b
s = b.replace('\r\n', '\n')
BS = chr(92)
def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:80], s.count(a))
    s = s.replace(a, n)

rep('''    if (sycl_pipe::enabled() && sycl_pipe::t_worker_device < 0) {
        sycl_pipe::held_forget_device(sycl_ctx->device);''', '''    if (sycl_pipe::enabled() && sycl_pipe::t_worker_device < 0) {
        if (sycl_pipe::trace() && cgraph->n_nodes <= 16 && sycl_pipe::pending(sycl_ctx->device)) {
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * n = cgraph->nodes[i];
                fprintf(stderr, "[SYCL-PIPE] small graph device=%d node %d/%d %s '%s' [%lld,%lld,%lld,%lld] src0='%s' src1='%s'" "''' + BS + '''n", sycl_ctx->device, i, cgraph->n_nodes,
                        ggml_op_name(n->op), n->name, (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3],
                        n->src[0] ? n->src[0]->name : "", n->src[1] ? n->src[1]->name : "");
            }
        }
        sycl_pipe::held_forget_device(sycl_ctx->device);''')
open('ggml-sycl.pipe9.cpp', 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print('p7 patched')
