# patch_pipe_p5.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Diagnostics: label every queued job; with GGML_SYCL_PIPELINE_TRACE=1 report jobs that took more than 30 ms."""
b = open('ggml-sycl.pipe6.cpp', 'rb').read().decode('utf-8')
crlf = '\r\n' in b
s = b.replace('\r\n', '\n')
BS = chr(92)

def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:80], s.count(a))
    s = s.replace(a, n)

rep('''static void enqueue(int device, std::function<void()> job) {
    worker & w = g_workers[device];
    {
        std::lock_guard<std::mutex> lock(w.m);
        w.q.push_back(std::move(job));
    }
    w.cv.notify_one();
}''', '''static void enqueue(int device, std::function<void()> job, const char * label = "job") {
    worker & w = g_workers[device];
    if (trace()) {
        const int64_t tq = ggml_time_us();
        job = [job, label, device, tq] {
            const int64_t t0 = ggml_time_us();
            job();
            const int64_t t1 = ggml_time_us();
            if (t1 - t0 > 30000) {
                fprintf(stderr, "[SYCL-PIPE] slow job device=%d label=%s queued=%lld start=%lld end=%lld''' + BS + '''n", device, label,
                        (long long) tq, (long long) t0, (long long) t1);
            }
        };
    }
    {
        std::lock_guard<std::mutex> lock(w.m);
        w.q.push_back(std::move(job));
    }
    w.cv.notify_one();
}''')

rep('''        enqueue(dev_src, read);''', '''        enqueue(dev_src, read, "cpy-read");''')
rep('''        enqueue(dev_dst, write);''', '''        enqueue(dev_dst, write, "cpy-write");''')

# the remaining sites take a lambda that ends with "});" : add the label by position
import re
def label_site(start_text, label):
    global s
    i = s.index(start_text)
    assert s.count(start_text) == 1, (start_text, s.count(start_text))
    # find the matching close of this enqueue call
    depth = 0
    j = s.index('(', i)
    k = j
    while True:
        c = s[k]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                break
        k += 1
    s = s[:k] + ', "' + label + '"' + s[k:]

label_site('enqueue(device, [st] {', 'event')
label_site('sycl_pipe::enqueue(device, [held, target, device] {', 'buf-set')
label_site('sycl_pipe::enqueue(sycl_ctx->device, [held, target, stream] {', 'set')
label_site('sycl_pipe::enqueue(sycl_ctx->device, [data, source, size, stream] {', 'get')
label_site('sycl_pipe::enqueue(sycl_ctx->device, [sycl_ctx, copy] {', 'graph')

open('ggml-sycl.pipe7.cpp', 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print('p5 patched')
