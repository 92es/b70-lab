# patch_pipe_p10.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Diagnostics: GGML_SYCL_PIPELINE_TRACE=2 prints the call stack (module+RVA) of the first host waits, to find who
synchronizes. Also: the 20 s report in drain() only fires when no card advanced since the last report.
Input: ggml-sycl.pipe11.cpp. Output: ggml-sycl.pipe12.cpp
"""
b = open('ggml-sycl.pipe11.cpp', 'rb').read().decode('utf-8')
crlf = '\r\n' in b
s = b.replace('\r\n', '\n')
BS = chr(92)
NL = BS + 'n'

def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:80], s.count(a))
    s = s.replace(a, n)

rep('''static void crash_on_abort(int) {''', '''static void print_stack(const char * tag, int max_frames) {
#if defined(_WIN32)
    void * frames[62];
    const USHORT n = CaptureStackBackTrace(1, (ULONG) std::min(max_frames, 62), frames, nullptr);
    for (USHORT i = 0; i < n; ++i) {
        HMODULE mod = nullptr;
        char    path[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) frames[i], &mod) && mod) {
            GetModuleFileNameA(mod, path, MAX_PATH);
        }
        const char * base = strrchr(path, (char) 92);
        fprintf(stderr, "%s   #%02d %s+0x%llx''' + NL + '''", tag, (int) i, base ? base + 1 : path, (unsigned long long) ((const char *) frames[i] - (const char *) mod));
    }
#else
    (void) tag; (void) max_frames;
#endif
}

static void crash_on_abort(int) {''')

rep('''    if (trace() && t_worker_device < 0 && pending(device)) {
        fprintf(stderr, "[SYCL-PIPE] host waits for device %d t=%lld''' + NL + '''", device, (long long) ggml_time_us());
    }''', '''    if (trace() && t_worker_device < 0 && pending(device)) {
        fprintf(stderr, "[SYCL-PIPE] host waits for device %d t=%lld''' + NL + '''", device, (long long) ggml_time_us());
        static std::atomic<int> n_stacks{0};
        static const bool stacks = ggml_sycl_get_env("GGML_SYCL_PIPELINE_TRACE", 0) >= 2;
        if (stacks && n_stacks.fetch_add(1) < 24) {
            print_stack("[SYCL-PIPE-STACK]", 14);
        }
    }''')

# progress-aware stall report
rep('''    while (!w.cv_idle.wait_for(lock, std::chrono::seconds(20), [&] { return !w.busy && w.q.empty(); })) {
        // stall report: a healthy batch finishes a card's part in about a second
        fprintf(stderr, "[SYCL-PIPE] still waiting for device %d (queued=%zu busy=%d);", device, w.q.size(), (int) w.busy);''',
    '''    int    last_sig   = -1;
    size_t last_queue = (size_t) -1;
    while (!w.cv_idle.wait_for(lock, std::chrono::seconds(20), [&] { return !w.busy && w.q.empty(); })) {
        // stall report: only when nothing moved in 20 s (a long queue that is being worked off is not a stall)
        int sig = 0;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            sig = sig * 31 + g_cur_node[i].load() * 7 + g_cur_op[i].load();
        }
        const bool moved = sig != last_sig || w.q.size() != last_queue;
        last_sig   = sig;
        last_queue = w.q.size();
        if (moved) {
            continue;
        }
        fprintf(stderr, "[SYCL-PIPE] still waiting for device %d (queued=%zu busy=%d);", device, w.q.size(), (int) w.busy);''')

open('ggml-sycl.pipe12.cpp', 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print('p10 patched')
