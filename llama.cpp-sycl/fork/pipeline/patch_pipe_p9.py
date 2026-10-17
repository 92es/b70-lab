# patch_pipe_p9.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Crash reporter for the pipeline test runtimes: on abort()/std::terminate print the pending C++ exception and the call
stack of the aborting thread as module+RVA (resolve with the linker .map file), and report exceptions that escape a
queued job. Installed when the pipeline starts. Input: ggml-sycl.pipe10.cpp. Output: ggml-sycl.pipe11.cpp
"""
b = open('ggml-sycl.pipe10.cpp', 'rb').read().decode('utf-8')
crlf = '\r\n' in b
s = b.replace('\r\n', '\n')
BS = chr(92)
NL = BS + 'n'

def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:80], s.count(a))
    s = s.replace(a, n)

rep('''#include <regex>
''', '''#include <regex>
#include <csignal>
#include <exception>
#include <stdexcept>
#include <system_error>
#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif
''')

rep('''static void worker_main(int device) {
    t_worker_device = device;''', '''// ---- crash reporter ---------------------------------------------------------------------------------------------
static void crash_report(const char * why) {
    fprintf(stderr, "[SYCL-CRASH] %s on thread of device %d (-1 = not a pipeline worker)''' + NL + '''", why, t_worker_device);
    if (std::exception_ptr e = std::current_exception()) {
        try {
            std::rethrow_exception(e);
        } catch (const sycl::exception & ex) {
            fprintf(stderr, "[SYCL-CRASH] sycl::exception: %s (code %d)''' + NL + '''", ex.what(), (int) ex.code().value());
        } catch (const std::system_error & ex) {
            fprintf(stderr, "[SYCL-CRASH] std::system_error: %s (code %d)''' + NL + '''", ex.what(), (int) ex.code().value());
        } catch (const std::exception & ex) {
            fprintf(stderr, "[SYCL-CRASH] std::exception: %s''' + NL + '''", ex.what());
        } catch (...) {
            fprintf(stderr, "[SYCL-CRASH] unknown exception type''' + NL + '''");
        }
    } else {
        fprintf(stderr, "[SYCL-CRASH] no C++ exception in flight''' + NL + '''");
    }
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        fprintf(stderr, "[SYCL-CRASH] dev%d: node %d %s''' + NL + '''", i, g_cur_node[i].load(), ggml_op_name((ggml_op) g_cur_op[i].load()));
    }
#if defined(_WIN32)
    void * frames[62];
    const USHORT n = CaptureStackBackTrace(0, 62, frames, nullptr);
    for (USHORT i = 0; i < n; ++i) {
        HMODULE mod = nullptr;
        char    path[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) frames[i], &mod) && mod) {
            GetModuleFileNameA(mod, path, MAX_PATH);
        }
        const char * base = strrchr(path, (char) 92);
        fprintf(stderr, "[SYCL-CRASH]   #%02d %s+0x%llx''' + NL + '''", (int) i, base ? base + 1 : path,
                (unsigned long long) ((const char *) frames[i] - (const char *) mod));
    }
#endif
    fflush(stderr);
}

static void crash_on_abort(int) {
    signal(SIGABRT, SIG_DFL);
    crash_report("abort()");
}

static void crash_on_terminate() {
    crash_report("std::terminate");
    signal(SIGABRT, SIG_DFL);
    std::abort();
}

static void crash_install() {
    signal(SIGABRT, crash_on_abort);
    std::set_terminate(crash_on_terminate);   // per thread on this runtime: called on every pipeline thread
}

static void worker_main(int device) {
    t_worker_device = device;
    crash_install();''')

rep('''        job();
        {
            std::lock_guard<std::mutex> lock(w.m);
            w.busy = false;''', '''        try {
            job();
        } catch (...) {
            crash_report("exception escaped a queued job");
            signal(SIGABRT, SIG_DFL);
            std::abort();
        }
        {
            std::lock_guard<std::mutex> lock(w.m);
            w.busy = false;''')

rep('''    const int n = ggml_sycl_info().device_count;
    {
        std::lock_guard<std::mutex> lock(g_start_mutex);
        if (g_started.load(std::memory_order_relaxed)) {
            return;
        }''', '''    const int n = ggml_sycl_info().device_count;
    crash_install();
    {
        std::lock_guard<std::mutex> lock(g_start_mutex);
        if (g_started.load(std::memory_order_relaxed)) {
            return;
        }''')

open('ggml-sycl.pipe11.cpp', 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
print('p9 patched')
