# patch_pipe.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
import sys
src, dst = sys.argv[1], sys.argv[2]
b = open(src, 'rb').read().decode('utf-8')
crlf = '\r\n' in b
s = b.replace('\r\n', '\n')

def rep(a, n, c=1):
    global s
    assert s.count(a) == c, (a[:70], s.count(a))
    s = s.replace(a, n)

# ---------------------------------------------------------------------------------------------------------------
# 1. the pipeline core
# ---------------------------------------------------------------------------------------------------------------
CORE = r'''
// ---- cross-card pipeline (GGML_SYCL_PIPELINE=1, default off) ------------------------------------------------------
// With a layer split only one card works at a time, because the host thread runs each card's part of a prompt batch
// to completion before it can prepare the next batch. With the switch on, every card gets a worker thread and a FIFO
// that acts as that card's command stream: a prompt-batch graph is copied and queued instead of executed, so the
// scheduler can go on to the next batch while the cards work. Everything else that touches a card's memory is either
// queued behind the pending work (writes) or waits for it (reads), which keeps the order the scheduler relies on.
// Decode-sized graphs are never queued: they run inline as before (after the FIFO has drained).
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
#include <deque>
#include <functional>
#include <thread>
#include <unordered_map>

namespace sycl_pipe {

struct worker {
    std::thread                        th;
    std::mutex                         m;
    std::condition_variable            cv;
    std::condition_variable            cv_idle;
    std::deque<std::function<void()>>  q;
    bool                               busy    = false;
    bool                               started = false;
};

static worker            g_workers[GGML_SYCL_MAX_DEVICES];
static std::mutex        g_start_mutex;
static std::atomic<bool> g_started{false};
static std::atomic<int>  g_registered{0};
static thread_local int  t_worker_device = -1;   // device whose FIFO this thread serves, -1 on other threads

static bool enabled() {
    static const bool v = ggml_sycl_get_env("GGML_SYCL_PIPELINE", 0) != 0;
    return v;
}

static int min_tokens() {
    static const int v = std::max(2, ggml_sycl_get_env("GGML_SYCL_PIPELINE_MIN_TOKENS", 64));
    return v;
}

// where each device's graph loop is (node index, op); read by the stall report below
static std::atomic<int> g_cur_node[GGML_SYCL_MAX_DEVICES];
static std::atomic<int> g_cur_op[GGML_SYCL_MAX_DEVICES];

static bool trace() {
    static const bool v = ggml_sycl_get_env("GGML_SYCL_PIPELINE_TRACE", 0) != 0;
    return v;
}

static void worker_main(int device) {
    t_worker_device = device;
    {
        // the per-thread device table is a plain map: register this thread while nothing else runs
        std::lock_guard<std::mutex> lock(g_start_mutex);
        dpct::select_device(device);
    }
    g_registered.fetch_add(1);
    worker & w = g_workers[device];
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(w.m);
            w.cv.wait(lock, [&] { return !w.q.empty(); });
            job = std::move(w.q.front());
            w.q.pop_front();
            w.busy = true;
        }
        job();
        {
            std::lock_guard<std::mutex> lock(w.m);
            w.busy = false;
            if (w.q.empty()) {
                w.cv_idle.notify_all();
            }
        }
    }
}

// start one worker per device, once; returns after every worker has registered itself
static void ensure_started() {
    if (g_started.load(std::memory_order_acquire)) {
        return;
    }
    const int n = ggml_sycl_info().device_count;
    {
        std::lock_guard<std::mutex> lock(g_start_mutex);
        if (g_started.load(std::memory_order_relaxed)) {
            return;
        }
        for (int i = 0; i < n; ++i) {
            g_workers[i].started = true;
            g_workers[i].th = std::thread(worker_main, i);
            g_workers[i].th.detach();
        }
    }
    // wait until every worker has registered its device for its thread
    while (g_registered.load() < n) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    g_started.store(true, std::memory_order_release);
    GGML_LOG_INFO("%s: cross-card pipeline enabled: %d worker threads, batch graphs of >= %d tokens are queued\n", __func__, n, min_tokens());
}

static bool active(int device) {
    return g_started.load(std::memory_order_acquire) && device >= 0 && device < GGML_SYCL_MAX_DEVICES && g_workers[device].started;
}

// work queued or running on this device's FIFO (always false on the FIFO's own thread: its jobs run in order already)
static bool pending(int device) {
    if (!active(device) || t_worker_device == device) {
        return false;
    }
    worker & w = g_workers[device];
    std::lock_guard<std::mutex> lock(w.m);
    return w.busy || !w.q.empty();
}

static void drain(int device) {
    if (!active(device) || t_worker_device == device) {
        return;
    }
    worker & w = g_workers[device];
    std::unique_lock<std::mutex> lock(w.m);
    while (!w.cv_idle.wait_for(lock, std::chrono::seconds(20), [&] { return !w.busy && w.q.empty(); })) {
        // stall report: a healthy batch finishes a card's part in about a second
        fprintf(stderr, "[SYCL-PIPE] still waiting for device %d (queued=%zu busy=%d);", device, w.q.size(), (int) w.busy);
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            fprintf(stderr, " dev%d: node %d %s;", i, g_cur_node[i].load(), ggml_op_name((ggml_op) g_cur_op[i].load()));
        }
        fprintf(stderr, "\n");
    }
}

static void enqueue(int device, std::function<void()> job) {
    worker & w = g_workers[device];
    {
        std::lock_guard<std::mutex> lock(w.m);
        w.q.push_back(std::move(job));
    }
    w.cv.notify_one();
}

// a private copy of one split graph: the scheduler reuses the memory of the tensor structs for the next graph
struct graph_copy {
    std::deque<ggml_tensor>                                tensors;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> map;
    std::vector<ggml_tensor *>                             nodes;
    std::vector<int32_t>                                   use_counts;
    std::vector<ggml_bitset_t>                             used;
    std::vector<ggml_tensor *>                             keys;
    ggml_cgraph                                            g;

    ggml_tensor * clone(const ggml_tensor * t) {
        if (t == nullptr) {
            return nullptr;
        }
        auto it = map.find(t);
        if (it != map.end()) {
            return it->second;
        }
        tensors.push_back(*t);
        ggml_tensor * n = &tensors.back();
        map.emplace(t, n);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            n->src[j] = clone(t->src[j]);
        }
        n->view_src = clone(t->view_src);
        return n;
    }

    explicit graph_copy(const ggml_cgraph * cgraph) {
        nodes.reserve(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            nodes.push_back(clone(cgraph->nodes[i]));
        }
        g           = *cgraph;
        g.size      = cgraph->n_nodes;
        g.n_leafs   = 0;
        g.nodes     = nodes.data();
        g.grads     = nullptr;
        g.grad_accs = nullptr;
        g.leafs     = nullptr;

        // use counts are looked up through the visited set by tensor address: rebuild it for the copies
        const size_t hsize = map.size() * 2 + 17;
        used.assign(ggml_bitset_size(hsize), 0);
        keys.assign(hsize, nullptr);
        use_counts.assign(hsize, 0);
        g.visited_hash_set.size = hsize;
        g.visited_hash_set.used = used.data();
        g.visited_hash_set.keys = keys.data();
        g.use_counts            = use_counts.data();
        // every copied tensor that the original set knows (nodes of this split and the tensors they read): the
        // fusion checks ask for the use count of sources too, and a missing entry reads as "unused"
        if (cgraph->use_counts != nullptr) {
            for (const auto & kv : map) {
                const size_t opos = ggml_hash_find(&cgraph->visited_hash_set, kv.first);
                if (opos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, opos)) {
                    continue;
                }
                const size_t pos = ggml_hash_find_or_insert(&g.visited_hash_set, kv.second);
                use_counts[pos]  = cgraph->use_counts[opos];
            }
        }
    }
};

// a prompt batch: the first matrix product of the split has at least min_tokens() columns
static bool is_batch_graph(const ggml_cgraph * cgraph) {
    const ggml_tensor * first_mm = nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[2] != nullptr) {
            // ids are [n_expert_used, n_tokens]
            return node->src[2]->ne[1] >= min_tokens();
        }
        if (node->op == GGML_OP_MUL_MAT && first_mm == nullptr) {
            first_mm = node;
        }
    }
    return first_mm != nullptr && first_mm->ne[1] >= min_tokens() && first_mm->ne[2] == 1 && first_mm->ne[3] == 1;
}

// events of the scheduler: complete when the FIFO has reached the point where they were recorded
struct event_state {
    std::mutex              m;
    std::condition_variable cv;
    bool                    done = false;
};
static std::mutex                                                          g_event_mutex;
static std::unordered_map<ggml_backend_event_t, std::shared_ptr<event_state>> g_events;

static void event_record(int device, ggml_backend_event_t event) {
    if (!active(device)) {
        return;
    }
    auto st = std::make_shared<event_state>();
    {
        std::lock_guard<std::mutex> lock(g_event_mutex);
        g_events[event] = st;
    }
    if (pending(device)) {
        enqueue(device, [st] {
            { std::lock_guard<std::mutex> lock(st->m); st->done = true; }
            st->cv.notify_all();
        });
    } else {
        st->done = true;
    }
}

static void event_wait(ggml_backend_event_t event) {
    std::shared_ptr<event_state> st;
    {
        std::lock_guard<std::mutex> lock(g_event_mutex);
        auto it = g_events.find(event);
        if (it == g_events.end()) {
            return;
        }
        st = it->second;
    }
    std::unique_lock<std::mutex> lock(st->m);
    st->cv.wait(lock, [&] { return st->done; });
}

static void event_forget(ggml_backend_event_t event) {
    std::lock_guard<std::mutex> lock(g_event_mutex);
    g_events.erase(event);
}

} // namespace sycl_pipe

'''
rep('static bool g_sycl_loaded = false;\n', CORE + 'static bool g_sycl_loaded = false;\n')

# ---------------------------------------------------------------------------------------------------------------
# 2. buffer entry points
# ---------------------------------------------------------------------------------------------------------------
# free
rep('''    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    delete ctx;
''', '''    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    sycl_pipe::drain(ctx->device);
    ggml_sycl_set_device(ctx->device);

    delete ctx;
''')

# set_tensor: queue the write behind pending work
rep('''    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32''', '''    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    if (sycl_pipe::pending(ctx->device)) {
        // the card is still working on queued batches: this write must land after them, with its own copy of the data
        auto   held   = std::make_shared<std::vector<char>>((const char *) data, (const char *) data + size);
        char * target = (char *) tensor->data + offset;
        const int device = ctx->device;
        sycl_pipe::enqueue(device, [held, target, device] {
            auto q = &(dpct::dev_mgr::instance().get_device(device).default_queue());
            SYCL_CHECK(CHECK_TRY_ERROR(q->wait()));
            SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(target, held->data(), held->size()).wait()));
        });
        return;
    }
    ggml_sycl_set_device(ctx->device);
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32''')

# set_tensor_2d: weights upload path, just wait
rep('''    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    ggml_sycl_set_device(ctx->device);
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));

    char *       dst = (char *) tensor->data + offset;''', '''    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    sycl_pipe::drain(ctx->device);
    ggml_sycl_set_device(ctx->device);
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));

    char *       dst = (char *) tensor->data + offset;''')

# get_tensor: reads wait for the FIFO
rep('''    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    // Drain the queue before reading: a USM copy''', '''    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    sycl_pipe::drain(ctx->device);
    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    // Drain the queue before reading: a USM copy''')

# cpy_tensor: with the pipeline running, read the source now and queue the write on the destination
rep('''        ggml_backend_sycl_buffer_context * src_ctx = (ggml_backend_sycl_buffer_context *)src->buffer->context;
        ggml_backend_sycl_buffer_context * dst_ctx = (ggml_backend_sycl_buffer_context *)dst->buffer->context;

        ggml_sycl_set_device(src_ctx->device);''', '''        ggml_backend_sycl_buffer_context * src_ctx = (ggml_backend_sycl_buffer_context *)src->buffer->context;
        ggml_backend_sycl_buffer_context * dst_ctx = (ggml_backend_sycl_buffer_context *)dst->buffer->context;

        if (sycl_pipe::active(dst_ctx->device)) {
            // the source must be complete now (the caller may let that card overwrite it next); the destination may
            // still be busy with earlier batches, so its write goes through its FIFO
            sycl_pipe::drain(src_ctx->device);
            if (sycl_pipe::pending(dst_ctx->device)) {
                const size_t nbytes = ggml_nbytes(src);
                auto held = std::make_shared<std::vector<char>>(nbytes);
                queue_ptr q_src = src_ctx->stream;
                SYCL_CHECK(CHECK_TRY_ERROR(q_src->wait()));
                SYCL_CHECK(CHECK_TRY_ERROR(q_src->memcpy(held->data(), (const char *) src->data, nbytes).wait()));
                char * target = (char *) dst->data;
                const int device = dst_ctx->device;
                sycl_pipe::enqueue(device, [held, target, device] {
                    auto q = &(dpct::dev_mgr::instance().get_device(device).default_queue());
                    SYCL_CHECK(CHECK_TRY_ERROR(q->wait()));
                    SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(target, held->data(), held->size()).wait()));
                });
                return true;
            }
        }

        ggml_sycl_set_device(src_ctx->device);''')

# clear / memset: wait
rep('''    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));
''', '''    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    sycl_pipe::drain(ctx->device);
    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));
''')
rep('''    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    if (size == 0) {''', '''    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    sycl_pipe::drain(ctx->device);
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    if (size == 0) {''')

# ---------------------------------------------------------------------------------------------------------------
# 3. backend entry points
# ---------------------------------------------------------------------------------------------------------------
rep('''static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
''', '''static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    sycl_pipe::drain(sycl_ctx->device);
''')

rep('''    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((char *)tensor->data + offset, data, size)));
}''', '''    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    if (sycl_pipe::pending(sycl_ctx->device)) {
        auto   held   = std::make_shared<std::vector<char>>((const char *) data, (const char *) data + size);
        char * target = (char *) tensor->data + offset;
        sycl_pipe::enqueue(sycl_ctx->device, [held, target, stream] {
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(target, held->data(), held->size()).wait()));
        });
        return;
    }
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((char *)tensor->data + offset, data, size)));
}''')

rep('''    static const bool wait_first = ggml_sycl_get_env("GGML_SYCL_GET_TENSOR_WAIT", 1) != 0;
    if (wait_first) {
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));
    }
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (const char *)tensor->data + offset, size)));
}''', '''    static const bool wait_first = ggml_sycl_get_env("GGML_SYCL_GET_TENSOR_WAIT", 1) != 0;
    if (sycl_pipe::pending(sycl_ctx->device)) {
        // asynchronous read: the caller keeps `data` valid until it synchronizes, so the copy can follow the queued work
        const char * source = (const char *) tensor->data + offset;
        sycl_pipe::enqueue(sycl_ctx->device, [data, source, size, stream] {
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(data, source, size).wait()));
        });
        return;
    }
    if (wait_first) {
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));
    }
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (const char *)tensor->data + offset, size)));
}''')

rep('''    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_UNUSED(backend);
}''', '''    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    sycl_pipe::drain(sycl_ctx->device);
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_UNUSED(backend);
}''')

# graph compute
rep('''static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);
''', '''static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

    if (sycl_pipe::enabled() && sycl_pipe::t_worker_device < 0) {
        if (sycl_pipe::is_batch_graph(cgraph)) {
            sycl_pipe::ensure_started();
            auto copy = std::make_shared<sycl_pipe::graph_copy>(cgraph);
            if (sycl_pipe::trace()) {
                fprintf(stderr, "[SYCL-PIPE] queue graph device=%d nodes=%d t=%lld\\n", sycl_ctx->device, cgraph->n_nodes, (long long) ggml_time_us());
            }
            sycl_pipe::enqueue(sycl_ctx->device, [sycl_ctx, copy] {
                const int64_t t0 = ggml_time_us();
                ggml_backend_sycl_graph_compute_impl(sycl_ctx, &copy->g);
                SYCL_CHECK(CHECK_TRY_ERROR(sycl_ctx->stream()->wait()));
                if (sycl_pipe::trace()) {
                    fprintf(stderr, "[SYCL-PIPE] done  graph device=%d start=%lld end=%lld\\n", sycl_ctx->device, (long long) t0, (long long) ggml_time_us());
                }
            });
            return GGML_STATUS_SUCCESS;
        }
        // a decode-sized graph runs inline, after whatever is still queued for this card
        sycl_pipe::drain(sycl_ctx->device);
    }
''')

# events
rep('''    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
''', '''    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
    sycl_pipe::event_record(sycl_ctx->device, event);
''')
rep('''    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    if (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));''', '''    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    if (ggml_backend_is_sycl(backend)) {
        sycl_pipe::event_wait(event);
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));''')
rep('''  if (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }
''', '''  sycl_pipe::event_forget(event);
  if (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }
''')

# ---------------------------------------------------------------------------------------------------------------
# 4. per-device graph serial (two cards can now be inside a graph at the same time)
# ---------------------------------------------------------------------------------------------------------------
rep('static std::atomic<uint64_t> g_ggml_sycl_graph_serial{0};\n',
    'static std::atomic<uint64_t> g_ggml_sycl_graph_serial[GGML_SYCL_MAX_DEVICES];   // per device: cards may compute concurrently\n')
rep('    const uint64_t graph_serial = g_ggml_sycl_graph_serial.load();\n',
    '    const uint64_t graph_serial = g_ggml_sycl_graph_serial[ctx.device].load();\n')
rep('    g_ggml_sycl_graph_serial.fetch_add(1);\n',
    '    g_ggml_sycl_graph_serial[sycl_ctx->device].fetch_add(1);\n')

rep('''        ggml_sycl_profile_node profile_node(sycl_ctx, node, profile_nodes ? &profile_totals : nullptr, &profile_mm);
''', '''        sycl_pipe::g_cur_node[sycl_ctx->device].store(i, std::memory_order_relaxed);
        sycl_pipe::g_cur_op[sycl_ctx->device].store((int) node->op, std::memory_order_relaxed);
        ggml_sycl_profile_node profile_node(sycl_ctx, node, profile_nodes ? &profile_totals : nullptr, &profile_mm);
''')

out = s.replace('\n', '\r\n') if crlf else s
open(dst, 'wb').write(out.encode('utf-8'))
print('patched ok, crlf', crlf, 'lines', s.count('\n'))
