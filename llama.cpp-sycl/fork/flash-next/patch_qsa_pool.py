# patch_qsa_pool.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Qwen3.8-Flash-Next (qwen4exp) sparse-attention indexer: fuse the per-token block pooling of the cached indexer keys.
The graph gathers EVERY cached key (GET_ROWS, n_kv x 128 floats), copies r strided slices (r CONTs), sums them
(r-1 ADDs) and scales by 1/r: 2r+1 launches whose memory traffic grows linearly with context depth, on each of the
12 attention layers, for every generated token. One kernel now reads each member key once and writes the pooled key.
Summation order matches the graph, so results are identical. GGML_SYCL_DISABLE_QSA_POOL_FUSION=1 turns it off."""
import os, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."


def patch(rel, edits, marker):
    p = os.path.join(ROOT, *rel.split("/"))
    raw = open(p, "rb").read().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    if marker in s:
        print(f"{rel}: already patched"); return
    for anchor, new, where in edits:
        assert s.count(anchor) == 1, f"{rel}: anchor found {s.count(anchor)}x: {anchor[:70]!r}"
        s = s.replace(anchor, new + anchor if where == "before" else anchor + new)
    if crlf:
        s = s.replace("\n", "\r\n")
    open(p, "wb").write(s.encode("utf-8"))
    print(f"{rel}: patched ({len(edits)} edits)")


FUNCS = r'''// ---- qwen4exp QSA indexer: gather + mean-pool of the cached block member keys in one launch ----------------
struct ggml_sycl_qsa_pool_match {
    const ggml_tensor * keys = nullptr;   // [d, n_kv, 1] F16/F32, rows nb[1] apart
    const ggml_tensor * idx  = nullptr;   // I32 [r * n_blocks]
    ggml_tensor *       out  = nullptr;   // SCALE node, F32 [d, n_blocks]
    int                 r    = 0;
    int                 last = -1;
};

static bool ggml_sycl_match_qsa_pool(const ggml_cgraph * graph, int i, ggml_sycl_qsa_pool_match & m) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_QSA_POOL_FUSION", 0) != 0;
    if (disabled || !g_ggml_sycl_enable_fusion) {
        return false;
    }
    const ggml_tensor * g = graph->nodes[i];
    if (g->op != GGML_OP_GET_ROWS || g->type != GGML_TYPE_F32 || g->view_src != nullptr ||
        (g->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(graph, i) != 1) {
        return false;
    }
    const ggml_tensor * keys = g->src[0];
    const ggml_tensor * idx  = g->src[1];
    const int64_t d     = g->ne[0];
    const int64_t n_idx = g->ne[1];
    if ((keys->type != GGML_TYPE_F16 && keys->type != GGML_TYPE_F32) || keys->nb[0] != ggml_type_size(keys->type) ||
        keys->ne[0] != d || keys->ne[2] != 1 || keys->ne[3] != 1 ||
        idx->type != GGML_TYPE_I32 || idx->nb[0] != sizeof(int32_t) || ggml_nelements(idx) != n_idx ||
        g->ne[2] != 1 || g->ne[3] != 1) {
        return false;
    }

    // CONT(slice 0), then {CONT(slice k), ADD}*, then SCALE; views in between are skipped
    const ggml_tensor * members = nullptr;   // the reshape of g that the slices view
    const ggml_tensor * acc     = nullptr;
    int64_t n_blocks = 0;
    int     r        = 0;
    int     j        = i;
    for (;;) {
        j = ggml_sycl_next_compute_node(graph, j);
        if (j < 0) {
            return false;
        }
        const ggml_tensor * node = graph->nodes[j];
        if (node->op == GGML_OP_SCALE) {
            if (r < 2 || node->src[0] != acc || ggml_get_op_params_f32(node, 1) != 0.0f) {
                return false;
            }
            break;
        }
        if (node->op != GGML_OP_CONT || r >= 16) {
            return false;
        }
        const ggml_tensor * slice = node->src[0];
        if (slice->view_src != g || slice->src[0] == nullptr || slice->type != GGML_TYPE_F32 ||
            !ggml_node_has_n_uses(graph, j, 1)) {
            return false;
        }
        if (r == 0) {
            members  = slice->src[0];
            n_blocks = slice->ne[1];
            if (n_blocks < 1 || n_idx % n_blocks != 0) {
                return false;
            }
        }
        const int64_t r_total = n_idx / n_blocks;
        // slice k = member k of every block: offset k*d floats, blocks r_total*d floats apart
        if (slice->src[0] != members || slice->ne[0] != d || slice->ne[1] != n_blocks || slice->ne[2] != 1 ||
            slice->ne[3] != 1 || slice->nb[0] != sizeof(float) || slice->nb[1] != (size_t) (r_total * d) * sizeof(float) ||
            slice->view_offs != (size_t) ((int64_t) r * d) * sizeof(float) || node->ne[0] != d || node->ne[1] != n_blocks) {
            return false;
        }
        if (r == 0) {
            acc = node;
        } else {
            const int ja = ggml_sycl_next_compute_node(graph, j);
            if (ja < 0) {
                return false;
            }
            const ggml_tensor * add = graph->nodes[ja];
            if (add->op != GGML_OP_ADD || add->src[0] != acc || add->src[1] != node || !ggml_node_has_n_uses(graph, ja, 1) ||
                add->ne[0] != d || add->ne[1] != n_blocks) {
                return false;
            }
            acc = add;
            j   = ja;
        }
        r++;
    }
    ggml_tensor * scale = graph->nodes[j];
    if (members == nullptr || (int64_t) r * n_blocks != n_idx || ggml_sycl_tensor_use_count(graph, members) != r ||
        scale->type != GGML_TYPE_F32 || scale->ne[0] != d || scale->ne[1] != n_blocks || !ggml_is_contiguous(scale)) {
        return false;
    }
    const size_t keys_bytes = (size_t) keys->nb[1] * (size_t) keys->ne[1];
    if (!ggml_sycl_ranges_disjoint(scale->data, ggml_nbytes(scale), keys->data, keys_bytes) ||
        !ggml_sycl_ranges_disjoint(scale->data, ggml_nbytes(scale), idx->data, ggml_nbytes(idx))) {
        return false;
    }
    m.keys = keys; m.idx = idx; m.out = scale; m.r = r; m.last = j;
    return true;
}

static void ggml_sycl_qsa_pool(ggml_backend_sycl_context & ctx, const ggml_sycl_qsa_pool_match & m) {
    const char *    keys       = (const char *) m.keys->data;
    const int32_t * idx        = (const int32_t *) m.idx->data;
    float *         out        = (float *) m.out->data;
    const size_t    row_stride = m.keys->nb[1];
    const size_t    d          = (size_t) m.out->ne[0];
    const size_t    n_blocks   = (size_t) m.out->ne[1];
    const int       r          = m.r;
    const float     factor     = ggml_get_op_params_f32(m.out, 0);
    if (m.keys->type == GGML_TYPE_F16) {
        ctx.stream()->parallel_for(sycl::range<2>(n_blocks, d), [=](sycl::id<2> id) {
            const size_t b = id[0];
            const size_t c = id[1];
            float sum = 0.0f;
            for (int k = 0; k < r; ++k) {
                const sycl::half * row = (const sycl::half *) (keys + (size_t) idx[b * r + k] * row_stride);
                sum += (float) row[c];
            }
            out[b * d + c] = sum * factor;
        });
    } else {
        ctx.stream()->parallel_for(sycl::range<2>(n_blocks, d), [=](sycl::id<2> id) {
            const size_t b = id[0];
            const size_t c = id[1];
            float sum = 0.0f;
            for (int k = 0; k < r; ++k) {
                const float * row = (const float *) (keys + (size_t) idx[b * r + k] * row_stride);
                sum += row[c];
            }
            out[b * d + c] = sum * factor;
        });
    }
}

'''

DISPATCH = r'''        if (node->op == GGML_OP_GET_ROWS) {
            ggml_sycl_qsa_pool_match qsa_pool;
            if (ggml_sycl_match_qsa_pool(cgraph, i, qsa_pool)) {
                ggml_sycl_qsa_pool(*sycl_ctx, qsa_pool);
                i = qsa_pool.last;
                continue;
            }
        }

'''

patch("ggml/src/ggml-sycl/ggml-sycl.cpp", [
    ("// Diagnostic per-op timer (GGML_SYCL_PROFILE_OPS=1). It waits on the queue after every op, which\n", FUNCS, "before"),
    ("        if (node->op == GGML_OP_MUL) {\n            ggml_sycl_hc_gate_match hc_gate;\n", DISPATCH, "before"),
], "ggml_sycl_match_qsa_pool")

TEST_STRUCT = r'''// qwen4exp QSA indexer block pooling: GET_ROWS of all member keys -> r strided slices (CONT) -> ADDs -> SCALE(1/r)
struct test_qsa_block_pool : public test_case {
    const ggml_type type;
    const int64_t d, n_kv, r;
    const bool extra_use;
    test_qsa_block_pool(ggml_type type, int64_t d, int64_t n_kv, int64_t r, bool extra_use = false)
        : type(type), d(d), n_kv(n_kv), r(r), extra_use(extra_use) {}
    std::string op_desc(ggml_tensor * t) override { GGML_UNUSED(t); return "QSA_BLOCK_POOL"; }
    bool run_whole_graph() override { return true; }
    std::string vars() override { return VARS_TO_STR5(type, d, n_kv, r, extra_use); }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_blocks = (n_kv + r - 1) / r;
        ggml_tensor * cache = ggml_new_tensor_4d(ctx, type, d, 1, n_kv + 5, 1);   // KV-cache-like: [d, n_head, n_cells]
        ggml_set_name(cache, "cache");
        ggml_tensor * keys = ggml_view_3d(ctx, cache, d, n_kv, 1, cache->nb[2], cache->nb[3], 0);
        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r * n_blocks, 1);
        ggml_set_name(rows, "rows");
        ggml_tensor * members = ggml_get_rows(ctx, keys, rows);
        members = ggml_reshape_4d(ctx, members, d, r, n_blocks, 1);
        ggml_tensor * pooled = nullptr;
        for (int64_t i = 0; i < r; ++i) {
            ggml_tensor * slice = ggml_cont(ctx, ggml_view_3d(ctx, members, d, n_blocks, 1, members->nb[2], members->nb[3], i * members->nb[1]));
            pooled = pooled ? ggml_add(ctx, pooled, slice) : slice;
        }
        pooled = ggml_scale(ctx, pooled, 1.0f / (float) r);
        if (extra_use) { pooled = ggml_add(ctx, pooled, ggml_cont(ctx, ggml_view_3d(ctx, members, d, n_blocks, 1, members->nb[2], members->nb[3], 0))); }
        ggml_set_name(pooled, "out");
        return pooled;
    }
    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_is_view_op(t->op)) { continue; }
            if (t->type == GGML_TYPE_I32) {
                std::vector<int> data(ggml_nelements(t));
                for (size_t i = 0; i < data.size(); i++) { data[i] = (int) (i < (size_t) n_kv ? i : n_kv - 1); }   // tail block repeats the last cell
                ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

'''

TEST_CASES = r'''    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        test_cases.emplace_back(new test_qsa_block_pool(type, 128, 256, 4));
        test_cases.emplace_back(new test_qsa_block_pool(type, 128, 1027, 4));   // incomplete tail block
        test_cases.emplace_back(new test_qsa_block_pool(type, 64, 96, 2));
        test_cases.emplace_back(new test_qsa_block_pool(type, 128, 256, 4, true)); // members has an extra consumer: must not fuse
    }
'''

patch("tests/test-backend-ops.cpp", [
    ("// GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE\n", TEST_STRUCT, "before"),
    ("    test_cases.emplace_back(new test_scale());\n", TEST_CASES, "before"),
], "test_qsa_block_pool")
print("done")
