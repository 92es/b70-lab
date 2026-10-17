# patch_f32mmv.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""SYCL: dedicated F32 x F32 mat-vec for small-batch MUL_MAT (src1 ne[1] <= 8).

F32 weights (MoE routers, DeltaNet alpha/beta, hyper-connection inject, indexer) are not handled by DMMV/MMVQ, so
every such matmul went through the oneMKL GEMM path with ~30-40 us of fixed cost regardless of size. This kernel is
one sub-group per weight row, fp32 accumulation, float4 loads when aligned. GGML_SYCL_DISABLE_F32_MMV=1 turns it off.
"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "ggml-sycl.cpp"
raw = p.read_bytes().decode("utf-8"); crlf = "\r\n" in raw; s = raw.replace("\r\n", "\n")


def rep(old, new):
    global s
    n = s.count(old); assert n == 1, f"anchor {n}x: {old[:140]}"
    s = s.replace(old, new)


kernel = r'''
// F32 weight x F32 activations, 1..8 activation columns. One sub-group per weight row; each lane sums a strided
// slice of the row (float4 when the row and the activation columns are 16-byte aligned), then a sub-group
// reduction. fp32 throughout, so results match the GEMM path up to summation order.
#define GGML_SYCL_F32_MMV_MAX_COLS 8
static bool ggml_sycl_can_use_f32_mmv(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool disabled = ggml_sycl_get_env("GGML_SYCL_DISABLE_F32_MMV", 0) != 0;
    if (disabled) {
        return false;
    }
    return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] >= 1 && src1->ne[1] <= GGML_SYCL_F32_MMV_MAX_COLS &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
           src0->ne[0] == src1->ne[0] && !ggml_backend_buffer_is_sycl_split(src0->buffer);
}

template <int ncols_y>
static void mul_mat_vec_f32_kernel(const float * __restrict__ w, const float * __restrict__ y, float * __restrict__ dst,
                                   const int ncols, const int nrows, const int64_t w_stride, const int64_t y_stride,
                                   const int64_t dst_stride, const bool vec4, const sycl::nd_item<3> & item) {
    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (row >= nrows) {
        return;
    }
    const int lane = item.get_local_id(2);
    const float * wr = w + (int64_t) row * w_stride;
    float sum[ncols_y];
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
        sum[j] = 0.0f;
    }
    if (vec4) {
        const sycl::float4 * w4 = reinterpret_cast<const sycl::float4 *>(wr);
        for (int c = lane; c < ncols / 4; c += WARP_SIZE) {
            const sycl::float4 wv = w4[c];
#pragma unroll
            for (int j = 0; j < ncols_y; ++j) {
                const sycl::float4 yv = reinterpret_cast<const sycl::float4 *>(y + j * y_stride)[c];
                sum[j] += wv.x() * yv.x() + wv.y() * yv.y() + wv.z() * yv.z() + wv.w() * yv.w();
            }
        }
    } else {
        for (int c = lane; c < ncols; c += WARP_SIZE) {
            const float wv = wr[c];
#pragma unroll
            for (int j = 0; j < ncols_y; ++j) {
                sum[j] += wv * y[j * y_stride + c];
            }
        }
    }
    const auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
        const float total = sycl::reduce_over_group(sg, sum[j], sycl::plus<float>());
        if (lane == 0) {
            dst[j * dst_stride + row] = total;
        }
    }
}

template <int ncols_y>
static void launch_mul_mat_vec_f32(const float * w, const float * y, float * dst, int ncols, int nrows, int64_t w_stride,
                                   int64_t y_stride, int64_t dst_stride, bool vec4, dpct::queue_ptr stream) {
    constexpr int rows_per_wg = 4;
    const sycl::range<3> block_dims(1, rows_per_wg, WARP_SIZE);
    const sycl::range<3> block_nums(1, 1, (nrows + rows_per_wg - 1) / rows_per_wg);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_f32_kernel<ncols_y>(w, y, dst, ncols, nrows, w_stride, y_stride, dst_stride,
                                                             vec4, item);
                         });
}

static void ggml_sycl_mul_mat_vec_f32(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                      ggml_tensor * dst) {
    const float * w = (const float *) src0->data;
    const float * y = (const float *) src1->data;
    float *       d = (float *) dst->data;
    const int     ncols      = (int) src0->ne[0];
    const int     nrows      = (int) src0->ne[1];
    const int64_t w_stride   = src0->nb[1] / sizeof(float);
    const int64_t y_stride   = src1->nb[1] / sizeof(float);
    const int64_t dst_stride = dst->nb[1] / sizeof(float);
    const bool vec4 = ncols % 4 == 0 && w_stride % 4 == 0 && y_stride % 4 == 0 &&
                      ((uintptr_t) w % 16) == 0 && ((uintptr_t) y % 16) == 0;
    dpct::queue_ptr stream = ctx.stream();
    switch (src1->ne[1]) {
        case 1: launch_mul_mat_vec_f32<1>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 2: launch_mul_mat_vec_f32<2>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 3: launch_mul_mat_vec_f32<3>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 4: launch_mul_mat_vec_f32<4>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 5: launch_mul_mat_vec_f32<5>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 6: launch_mul_mat_vec_f32<6>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 7: launch_mul_mat_vec_f32<7>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        case 8: launch_mul_mat_vec_f32<8>(w, y, d, ncols, nrows, w_stride, y_stride, dst_stride, vec4, stream); break;
        default: GGML_ABORT("unsupported ncols_y for F32 mat-vec");
    }
}

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
'''
rep("\nstatic void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {\n",
    kernel)

# dispatch: right after the scope debug print at the top of ggml_sycl_mul_mat (before split handling)
rep("""static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
""", """static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    if (dst->op == GGML_OP_MUL_MAT && ggml_sycl_can_use_f32_mmv(src0, src1, dst)) {
        ggml_sycl_mul_mat_vec_f32(ctx, src0, src1, dst);
        return;
    }
""")
if crlf:
    s = s.replace("\n", "\r\n")
p.write_bytes(s.encode("utf-8"))
print("added F32 mat-vec path")
