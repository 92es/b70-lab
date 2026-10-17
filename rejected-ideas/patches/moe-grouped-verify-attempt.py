# moe-grouped-verify-attempt.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
p = r'<FORK_SOURCE_DIR>\ggml\src\ggml-sycl\mmvq.cpp'
raw = open(p, encoding='utf-8', newline='').read(); nl = '\r\n' if '\r\n' in raw else '\n'
s = raw.replace('\r\n', '\n')

KERNEL = r'''
// Grouped variant (local, GGML_SYCL_MOE_GROUPED=1): in a multi-token batch (MTP verification) several entries often route to
// the same expert. Each work-group checks the entries before it: entry e owns a group when the number of earlier entries with
// the same expert is a multiple of MOE_GROUP_MAX; the owner walks the expert rows once and produces up to MOE_GROUP_MAX entries
// (its own and the following same-expert ones), so each weight block is fetched once and reused from cache. Other work-groups exit.
#define MOE_GROUP_MAX 8
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_reorder_grouped(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    float * __restrict__ dst_base, const int32_t * __restrict__ ids_dev,
    const int ncols, const int nrows, const int n_entries,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const int src1_row_div,
    const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const int e   = item_ct1.get_group(1);
    const int i02 = ids_dev[e];
    int before = 0;
    for (int j = 0; j < e; ++j) {
        before += ids_dev[j] == i02;
    }
    if (before % MOE_GROUP_MAX != 0) {
        return;   // produced by an earlier owner of this expert
    }
    int members[MOE_GROUP_MAX];
    int ng = 0;
    members[ng++] = e;
    for (int j = e + 1; j < n_entries && ng < MOE_GROUP_MAX; ++j) {
        if (ids_dev[j] == i02) {
            members[ng++] = j;
        }
    }

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }
    const char * vx = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
    const auto   sg = item_ct1.get_sub_group();

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    float partial[MOE_GROUP_MAX];
#pragma unroll
    for (int g = 0; g < MOE_GROUP_MAX; ++g) {
        partial[g] = 0.0f;
    }
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int  ibx       = row * blocks_per_row + i;
        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
        const int  iby       = i * block_type::block_to_q8_1_ratio();
#pragma unroll
        for (int g = 0; g < MOE_GROUP_MAX; ++g) {
            if (g < ng) {
                const char *        vy             = (const char *) vy_base + (size_t) (members[g] / src1_row_div) * src1_row_stride;
                const int8_t *      q8_1_quant_ptr = (const int8_t *) vy + iby * QK8_1;
                const sycl::half2 * q8_1_ds_ptr    = (const sycl::half2 *) ((const char *) vy + ncols + iby * sizeof(sycl::half2));
#pragma unroll
                for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
                    const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);
                    partial[g] += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                }
            }
        }
    }
#pragma unroll
    for (int g = 0; g < MOE_GROUP_MAX; ++g) {
        if (g < ng) {
            const float sum = sycl::reduce_over_group(sg, partial[g], std::plus<>());
            if (sg.leader()) {
                float * dst = (float *) ((char *) dst_base + (size_t) members[g] * dst_row_stride);
                dst[row] = sum;
            }
        }
    }
}
'''

anchor = 'template <typename reorder_vec_dot_q_sycl>\nstatic void launch_mul_mat_vec_q_moe_reorder('
assert s.count(anchor) == 1
s = s.replace(anchor, KERNEL.lstrip('\n') + '\n' + anchor)

old_launch = '''    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, (unsigned) n_experts_used, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>('''
assert s.count(old_launch) == 1
new_launch = '''    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, (unsigned) n_experts_used, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    static const bool grouped = ggml_sycl_get_env("GGML_SYCL_MOE_GROUPED", 0) != 0;
    if (grouped && n_experts_used > 1) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_moe_reorder_grouped<reorder_vec_dot_q_sycl>(
                        vx_base, vy, dst_base, ids_dev, ncols, nrows, n_experts_used,
                        expert_weight_stride, dst_row_stride, src1_row_stride, src1_row_div, item);
                });
        });
        return;
    }
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>('''
s = s.replace(old_launch, new_launch)
open(p, 'w', encoding='utf-8', newline='').write(s.replace('\n', nl))
print('grouped MoE reorder kernel added (GGML_SYCL_MOE_GROUPED=1)')
