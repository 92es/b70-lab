# patch_restore_pin.py
#
# Prepared with AI assistance.
# Derived from llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors); see LICENSES/llama.cpp-LICENSE.
# Base: a local llama.cpp fork tree (upstream master snapshot 66041e + the local changes listed in
# llama.cpp-sycl/README.md). This is NOT stock upstream: it will not apply to a fresh clone without porting.
# Status, measured results and the runtime it shipped in: llama.cpp-sycl/README.md.
#
"""Pin the row gather of the DSV4 state restore to the card of its own layer.

The gather has no weights, so the scheduler gave it to the card of the previous layer when the layer is the first one
on its card: state (card N) -> card N-1 -> gather -> card N, on every batch and every decoded token, and the split of
card N-1 was cut in two around it. Inputs: deepseek4.cpp, llama-context.cpp. Outputs: *.pin.cpp
"""
def load(fn):
    b = open(fn, 'rb').read().decode('utf-8')
    return ('\r\n' in b), b.replace('\r\n', '\n')
def save(fn, crlf, s):
    open(fn, 'wb').write((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))
def rep(s, a, n, c=1):
    assert s.count(a) == c, (a[:70], s.count(a))
    return s.replace(a, n)

crlf, s = load('deepseek4.cpp')
s = rep(s, '''static dsv4_state_tensors dsv4_build_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il) {''', '''// pin(t): names the row gather so that the context can place it on the card of layer il (see graph_get_cb)
template <typename F>
static dsv4_state_tensors dsv4_build_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il,
        F && pin) {''')
s = rep(s, '''    ggml_tensor * kv_rows = ggml_get_rows(ctx, restored.kv, inp.state_restore_src_idxs);
    restored.kv = state->cpy_kv(ctx, kv_rows, inp.state_restore_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, restored.score, inp.state_restore_src_idxs);
    restored.score = state->cpy_score(ctx, score_rows, inp.state_restore_dst_idxs, il);''', '''    ggml_tensor * kv_rows = ggml_get_rows(ctx, restored.kv, inp.state_restore_src_idxs);
    pin(kv_rows);
    restored.kv = state->cpy_kv(ctx, kv_rows, inp.state_restore_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, restored.score, inp.state_restore_src_idxs);
    pin(score_rows);
    restored.score = state->cpy_score(ctx, score_rows, inp.state_restore_dst_idxs, il);''')
PIN = ', [&](ggml_tensor * t) { cb(t, "state_restore_rows", il); });'
s = rep(s, '''        const dsv4_state_tensors csa_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_csa(), csa_state, il);''', '''        const dsv4_state_tensors csa_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_csa(), csa_state, il''' + PIN)
import re
m = re.search(r'const dsv4_state_tensors lid_restored = dsv4_build_state_restore\(\n\s*ctx0, inp_dsv4->get_lid\(\), lid_state, il\);', s)
assert m, 'lid call'
s = s[:m.end() - 2] + PIN + s[m.end():]
a = 'hca_restored = dsv4_build_state_restore(ctx0, inp_dsv4->get_hca(), hca_state, il);'
assert s.count(a) == 2
s = s.replace(a, a[:-2] + PIN)
save('deepseek4.pin.cpp', crlf, s)

crlf, s = load('llama-context.cpp')
s = rep(s, '''            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {''',
           '''            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0 || strcmp(name, "state_restore_rows") == 0)) {''')
save('llama-context.pin.cpp', crlf, s)
print('ok')
