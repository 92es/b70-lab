// ep_probe: which ways can two Arc cards in ONE process share memory under Windows Level Zero?
//   1. IPC handle across two per-device contexts (what Strata's ep_mem.hpp does on Linux)
//   2. zeDeviceCanAccessPeer + one context holding both devices: card 1 kernel writes card 0's device memory
//   3. host USM in that shared context: both cards' kernels read/write it
// Build: icx -fsycl ep_probe.cpp ze_loader.lib    Run with ONEAPI_DEVICE_SELECTOR=level_zero:A,B
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cstdlib>

using namespace sycl;
static const char* ok(ze_result_t z) { return z == ZE_RESULT_SUCCESS ? "OK" : "FAIL"; }

static int run();
int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try { return run(); } catch (std::exception& e) { std::printf("EXCEPTION: %s\n", e.what()); return 2; }
}
static int run() {
    auto gpus = device::get_devices(info::device_type::gpu);
    if (gpus.size() < 2) { std::printf("need 2 GPUs, have %zu\n", gpus.size()); return 1; }
    device d0 = gpus[0], d1 = gpus[1];
    std::printf("dev0 %s | dev1 %s\n", d0.get_info<info::device::name>().c_str(), d1.get_info<info::device::name>().c_str());
    auto z0 = get_native<backend::ext_oneapi_level_zero>(d0), z1 = get_native<backend::ext_oneapi_level_zero>(d1);
    ze_bool_t p01 = 0, p10 = 0;
    zeDeviceCanAccessPeer(z0, z1, &p01); zeDeviceCanAccessPeer(z1, z0, &p10);
    std::printf("zeDeviceCanAccessPeer 0->1 %d, 1->0 %d\n", (int) p01, (int) p10);
    std::printf("sycl ext_oneapi_can_access_peer 0->1 %d\n", (int) d0.ext_oneapi_can_access_peer(d1));

    const size_t n = 1 << 20, bytes = n * sizeof(float);
    // 1. IPC across per-device contexts
    {
        context c0(d0), c1(d1);
        auto zc0 = get_native<backend::ext_oneapi_level_zero>(c0), zc1 = get_native<backend::ext_oneapi_level_zero>(c1);
        ze_device_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
        void* p = nullptr; void* r = nullptr; ze_ipc_mem_handle_t h{};
        ze_result_t a = zeMemAllocDevice(zc0, &desc, bytes, 4096, z0, &p);
        ze_result_t g = zeMemGetIpcHandle(zc0, p, &h);
        ze_result_t o = zeMemOpenIpcHandle(zc1, z1, h, 0, &r);
        std::printf("1. IPC: alloc %s get %s (0x%x) open %s (0x%x)\n", ok(a), ok(g), (unsigned) g, ok(o), (unsigned) o);
        if (r) zeMemCloseIpcHandle(zc1, r);
        if (p) zeMemFree(zc0, p);
    }
    // 2./3. one context for both devices
    context cs({d0, d1});
    queue q0(cs, d0), q1(cs, d1);
    {
        float* m0 = malloc_device<float>(n, q0);   // lives on card 0
        q0.fill(m0, 1.0f, n).wait();
        bool good = false;
        if (std::getenv("PROBE_PEER")) try {
            q1.parallel_for(range<1>(n), [=](id<1> i) { m0[i] = m0[i] + 2.0f; }).wait();   // card 1 kernel touches card 0 memory
            std::vector<float> h(n); q0.memcpy(h.data(), m0, bytes).wait();
            good = h[0] == 3.0f && h[n - 1] == 3.0f;
        } catch (std::exception& e) { std::printf("2. peer kernel threw: %s\n", e.what()); }
        std::printf("2. shared context, card 1 kernel on card 0 memory: %s\n", good ? "OK" : "FAIL");
        // copy speed card0 -> card1 in the shared context
        try {
            float* m1 = malloc_device<float>(n * 16, q1); float* m0b = malloc_device<float>(n * 16, q0);
            q1.memcpy(m1, m0b, bytes * 16).wait();
            auto t = std::chrono::steady_clock::now();
            for (int k = 0; k < 10; ++k) q1.memcpy(m1, m0b, bytes * 16).wait();
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
            std::printf("   d2d copy 0->1: %.1f GB/s (64 MiB x10)\n", 10.0 * bytes * 16 / s / 1e9);
            free(m1, cs); free(m0b, cs);
        } catch (std::exception& e) { std::printf("   d2d copy 0->1 threw: %s\n", e.what()); }
        free(m0, cs);
    }
    {
        float* hm = malloc_host<float>(n, cs);
        for (size_t i = 0; i < n; ++i) hm[i] = 1.0f;
        q0.parallel_for(range<1>(n), [=](id<1> i) { hm[i] += 1.0f; }).wait();
        q1.parallel_for(range<1>(n), [=](id<1> i) { hm[i] += 1.0f; }).wait();
        std::printf("3. shared-context host USM, both cards write: %s\n", (hm[0] == 3.0f && hm[n - 1] == 3.0f) ? "OK" : "FAIL");
        // a small flag round trip: card 0 writes, card 1 reads, through host memory (latency of one hop)
        auto t = std::chrono::steady_clock::now();
        for (int k = 0; k < 200; ++k) {
            q0.single_task([=]() { hm[0] = (float) k; }).wait();
            q1.single_task([=]() { hm[1] = hm[0]; }).wait();
        }
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count() / 200;
        std::printf("   two-kernel host hop: %.0f us per round (incl. launch+wait)\n", us);
        free(hm, cs);
    }
    // 4. per-device contexts (as Strata), host pages imported into the driver: both cards' kernels use the same pointer
    {
        context c0(d0), c1(d1);
        queue a0(c0, d0), a1(c1, d1);
        auto drv = get_native<backend::ext_oneapi_level_zero>(d0.get_platform());
        using import_fn = ze_result_t (*)(ze_driver_handle_t, void*, size_t);
        using release_fn = ze_result_t (*)(ze_driver_handle_t, void*);
        import_fn imp = nullptr; release_fn rel = nullptr;
        ze_result_t g1 = zeDriverGetExtensionFunctionAddress(drv, "zexDriverImportExternalPointer", (void**) &imp);
        zeDriverGetExtensionFunctionAddress(drv, "zexDriverReleaseImportedPointer", (void**) &rel);
        std::printf("4. zexDriverImportExternalPointer: %s\n", imp ? "found" : ok(g1));
        const size_t hb = 2u << 20;
        float* hp = (float*) _aligned_malloc(hb, 1u << 16);
        for (size_t i = 0; i < hb / 4; ++i) hp[i] = 1.0f;
        ze_result_t zi = imp ? imp(drv, hp, hb) : ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
        std::printf("   import: %s (0x%x)\n", ok(zi), (unsigned) zi);
        if (zi == ZE_RESULT_SUCCESS) {
            const size_t m = hb / 4;
            try {
                a0.parallel_for(range<1>(m), [=](id<1> i) { hp[i] += 1.0f; }).wait();
                a1.parallel_for(range<1>(m), [=](id<1> i) { hp[i] += 1.0f; }).wait();
                std::printf("   card0 ctx + card1 ctx kernels on imported host memory: %s\n",
                            (hp[0] == 3.0f && hp[m - 1] == 3.0f) ? "OK" : "FAIL");
                // device-side spin: card 1 waits for a flag card 0 sets (no host in the loop), many rounds
                volatile int* fl = (volatile int*) hp;
                fl[0] = 0; fl[16] = 0;
                auto t = std::chrono::steady_clock::now();
                const int R = 2000;
                auto e1 = a1.single_task([=]() {
                    for (int k = 1; k <= R; ++k) {
                        while (sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                                sycl::access::address_space::global_space>(*(int*) &fl[0]).load() < k) {}
                        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                         sycl::access::address_space::global_space>(*(int*) &fl[16]).store(k);
                    }
                });
                auto e0 = a0.single_task([=]() {
                    for (int k = 1; k <= R; ++k) {
                        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                         sycl::access::address_space::global_space>(*(int*) &fl[0]).store(k);
                        while (sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                                sycl::access::address_space::global_space>(*(int*) &fl[16]).load() < k) {}
                    }
                });
                e0.wait(); e1.wait();
                double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count() / R;
                std::printf("   device-spin ping-pong card0<->card1 via host: %.2f us per round trip (%d rounds)\n", us, R);
            } catch (std::exception& e) { std::printf("   imported-memory kernels threw: %s\n", e.what()); }
            if (rel) rel(drv, hp);
        }
        _aligned_free(hp);
    }
    return 0;
}
