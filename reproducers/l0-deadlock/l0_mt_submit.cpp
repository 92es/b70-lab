// Minimal test: several host threads, each submitting kernels to its own in-order SYCL queue on a different GPU.
// Observed with a larger program (llama.cpp SYCL backend with one worker thread per device): two threads block
// forever inside zeCommandListAppendLaunchKernelWithArguments, one in RtlAcquireSRWLockExclusive and one in
// RtlAcquireSRWLockShared (ze_intel_gpu64.dll). This program checks whether plain SYCL code shows the same hang.
//
// usage: l0_mt_submit [threads=2] [seconds=120] [mode=0] [flags=0]
//   flags: 1 = all threads use device 0 (one queue each), 2 = one mutex around every oneMKL call, 4 = start at device 1,
//          8 = float gemm only, 16 = half gemm only,
//          32 = warm-up: before the threads start, the main thread runs every gemm size once on every queue
//   mode 0: every thread submits the same set of kernels
//   mode 1: threads also create new kernels while others submit (distinct kernel types per thread)
//   mode 2: mode 1 plus USM memcpy to and from host between kernels
//   mode 3: (build with USE_MKL) mode 2 plus oneMKL gemm calls of changing size, float and half
// exit code 0 = finished, 2 = a thread made no progress for 30 s (hang)
#include <sycl/sycl.hpp>
#ifdef USE_MKL
#include <oneapi/mkl.hpp>
#endif

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

static const size_t N = 1 << 16;

template <int K>
static sycl::event launch(sycl::queue & q, float * a, float * b) {
    return q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        float v = a[i];
        for (int k = 0; k < K + 1; ++k) {
            v = v * 1.0001f + b[i];
        }
        a[i] = v;
    });
}

static sycl::event launch_any(int which, sycl::queue & q, float * a, float * b) {
    switch (which & 7) {
        case 0: return launch<0>(q, a, b);
        case 1: return launch<1>(q, a, b);
        case 2: return launch<2>(q, a, b);
        case 3: return launch<3>(q, a, b);
        case 4: return launch<4>(q, a, b);
        case 5: return launch<5>(q, a, b);
        case 6: return launch<6>(q, a, b);
        default: return launch<7>(q, a, b);
    }
}

int main(int argc, char ** argv) {
    const int n_threads_req = argc > 1 ? atoi(argv[1]) : 2;
    const int seconds       = argc > 2 ? atoi(argv[2]) : 120;
    const int mode          = argc > 3 ? atoi(argv[3]) : 0;
    const int flags         = argc > 4 ? atoi(argv[4]) : 0;
    static std::mutex mkl_mutex;

    std::vector<sycl::device> devs;
    for (const auto & d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            devs.push_back(d);
        }
    }
    if (devs.empty()) {
        printf("no Level Zero GPU found\n");
        return 1;
    }
    const int n_threads = std::max(1, n_threads_req);
    printf("devices=%zu threads=%d seconds=%d mode=%d flags=%d\n", devs.size(), n_threads, seconds, mode, flags);
    for (size_t i = 0; i < devs.size(); ++i) {
        printf("  dev%zu: %s, driver %s\n", i, devs[i].get_info<sycl::info::device::name>().c_str(),
               devs[i].get_info<sycl::info::device::driver_version>().c_str());
    }

    // one queue per thread; thread t uses device t % n_devices (default context: shared by the platform's devices)
    std::vector<sycl::queue> queues;
    for (int t = 0; t < n_threads; ++t) {
        const size_t di = (flags & 1) ? 0 : (t + ((flags & 4) ? 1 : 0)) % devs.size();
        queues.emplace_back(devs[di], sycl::property::queue::in_order());
    }

#ifdef USE_MKL
    if (mode >= 3 && (flags & 32)) {
        for (auto & q : queues) {
            const int64_t M  = 512;
            float *       fa = sycl::malloc_device<float>(M * M, q);
            float *       fc = sycl::malloc_device<float>(M * M, q);
            sycl::half *  ha = sycl::malloc_device<sycl::half>(M * M, q);
            sycl::half *  hc = sycl::malloc_device<sycl::half>(M * M, q);
            q.fill(fa, 0.5f, M * M);
            q.fill(ha, sycl::half(0.5f), M * M).wait();
            for (long long it = 7; it < 8 * 13 * 9 * 6 * 2; it += 8) {
                const int64_t m = 64 + 32 * ((it >> 3) % 13), n = 16 + 16 * ((it >> 5) % 9), k = 128 + 64 * ((it >> 4) % 6);
                oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, m, n, k, sycl::half(1.0f), ha, k, ha, k, sycl::half(0.0f), hc, m);
                oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, 1.0f, fa, m, fa, k, 0.0f, fc, m);
            }
            q.wait();
            sycl::free(fa, q); sycl::free(fc, q); sycl::free(ha, q); sycl::free(hc, q);
        }
        printf("warm-up done" "\n");
        fflush(stdout);
    }
#endif
    std::vector<std::atomic<long long>> progress(n_threads);
    for (auto & p : progress) {
        p.store(0);
    }
    std::atomic<bool> stop{false};

    std::vector<std::thread> workers;
    for (int t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, t] {
            sycl::queue & q = queues[t];
            float * a = sycl::malloc_device<float>(N, q);
            float * b = sycl::malloc_device<float>(N, q);
            std::vector<float> host(N, 1.0f);
            q.memcpy(a, host.data(), N * sizeof(float));
            q.memcpy(b, host.data(), N * sizeof(float)).wait();
#ifdef USE_MKL
            const int64_t M  = 512;
            float *       fa = sycl::malloc_device<float>(M * M, q);
            float *       fb = sycl::malloc_device<float>(M * M, q);
            float *       fc = sycl::malloc_device<float>(M * M, q);
            sycl::half *  ha = sycl::malloc_device<sycl::half>(M * M, q);
            sycl::half *  hb = sycl::malloc_device<sycl::half>(M * M, q);
            sycl::half *  hc = sycl::malloc_device<sycl::half>(M * M, q);
            q.fill(fa, 0.5f, M * M); q.fill(fb, 0.25f, M * M);
            q.fill(ha, sycl::half(0.5f), M * M); q.fill(hb, sycl::half(0.25f), M * M).wait();
#endif
            long long it = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                // mode 0: kernel 0 only. modes 1/2: walk through the kernel types, offset per thread, so that
                // first-time kernel creation on one thread overlaps with launches on the others.
                const int which = mode == 0 ? 0 : (int) ((it / 512 + t * 3) & 7);
                sycl::event e = launch_any(which, q, a, b);
                if (mode >= 2 && (it & 31) == 31) {
                    q.memcpy(host.data(), a, 4096);
                    q.memcpy(b, host.data(), 4096);
                }
#ifdef USE_MKL
                if (mode >= 3 && (it & 7) == 7) {
                    // sizes change so that the library picks different kernels over time
                    const int64_t m = 64 + 32 * ((it >> 3) % 13), n = 16 + 16 * ((it >> 5) % 9), k = 128 + 64 * ((it >> 4) % 6);
                    std::unique_lock<std::mutex> lk(mkl_mutex, std::defer_lock);
                    if (flags & 2) {
                        lk.lock();
                    }
                    const bool use_half = (flags & 16) ? true : (flags & 8) ? false : (((it >> 3) & 1) != 0);
                    if (use_half) {
                        e = oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, m, n, k, sycl::half(1.0f), ha, k, hb, k, sycl::half(0.0f), hc, m);
                    } else {
                        e = oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, 1.0f, fa, m, fb, k, 0.0f, fc, m);
                    }
                }
#endif
                if ((it & 63) == 63) {
                    e.wait();
                }
                progress[t].store(++it, std::memory_order_relaxed);
            }
            q.wait();
            sycl::free(a, q);
            sycl::free(b, q);
        });
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<long long> last(n_threads, 0);
    std::vector<int>       stalled_s(n_threads, 0);
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        const int elapsed = (int) std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count();
        int worst = 0;
        printf("t=%4ds", elapsed);
        for (int t = 0; t < n_threads; ++t) {
            const long long p = progress[t].load();
            stalled_s[t] = p == last[t] ? stalled_s[t] + 5 : 0;
            printf("  thr%d=%lld%s", t, p, stalled_s[t] > 0 ? "(stuck)" : "");
            worst   = std::max(worst, stalled_s[t]);
            last[t] = p;
        }
        printf("\n");
        fflush(stdout);
        if (worst >= 30) {
            printf("HANG: a thread made no progress for %d s\n", worst);
            fflush(stdout);
            std::_Exit(2);
        }
        if (elapsed >= seconds) {
            break;
        }
    }
    stop.store(true);
    for (auto & w : workers) {
        w.join();
    }
    printf("finished without a hang\n");
    return 0;
}
