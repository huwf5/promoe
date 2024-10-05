#include <cstdint>
#include <cuda_runtime.h>
#include "utils.hpp"
#include <cuda/std/chrono>

__global__ void sleep_kernel_nanosleep(uint us) {
    __nanosleep(us * 1000);
}

__global__ void sleep_kernel_chrono(uint us) {
    auto start = cuda::std::chrono::high_resolution_clock::now();
    while (cuda::std::chrono::high_resolution_clock::now() - start < cuda::std::chrono::microseconds(us)) {
        __syncthreads();
    }
}

void cuda_sleep(uint us, int64_t stream) {
    sleep_kernel_chrono<<<1, 1, 0, (cudaStream_t)stream>>>(us);
}
