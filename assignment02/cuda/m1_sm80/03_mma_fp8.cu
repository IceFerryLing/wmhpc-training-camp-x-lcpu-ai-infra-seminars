#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "../common.h"


__device__ static void load_manual(const uint8_t* A, const uint8_t* B,
                                   unsigned (&a)[4], unsigned (&b)[2]) {
    const int lane = threadIdx.x;
    const int group = lane >> 2;
    const int tig = lane & 3;

    for (int i = 0; i < 16; ++i) {
        const int row = group + 8 * ((i % 8) / 4);
        const int col = tig * 4 + (i / 8) * 16 + (i % 4);
        const unsigned byte = static_cast<unsigned>(A[row * 32 + col]);
        a[i / 4] |= byte << (8 * (i % 4));
    }
    for (int i = 0; i < 8; ++i) {
        const int row = (i % 4) + tig * 4 + (i / 4) * 16;
        const int col = group;
        const unsigned byte = static_cast<unsigned>(B[row * 8 + col]);
        b[i / 4] |= byte << (8 * (i % 4));
    }


}

__global__ static void mma_kernel(const uint8_t* A, const uint8_t* B,
                                  float* D) {
    unsigned a[4] = {0, 0, 0, 0};
    unsigned b[2] = {0, 0};
    load_manual(A, B, a, b);

    float c[4] = {0.f, 0.f, 0.f, 0.f};
    float d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}," 
        "{%4,%5,%6,%7},"
        "{%8,%9},      " 
        "{%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
          "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );

    const int group = threadIdx.x >> 2;
    const int tig = threadIdx.x & 3;
    D[group * 8 + tig * 2] = d[0];
    D[group * 8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

static uint8_t encode_fp8(float x) {
    __nv_fp8_e4m3 v = __nv_fp8_e4m3(x);
    return *reinterpret_cast<const uint8_t*>(&v);
}

static float decode_fp8(uint8_t raw) {
    __nv_fp8_e4m3 v = *reinterpret_cast<__nv_fp8_e4m3*>(&raw);
    return static_cast<float>(v);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::printf("FAIL: usage: %s SEED\n", argv[0]);
        return 1;
    }
    const unsigned seed = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-8, 8);

    uint8_t hA[16 * 32], hB[32 * 8];
    float fA[16 * 32], fB[32 * 8], ref[16 * 8] = {};
    for (int i = 0; i < 16 * 32; ++i) {
        hA[i] = encode_fp8(static_cast<float>(dist(rng)));
        fA[i] = decode_fp8(hA[i]);
    }
    for (int i = 0; i < 32 * 8; ++i) {
        hB[i] = encode_fp8(static_cast<float>(dist(rng)));
        fB[i] = decode_fp8(hB[i]);
    }
    for (int r = 0; r < 16; ++r)
        for (int n = 0; n < 8; ++n)
            for (int k = 0; k < 32; ++k)
                ref[r * 8 + n] += fA[r * 32 + k] * fB[k * 8 + n];

    uint8_t *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, sizeof(ref)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    mma_kernel<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();

    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < 16 * 8; ++i) {
        if (got[i] != ref[i]) {
            if (bad < 5)
                std::printf("MISMATCH at %d: got %.9g, want %.9g\n", i,
                            got[i], ref[i]);
            ++bad;
        }
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
    if (bad) {
        std::printf("FAIL: %ld / 128 mismatches\n", bad);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
