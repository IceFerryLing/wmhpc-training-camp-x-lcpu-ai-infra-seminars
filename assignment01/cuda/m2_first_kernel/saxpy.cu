#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = (call);                                           \
        if (error != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error));                               \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

__global__ void saxpy(const float *x, float *y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; index < n; index += stride) y[index] = 2.0f * x[index] + y[index];
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <n>\n", argv[0]);
        return 2;
    }

    errno = 0;
    char *end = nullptr;
    long parsed = strtol(argv[1], &end, 10);
    if (errno != 0 || *argv[1] == '\0' || *end != '\0' || parsed < 0 ||
        parsed > INT_MAX) {
        fprintf(stderr, "n must be an integer in [0, %d]\n", INT_MAX);
        return 2;
    }

    int n = static_cast<int>(parsed);
    if (n == 0) {
        printf("SUM=0\n");
        return 0;
    }

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *h_x = static_cast<float *>(malloc(bytes));
    float *h_y = static_cast<float *>(malloc(bytes));
    if (h_x == nullptr || h_y == nullptr) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }
    for (int index = 0; index < n; ++index) {
        h_x[index] = ((index % 2048) - 1024) * 0.5f;
        h_y[index] = static_cast<float>((index % 1024) - 512);
    }

    float *d_x = nullptr;
    float *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    saxpy<<<blocks, threads>>>(d_x, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int index = 0; index < n; ++index) sum += h_y[index];
    printf("SUM=%.0f n=%d kernel_ms=%.4f\n", sum, n, milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    free(h_y);
    return 0;
}
