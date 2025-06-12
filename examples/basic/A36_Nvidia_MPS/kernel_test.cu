#include <stdio.h>
#include <cuda_runtime.h>

__global__ void compute_kernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];
    for (int i = 0; i < 1e7; ++i) {
        val = sinf(val) + cosf(val);
    }
    data[idx] = val;
}

int main() {
    const int N = 1024 * 256;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    cudaMemset(d_data, 0, N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    compute_kernel<<<N / 256, 256>>>(d_data);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Run time: %.2f ms\n", ms);

    cudaFree(d_data);
    return 0;
}
