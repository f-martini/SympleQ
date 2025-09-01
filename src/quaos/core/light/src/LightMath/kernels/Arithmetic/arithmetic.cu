
#include "symplectic_math.cuh"

__global__ void add_kernel(int a, int b, int *result) { *result = a + b; }

void launch_add_kernel(int a, int b, int *result) {
    int *d_result;
    cudaMalloc(&d_result, sizeof(int));
    add_kernel<<<1, 1>>>(a, b, d_result);
    cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}