/*

2. Error Handling in CUDA
Example: Kernel launch with error handling.
*/
#include <cuda_runtime.h>
#include <iostream>

__global__ void simpleKernel() {
    // Simple dummy kernel
}

int main() {
    simpleKernel<<<1, 1>>>();
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Post-synchronization error: " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}
