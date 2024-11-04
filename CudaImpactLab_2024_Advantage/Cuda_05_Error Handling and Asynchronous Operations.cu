/*
5. Error Handling and Asynchronous Operations
Example: Error checking after kernel launches.
*/
#include <cuda_runtime.h>
#include <iostream>

__global__ void dummyKernel() {
    // Simulate a simple task
}

int main() {
    // Launch kernel
    dummyKernel<<<1, 1>>>();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    // Synchronize device
    cudaDeviceSynchronize();

    // Check for post-synchronization errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Synchronization Error: " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}
