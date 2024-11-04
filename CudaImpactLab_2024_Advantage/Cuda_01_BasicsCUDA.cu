/*1. Basics of CUDA Programming
Example: A simple "Hello, World!" program using CUDA.*/

#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello, World from GPU!\n");
}

int main() {
    // Launch the kernel with a single thread.
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize(); // Ensure all threads complete before finishing.

    return 0;
}
