/*
1. Special CUDA Variables for Grid and Block Indexing
Example: Using gridDim, blockDim, blockIdx, and threadIdx in a CUDA kernel.

*/

#include <cuda_runtime.h>
#include <iostream>

__global__ void indexExample() {
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("BlockIdx.x: %d, ThreadIdx.x: %d, GlobalIdx: %d\n", blockIdx.x, threadIdx.x, globalIdx);
}

int main() {
    int numBlocks = 4;
    int threadsPerBlock = 64;
    indexExample << <numBlocks, threadsPerBlock >> > ();
    cudaDeviceSynchronize(); // Ensure all printf calls are completed before ending the program
    return 0;
}
