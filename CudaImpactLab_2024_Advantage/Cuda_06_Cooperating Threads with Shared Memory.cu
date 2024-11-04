/*
1. Cooperating Threads with Shared Memory
Example: 1D stencil operation using shared memory and synchronization.
*/
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define RADIUS 3

__global__ void stencil_1d(int *in, int *out, int n) {
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // Load data into shared memory
    if (gindex < n) {
        temp[lindex] = in[gindex];
        if (threadIdx.x < RADIUS) {
            if (gindex >= RADIUS) temp[lindex - RADIUS] = in[gindex - RADIUS];
            if (gindex + BLOCK_SIZE < n) temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
        }
    }

    // Synchronize threads within the block
    __syncthreads();

    // Apply the stencil operation
    if (gindex < n && gindex >= RADIUS && gindex < n - RADIUS) {
        int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            result += temp[lindex + offset];
        }
        out[gindex] = result;
    }
}

int main() {
    int n = 1024;
    int *h_in = (int*)malloc(n * sizeof(int));
    int *h_out = (int*)malloc(n * sizeof(int));
    int *d_in, *d_out;

    for (int i = 0; i < n; i++) h_in[i] = i;

    cudaMalloc((void**)&d_in, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stencil_1d<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    return 0;
}
