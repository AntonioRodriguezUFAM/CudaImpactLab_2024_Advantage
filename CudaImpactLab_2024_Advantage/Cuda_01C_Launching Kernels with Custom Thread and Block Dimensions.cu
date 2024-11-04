/*
4. Launching Kernels with Custom Thread and Block Dimensions
Example: Using dim3 to configure a kernel launch.

*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void customDimKernel() {
    int globalIdxX = threadIdx.x + blockIdx.x * blockDim.x;
    int globalIdxY = threadIdx.y + blockIdx.y * blockDim.y;
    int globalIdxZ = threadIdx.z + blockIdx.z * blockDim.z;
    printf("GlobalIdx: (%d, %d, %d)\n", globalIdxX, globalIdxY, globalIdxZ);
}

int main() {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(4, 4, 4);
    customDimKernel << <gridSize, blockSize >> > ();
    cudaDeviceSynchronize();
    return 0;
}
