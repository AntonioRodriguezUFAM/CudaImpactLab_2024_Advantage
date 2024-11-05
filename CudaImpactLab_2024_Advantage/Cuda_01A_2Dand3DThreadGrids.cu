/*

2. 2D and 3D Thread Grids
Example: Launching a kernel with a 2D grid and a 2D block layout.
*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel2DGrid() {
    int globalIdxX = threadIdx.x + blockIdx.x * blockDim.x;
    int globalIdxY = threadIdx.y + blockIdx.y * blockDim.y;
    printf("BlockIdx: (%d, %d), ThreadIdx: (%d, %d), GlobalIdx: (%d, %d)\n",
        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, globalIdxX, globalIdxY);
}

__global__ void kernel3DGrid() {
    int globalIdxX = threadIdx.x + blockIdx.x * blockDim.x;
    int globalIdxY = threadIdx.y + blockIdx.y * blockDim.y;
    int globalIdxZ = threadIdx.z + blockIdx.z * blockDim.z;
    printf("BlockIdx: (%d, %d), ThreadIdx: (%d, %d), GlobalIdx: (%d, %d, %d)\n",
        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, globalIdxX, globalIdxY, globalIdxZ);
}

int main() {
    dim3 blockSize(16, 16);
    dim3 gridSize(4, 4);
    kernel2DGrid << <gridSize, blockSize >> > ();
    kernel3DGrid << <gridSize, blockSize >> > ();
    cudaDeviceSynchronize();
    return 0;
}
