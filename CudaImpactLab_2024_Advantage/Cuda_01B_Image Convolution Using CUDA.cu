/*
. Image Convolution Using CUDA
Example: Simplified convolution kernel for 2D image processing.

*/

#include <cuda_runtime.h>
#include <stdio.h>

#define KERNEL_RADIUS 1
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)

__global__ void imageConvolution(const float* input, float* output, int width, int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height) {
        float sum = 0;
        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
                int currentCol = min(max(col + i, 0), width - 1);
                int currentRow = min(max(row + j, 0), height - 1);
                sum += input[currentRow * width + currentCol];
            }
        }
        output[row * width + col] = sum;
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    size_t size = width * height * sizeof(float);
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    float* d_input, * d_output;

    // Initialize input data
    for (int i = 0; i < width * height; i++) {
        h_input[i] = 1.0f; // Example data
    }

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch kernel with 2D grid and block
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    imageConvolution << <gridSize, blockSize >> > (d_input, d_output, width, height);

    // Copy output data back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
