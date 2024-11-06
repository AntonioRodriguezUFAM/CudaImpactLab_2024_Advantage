/*
 Image Load, Process, and Save Using stb

*/

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// CUDA kernel for matrix multiplication (image processing)
__global__ void matrixMul(const float* A, const float* B, float* C, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float Cvalue = 0;

    if (row < height && col < width) {
        for (int k = 0; k < width; ++k) {
            Cvalue += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = Cvalue;
    }
}

int main() {
    int width, height, channels;

    // Load the images using stb_image (grayscale)
    unsigned char* imgA = stbi_load("images/apple.jpg", &width, &height, &channels, 1);
    unsigned char* imgB = stbi_load("images/ship_4k_rgba.png", &width, &height, &channels, 1);

    if (!imgA || !imgB) {
        printf("Error: Could not load images.\n");
        if (imgA) stbi_image_free(imgA);
        if (imgB) stbi_image_free(imgB);
        return -1;
    }

    // Convert the images to float arrays for CUDA processing
    float* h_A = (float*)malloc(width * height * sizeof(float));
    float* h_B = (float*)malloc(width * height * sizeof(float));
    float* h_C = (float*)malloc(width * height * sizeof(float));

    for (int i = 0; i < width * height; i++) {
        h_A[i] = imgA[i] / 255.0f;
        h_B[i] = imgB[i] / 255.0f;
    }

    // Allocate device memory
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, width * height * sizeof(float));
    cudaMalloc((void**)&d_B, width * height * sizeof(float));
    cudaMalloc((void**)&d_C, width * height * sizeof(float));

    // Copy images to device
    cudaMemcpy(d_A, h_A, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 2D grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    matrixMul << <dimGrid, dimBlock >> > (d_A, d_B, d_C, width, height);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert the result back to unsigned char for saving
    unsigned char* result_img = (unsigned char*)malloc(width * height);
    for (int i = 0; i < width * height; i++) {
        result_img[i] = (unsigned char)(h_C[i] * 255.0f);
    }

    // Save the result using stb_image_write
    stbi_write_jpg("images/result_image.jpg", width, height, 1, result_img, 100);

    // Cleanup
    stbi_image_free(imgA);
    stbi_image_free(imgB);
    free(h_A);
    free(h_B);
    free(h_C);
    free(result_img);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
