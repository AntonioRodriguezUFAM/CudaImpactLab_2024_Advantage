
/*
Cuda_01E_ImageConvolutionUsingSTBLibrary
*/

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define KERNEL_RADIUS 1
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)

// CUDA kernel for 2D image convolution
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
    int width, height, channels;

    // Load image using stb_image (grayscale)
    unsigned char* img = stbi_load("input_image.jpg", &width, &height, &channels, 1);
    if (!img) {
        printf("Error: Could not load image.\n");
        return -1;
    }

    // Convert image to float for CUDA processing
    /*
    Precision and Normalization:
    Image data loaded from stb_image or any other library typically comes in as unsigned char (8-bit integers)
    with values ranging from 0 to 255.
    Converting these pixel values to float and normalizing them to the range [0, 1] 
    improves the precision for computations. 
    This is crucial for complex image processing tasks where operations can yield fractional results 
    that unsigned char data types cannot represent.

    Why Normalize to [0, 1]
    Normalization helps to maintain a uniform range for input data, which can make processing more predictable
    and reduce the risk of numerical instability.
    When processing is complete, the data is often converted back to 0–255 (unsigned char)
    for saving or display purposes.
    */

    float* h_input = (float*)malloc(width * height * sizeof(float));
    float* h_output = (float*)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) {
        h_input[i] = img[i] / 255.0f; // Normalize to [0, 1]
    }

    // Allocate device memory
    float* d_input, * d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 2D grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    imageConvolution << <gridSize, blockSize >> > (d_input, d_output, width, height);

    // Copy output data back to host
    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert output back to unsigned char for saving

    /*    Display and File Format Requirements:
    Most common image formats (e.g., JPEG, PNG) expect pixel values to be in the unsigned char range (0–255). 
    If you save an image using floating-point data, it may not be interpreted correctly by image viewers or file formats.
    Converting the float values (which are typically in the range [0, 1] after processing) 
    back to unsigned char allows you to save the image in a format that is compatible 
    with standard image viewing and storage tools
    */
    unsigned char* result_img = (unsigned char*)malloc(width * height);
    for (int i = 0; i < width * height; i++) {
        result_img[i] = (unsigned char)(h_output[i] * 255.0f); // Convert back to [0, 255]
    }

    // Save the result using stb_image_write
    stbi_write_jpg("output_image.jpg", width, height, 1, result_img, 100);

    // Cleanup
    stbi_image_free(img);
    free(h_input);
    free(h_output);
    free(result_img);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
