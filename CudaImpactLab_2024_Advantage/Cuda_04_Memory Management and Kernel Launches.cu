/*
4. Memory Management and Kernel Launches
Example: Memory allocation and copying data between host and device.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int N = 1024;
    int size = N * sizeof(float);
    float *h_data = (float*)malloc(size);
    float *d_data;

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(i);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_data, size);

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Perform operations on the device (dummy kernel can be added here)

    // Copy data back to host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    free(h_data);

    return 0;
}
