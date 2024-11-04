/*
3. Asynchronous Operations
Example: Asynchronous memory transfers with CUDA streams.
*/
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int n = 1000000;
    float *h_data = (float*)malloc(n * sizeof(float));
    float *d_data;
    cudaStream_t stream;

    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_data[i] = static_cast<float>(i);
    }

    cudaMalloc((void**)&d_data, n * sizeof(float));
    cudaStreamCreate(&stream);

    // Asynchronous data transfer
    cudaMemcpyAsync(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Perform operations (can be kernel calls in the same stream)
    // ...

    // Synchronize stream to ensure all operations are complete
    cudaStreamSynchronize(stream);

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
