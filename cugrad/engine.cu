#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>


// add kernel
// a,b -> (m*n)
__global__ void add_kernel1(float* a, float* b, float* out, int m, int n){
    int row = threadIdx.x + (blockDim.x * blockIdx.x);
    int col = threadIdx.y + (blockDim.y * blockIdx.y);
    if(row < m && col < n){
        out[row * n + col] = a[row * n + col] + b[row * n + col];
    }
}



// mul kernel
__global__ void matmul_kernel1(float* a, float* b, float* out, int m, int n, int k){
    int row = threadIdx.x + (blockDim.x * blockIdx.x);
    int col = threadIdx.y + (blockDim.y * blockIdx.y);
    if(row < m && col < n){
        float acc = 0.0f;
        for(int i = 0; i < n; i++){
            acc += a[row * n + i] * b[i * k + col];
        }
        out[row * k + col] = acc;
    }
}



// pow kernel --> may not need this
__global__ void pow_kernel1(float* a, float* out, float e, int m, int n){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if(tidx < m){
        out[tidx] = powf(a[tidx], e);
    }
}



// tanh kernel
__global__ void tanh_kernel1(float* a, float* t, int m, int n){
    int row = threadIdx.x + (blockDim.x * blockIdx.x);
    int col = threadIdx.y + (blockDim.y * blockIdx.y);
    if(row < m && col < n){
        int idx = row * n + col;
        t[idx] = (expf(2*a[idx]) - 1) / (expf(2*a[idx]) + 1);
    }

}







int main() {
    // Matrix dimensions (m x n)
    int m = 3; // Number of rows
    int n = 4; // Number of columns

    // Host memory allocation
    float h_a[m * n], h_b[m * n], h_out[m * n];

    // Initialize input matrices
    for (int i = 0; i < m * n; i++) {
        h_a[i] = i + 1; // Fill with values 1, 2, 3, ...
        h_b[i] = (i + 1) * 2; // Fill with values 2, 4, 6, ...
    }

    // Device memory allocation
    float *d_a, *d_b, *d_out;
    cudaMalloc((void**)&d_a, m * n * sizeof(float));
    cudaMalloc((void**)&d_b, m * n * sizeof(float));
    cudaMalloc((void**)&d_out, m * n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    add_kernel1<<<gridDim, blockDim>>>(d_a, d_b, d_out, m, n);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << h_a[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << h_b[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Result (A + B):" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << h_out[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}