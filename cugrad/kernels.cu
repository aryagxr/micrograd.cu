#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>


/* All kernels go here
operator functions
matmul
backward
*/

// add kernel for two matrices
// a,b -> (m*n)
__global__ void add_kernel1(float* a, float* b, float* out, int m, int n){
    int row = threadIdx.x + (blockDim.x * blockIdx.x);
    int col = threadIdx.y + (blockDim.y * blockIdx.y);
    if(row < m && col < n){
        out[row * n + col] = a[row * n + col] + b[row * n + col];
    }
}


// add kernel for two vectors
__global__ void vect_add_kernel1(float* a, float* b, float* out, int k){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if(tidx < k){
        out[tidx] = a[tidx] + b[tidx];
    }
}


// add kernel for vector + scalar
__global__ void vect_scalar_add_kernel1(float* a, float b, float* out, int k){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if(tidx < k){
        out[tidx] = a[tidx] + b;
    }
}




// scalar mul kernel
// k-> size of input, 1d vector
// b -> scalar 
__global__ void scalarmul_kernel1(float* a, float b, float* out, int k){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if(tidx < k){
        out[tidx] = a[tidx] * b;
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



// tanh kernel -> matrix
__global__ void tanh_kernel1(float* a, float* t, int m, int n){
    int row = threadIdx.x + (blockDim.x * blockIdx.x);
    int col = threadIdx.y + (blockDim.y * blockIdx.y);
    if(row < m && col < n){
        int idx = row * n + col;
        t[idx] = tanhf(a[idx]);
        //t[idx] = (expf(2*a[idx]) - 1) / (expf(2*a[idx]) + 1);
    }

}

// tanh vector kernel
__global__ void tanh_vect_kernel1(float* a, float* t, int k){
    int tidx = threadIdx.x + (blockDim.x * blockIdx.x);
    if(tidx < k){
        t[tidx] = tanhf(a[tidx]);
    }
}






int main() {

    // input size
    const int N = 1;

    // Host inputs
    float h_x1[N] = {2.0f};
    float h_x2[N] = {0.0f};
    float w1 = -3.0f;
    float w2 = 1.0f;
    float b  = 6.7f;

    // Host output
    float h_out[N];

    // Device memory
    float *d_x1, *d_x2, *d_x1w1, *d_x2w2, *d_sum, *d_n, *d_out;

    cudaMalloc(&d_x1, sizeof(float) * N);
    cudaMalloc(&d_x2, sizeof(float) * N);
    cudaMalloc(&d_x1w1, sizeof(float) * N);
    cudaMalloc(&d_x2w2, sizeof(float) * N);
    cudaMalloc(&d_sum, sizeof(float) * N);
    cudaMalloc(&d_n, sizeof(float) * N);
    cudaMalloc(&d_out, sizeof(float) * N);

    cudaMemcpy(d_x1, h_x1, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Multiply inputs by weights
    scalarmul_kernel1<<<1, N>>>(d_x1, w1, d_x1w1, N); // x1 * w1
    scalarmul_kernel1<<<1, N>>>(d_x2, w2, d_x2w2, N); // x2 * w2

    // Add x1w1 + x2w2
    vect_add_kernel1<<<1, N>>>(d_x1w1, d_x2w2, d_sum, N);

    // Add bias
    vect_scalar_add_kernel1<<<1, N>>>(d_sum, b, d_n, N);

    // Apply tanh
    tanh_vect_kernel1<<<1, N>>>(d_n, d_out, N);

    // Copy result to host
    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    std::cout << "Forward pass result (CUDA):\n";
    std::cout << "o: " << h_out[0] << std::endl;

    // Free memory
    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_x1w1);
    cudaFree(d_x2w2);
    cudaFree(d_sum);
    cudaFree(d_n);
    cudaFree(d_out);

    return 0;


    // // Matrix dimensions (m x n)
    // int m = 3; // Number of rows
    // int n = 4; // Number of columns

    // // Host memory allocation
    // float h_a[m * n], h_b[m * n], h_out[m * n];

    // // Initialize input matrices
    // for (int i = 0; i < m * n; i++) {
    //     h_a[i] = i + 1; // Fill with values 1, 2, 3, ...
    //     h_b[i] = (i + 1) * 2; // Fill with values 2, 4, 6, ...
    // }

    // // Device memory allocation
    // float *d_a, *d_b, *d_out;
    // cudaMalloc((void**)&d_a, m * n * sizeof(float));
    // cudaMalloc((void**)&d_b, m * n * sizeof(float));
    // cudaMalloc((void**)&d_out, m * n * sizeof(float));

    // // Copy data from host to device
    // cudaMemcpy(d_a, h_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, h_b, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // // Define block and grid dimensions
    // dim3 blockDim(16, 16); // 16x16 threads per block
    // dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // // Launch the kernel
    // add_kernel1<<<gridDim, blockDim>>>(d_a, d_b, d_out, m, n);

    // // Copy result back to host
    // cudaMemcpy(h_out, d_out, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // // Print the result
    // std::cout << "Matrix A:" << std::endl;
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         std::cout << h_a[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Matrix B:" << std::endl;
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         std::cout << h_b[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Result (A + B):" << std::endl;
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         std::cout << h_out[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // // Free device memory
    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_out);

    // return 0;
}