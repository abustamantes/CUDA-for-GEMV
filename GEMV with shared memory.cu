#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>

using namespace std;

#define TILE_SIZE 16
#define BLOCK_SIZE 256

double atomicAdd(double C, double tmp);


__global__ void matvec(double* A, double* B, double* C, int n)
{
    __shared__ double s_A[TILE_SIZE][TILE_SIZE];
    __shared__ double s_B[TILE_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int i = bx * blockDim.x + tx;

    if (i < n) {
        s_B[tx] = B[i];
        for (int j = 0; j < n; j += TILE_SIZE) {
            s_A[tx][j + threadIdx.y] = A[(i * n) + j + threadIdx.y];
        }
    }
    __syncthreads();

    if (i < n) {
        double tmp = 0.0;
        for (int j = 0; j < n; j += TILE_SIZE) {
            tmp += s_A[threadIdx.x][j + threadIdx.y] * s_B[j + threadIdx.y];
        }
        atomicAdd(&C[i], tmp);
    }
}

int main()
{
    int n = 5000;
    double* A, * B, * C;
    double* d_A, * d_B, * d_C;

    A = (double*)malloc(n * n * sizeof(double));
    B = (double*)malloc(n * sizeof(double));
    C = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n * n; i++) {
        A[i] = rand() / (double)RAND_MAX;
    }
    for (int i = 0; i < n; i++) {
        B[i] = rand() / (double)RAND_MAX;
        C[i] = 0.0;
    }

    cudaMalloc((void**)&d_A, n * n * sizeof(double));
    cudaMalloc((void**)&d_B, n * sizeof(double));
    cudaMalloc((void**)&d_C, n * sizeof(double));

    cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, n * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, BLOCK_SIZE / TILE_SIZE, 1);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

    matvec << <dimGrid, dimBlock >> > (d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
