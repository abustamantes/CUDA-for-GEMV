#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>


using namespace std;

__global__ void matVecMult(double* A, double* x, double* y, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        double sum = 0;
        for (int j = 0; j < N; j += 8) {
            sum += A[i * N + j] * x[j];
            sum += A[i * N + j + 1] * x[j + 1];
            sum += A[i * N + j + 2] * x[j + 2];
            sum += A[i * N + j + 3] * x[j + 3];
            sum += A[i * N + j + 4] * x[j + 4];
            sum += A[i * N + j + 5] * x[j + 5];
            sum += A[i * N + j + 6] * x[j + 6];
            sum += A[i * N + j + 7] * x[j + 7];
        }
        y[i] = sum;
    }
}

int main() {
    int N = 3;
    int size = N * N * sizeof(double);
    int vec_size = N * sizeof(double);

    double* h_A = new double[N * N];
    double* h_x = new double[N];
    double* h_y = new double[N];

    // initialize A, x, and y on the host
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (double)RAND_MAX;
    }
    for (int i = 0; i < N; i++) {
        h_x[i] = rand() / (double)RAND_MAX;
    }

    printf("The matrix is:\n");
    printf("***************");
    for (int i = 0;i < N*N ;i++) {
        if (i % N == 0) {

            printf("\n");
        }
        cout << h_A[i] << " ";
    }
    printf("\n\n");

    printf("The vector is:\n");
    printf("***************\n");
    for (int i = 0;i < N;i++) {
        cout << h_x[i] << " ";
    }
    printf("\n\n");

    double* d_A, * d_x, * d_y;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_x, vec_size);
    cudaMalloc(&d_y, vec_size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, vec_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    matVecMult << <blocksPerGrid, threadsPerBlock >> > (d_A, d_x, d_y, N);

    cudaMemcpy(h_y, d_y, vec_size, cudaMemcpyDeviceToHost);

    printf("The result is:\n");
    printf("***************\n");
    for (int i = 0;i < N;i++) {
        cout << h_y[i] << " ";
    }
    printf("\n");

    /*
    // check results
    double* y_ref = new double[N];
    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            sum += h_A[i * N + j] * h_x[j];
        }
        y_ref[i] = sum;
    }
    for (int i = 0; i < N; i++) {
        if (abs(h_y[i] - y_ref[i]) > 1e-6) {
            cerr << "Error: mismatch at index " << i << endl;
            break;
        }
    }
    */

    delete[] h_A;
    delete[] h_x;
    delete[] h_y;
    //delete[] y_ref;
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}


