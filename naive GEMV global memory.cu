#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

#define MATRIX_SIZE 3
#define VECTOR_SIZE 3

__global__ void matrixVectorMultiplication(double* matrix, double* vector, double* result)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < MATRIX_SIZE)
    {
        double dot_product = 0;
        for (int i = 0; i < MATRIX_SIZE; i++)
        {
            dot_product += matrix[row * MATRIX_SIZE + i] * vector[i];
        }
        result[row] = dot_product;
    }
}

int main()
{
    double* matrix, * vector, * result;
    double* dev_matrix, * dev_vector, * dev_result;
    size_t matrix_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(double);
    size_t vector_size = VECTOR_SIZE * sizeof(double);

    // allocate memory on host
    matrix = (double*)malloc(matrix_size);
    vector = (double*)malloc(vector_size);
    result = (double*)malloc(vector_size);

    // initialize matrix and vector
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            matrix[i * MATRIX_SIZE + j] = ((double)rand() / (double)RAND_MAX);
        }
        vector[i] = ((double)rand() / (double)RAND_MAX);
        result[i] = 0.0;
    }

    printf("The matrix is:\n");
    printf("***************");
    for (int i =0;i < MATRIX_SIZE*MATRIX_SIZE;i++) {
        if (i % VECTOR_SIZE == 0) {
            
            printf("\n");
        }
        cout << matrix[i] << " "; 
    }
    printf("\n\n");

    printf("The vector is:\n");
    printf("***************\n");
    for (int i = 0;i < VECTOR_SIZE;i++) {
        cout << vector[i] << " ";
    }
    printf("\n\n");

    // allocate memory on device
    cudaMalloc((void**)&dev_matrix, matrix_size);
    cudaMalloc((void**)&dev_vector, vector_size);
    cudaMalloc((void**)&dev_result, vector_size);

    // copy data from host to device
    cudaMemcpy(dev_matrix, matrix, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vector, vector, vector_size, cudaMemcpyHostToDevice);

    // define block and grid size
    int block_size = 256;
    int grid_size = (MATRIX_SIZE + block_size - 1) / block_size;

    // launch kernel
    matrixVectorMultiplication << <grid_size, block_size >> > (dev_matrix, dev_vector, dev_result);

    // copy data from device to host
    cudaMemcpy(result, dev_result, vector_size, cudaMemcpyDeviceToHost);

    printf("The result is:\n");
    printf("***************\n");
    for (int i = 0;i < VECTOR_SIZE;i++) {
        cout << result[i] <<" ";
    }
    printf("\n");

    // free memory on device and host
    cudaFree(dev_matrix);
    cudaFree(dev_vector);
    cudaFree(dev_result);
    free(matrix);
    free(vector);
    free(result);

    return 0;
}
