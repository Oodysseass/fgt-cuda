#include <iostream>
#include <cuda_runtime.h>
#include "../headers/mtx.hpp"

__global__ void sparseMatrixMult(CSRMatrix *A, CSCMatrix *B, CSRMatrix *C);

__global__ void hadamardProduct(int *A, int *B);

__host__ void makeUnit(CSCMatrix *A, int rows);

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: ./main <filename>" << std::endl;
        return 1;
    }

    // variable declaration and memory allocation
    CSRMatrix *adjacent, *p1;
    CSRMatrix *devAdjacent, *devp1;
    CSCMatrix *unitVector;
    CSCMatrix *devUnitVector;
    int blocksPerGrid, threadsPerBlock = 256;

    cudaMallocHost(&adjacent, sizeof(CSRMatrix));
    cudaMallocHost(&p1, sizeof(CSRMatrix));
    cudaMallocHost(&unitVector, sizeof(CSCMatrix));
    cudaMalloc(&devAdjacent, sizeof(CSRMatrix));
    cudaMalloc(&devp1, sizeof(CSRMatrix));
    cudaMalloc(&devUnitVector, sizeof(CSCMatrix));

    // get adjacent matrix and copy to GPU
    readMTX(adjacent, argv[1]);
    cudaMemcpy(devAdjacent, adjacent, sizeof(CSRMatrix), cudaMemcpyHostToDevice);

    // ~~~~~~~ calculate p1
    makeUnit(unitVector, adjacent->columns);
    cudaMemcpy(devUnitVector, unitVector, sizeof(CSCMatrix), cudaMemcpyHostToDevice);
    blocksPerGrid = (adjacent->rows + threadsPerBlock - 1) / threadsPerBlock;
    sparseMatrixMult<<<blocksPerGrid, threadsPerBlock>>>(devAdjacent, devUnitVector, devp1);

    std::cout << "#Rows/Columns: " << adjacent->rows << std::endl;
    std::cout << "#Non-zeros: " << adjacent->nz << std::endl;

    return 0;
}

__global__ void sparseMatrixMult(CSRMatrix *A, CSCMatrix *B, CSRMatrix *C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < A->rows && col < B->columns)
    {

    }

    printf("Row: %d, Col: %d\n", row, col);

}

__host__ void makeUnit(CSCMatrix *A, int rows)
{
    A = new CSCMatrix(1, rows, rows);
    A->colIndex[0] = 0;
    A->colIndex[1] = A->nz;

    for (int i = 0; i < A->nz; i++)
    {
        A->nzIndex[i] = i;
        A->nzValues[i] = 1;
    }
}