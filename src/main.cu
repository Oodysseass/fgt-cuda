#include <iostream>
#include <cuda_runtime.h>
#include "../headers/mtx.hpp"

template<typename MatrixType>
__host__ void copyToDev(MatrixType *A, MatrixType *devA);

__global__ void sparseMatrixMult(CSRMatrix *A, CSCMatrix *B, CSRMatrix *C);

__host__ void makeUnit(CSCMatrix *A);

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: ./main <filename>" << std::endl;
        return 1;
    }

    // ~~~~~~~ variable declaration and memory allocation
    CSRMatrix *adjacent, *p1;
    CSRMatrix *devAdjacent, *devp1;
    CSCMatrix *unitVector;
    CSCMatrix *devUnitVector;
    int blocksPerGrid, threadsPerBlock = 256;

    // get adjacent matrix and copy to GPU
    CSCMatrix tempAdjacent = readMTX(argv[1]);
    adjacent = new CSRMatrix(tempAdjacent.rows, tempAdjacent.columns,
                                tempAdjacent.nz);
    convert(tempAdjacent, adjacent);
    copyToDev(adjacent, devAdjacent);

    // ~~~~~~~ calculate p1
    // calculate e
    unitVector = new CSCMatrix(1, adjacent->columns, adjacent->columns);
    makeUnit(unitVector);
    // allocate device memory
    copyToDev(unitVector, devUnitVector);
    blocksPerGrid = (adjacent->rows + threadsPerBlock - 1) / threadsPerBlock;
    // perform sparse matrix - vector multiplication
    sparseMatrixMult<<<blocksPerGrid, threadsPerBlock>>>(devAdjacent,
                                                            devUnitVector, devp1);

    std::cout << "#Rows/Columns: " << adjacent->rows << std::endl;
    std::cout << "#Non-zeros: " << adjacent->nz << std::endl;

    return 0;
}

template<typename MatrixType>
__host__ void copyToDev(MatrixType *A, MatrixType *devA)
{
    size_t matrixSize = sizeof(MatrixType);
    size_t rowSize = sizeof(int) * A->rows;
    size_t colSize = sizeof(int) * A->columns;
    size_t nzSize = sizeof(int) * A->nz;

    cudaMalloc(&devA, matrixSize + rowSize + colSize + nzSize);
    cudaMalloc(&devA->rowIndex, rowSize);
    cudaMalloc(&devA->nzIndex, colSize);
    cudaMalloc(&devA->nzValues, nzSize);

    cudaMemcpy(&devA->rows, &A->rows, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&devA->columns, &A->columns, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&devA->nz, &A->nz, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devA->rowIndex, A->rowIndex, rowSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devA->nzIndex, A->nzIndex, colSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devA->nzValues, A->nzValues, nzSize, cudaMemcpyHostToDevice);
}

__global__ void sparseMatrixMult(CSRMatrix *A, CSCMatrix *B, CSRMatrix *C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < A->rows && col < B->columns)
    {
        int sum = 0;
        for (int i = A->rowIndex[row]; i < A->rowIndex[row + 1]; i++)
        {
            int k = A->nzIndex[i];
            for (int j = B->colIndex[col]; j < B->colIndex[col + 1]; j++)
                if (B->nzIndex[j] == k)
                    sum += A->nzValues[i] * B->nzValues[j];
        }
        if (sum != 0)
        {
            atomicAdd(&C->rowIndex[row + 1], 1);
            int idx = atomicAdd(&C->rowIndex[A->rows], 1);
            C->nzIndex[idx] = col;
            C->nzValues[idx] = sum;
        }
    }

    printf("mpainw\n");
}

__host__ void makeUnit(CSCMatrix *A)
{
    A->colIndex[0] = 0;
    A->colIndex[1] = A->nz;

    for (int i = 0; i < A->nz; i++)
    {
        A->nzIndex[i] = i;
        A->nzValues[i] = 1;
    }
}