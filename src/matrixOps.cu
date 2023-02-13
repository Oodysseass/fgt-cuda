#include "matrixOps.hpp"

void sparseMult(CSRMatrix *A, CSRMatrix *B, CSRMatrix *C)
{
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    CSRMatrix *devA, *devB, *devC;

    // allocate A
    CHECK_CUDA( cudaMalloc((void **) &devA->rowIndex,
                            (A->rows + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **) &devA->nzIndex,
                            (A->columns) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **) &devA->nzValues,
                            (A->nz) * sizeof(int)) );
    // allocate B
    CHECK_CUDA( cudaMalloc((void **) &devB->rowIndex,
                            (B->rows + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **) &devB->nzIndex,
                            (B->columns) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **) &devB->nzValues,
                            (B->nz) * sizeof(int)) );
    // allocate only rowIndexes of C
    CHECK_CUDA( cudaMalloc((void **) &devB->rowIndex,
                            (A->rows) * sizeof(int)) );

    // copy A
    CHECK_CUDA( cudaMemcpy(devA->rowIndex, A->rowIndex,
                            (A->rows + 1) * sizeof(int),
                            cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(devA->nzIndex, A->nzIndex,
                            (A->columns) * sizeof(int),
                            cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(devA->nzValues, A->nzValues,
                            (A->nz) * sizeof(int),
                            cudaMemcpyHostToDevice) );
    // copy B
    CHECK_CUDA( cudaMemcpy(devB->rowIndex, B->rowIndex,
                            (B->rows + 1) * sizeof(int),
                            cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(devB->nzIndex, B->nzIndex,
                            (B->columns) * sizeof(int),
                            cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(devB->nzValues, B->nzValues,
                            (B->nz) * sizeof(int),
                            cudaMemcpyHostToDevice) );

}