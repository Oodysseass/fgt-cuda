#include "../headers/matrixOps.hpp"

void sparseMult(CSRMatrix *A, CSRMatrix *B, CSRMatrix *C)
{
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    CSRMatrix *devA, *devB, *devC;

    // allocate A
    CHECK_CUDA(cudaMalloc((void **)&devA->rowIndex,
                          (A->rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devA->nzIndex,
                          (A->nz) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devA->nzValues,
                          (A->nz) * sizeof(int)))
    // allocate B
    CHECK_CUDA(cudaMalloc((void **)&devB->rowIndex,
                          (B->rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devB->nzIndex,
                          (B->nz) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devB->nzValues,
                          (B->nz) * sizeof(int)))
    // allocate only rowIndexes of C
    CHECK_CUDA(cudaMalloc((void **)&devC->rowIndex,
                          (A->rows + 1) * sizeof(int)))

    // copy A
    CHECK_CUDA(cudaMemcpy(devA->rowIndex, A->rowIndex,
                          (A->rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(devA->nzIndex, A->nzIndex,
                          (A->nz) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(devA->nzValues, A->nzValues,
                          (A->nz) * sizeof(int),
                          cudaMemcpyHostToDevice))
    // copy B
    CHECK_CUDA(cudaMemcpy(devB->rowIndex, B->rowIndex,
                          (B->rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(devB->nzIndex, B->nzIndex,
                          (B->nz) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(devB->nzValues, B->nzValues,
                          (B->nz) * sizeof(int),
                          cudaMemcpyHostToDevice))

    // preparing for cusparse api
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))

    // create matrixes in cusparse csr format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A->rows, A->columns, A->nz,
                                     devA->rowIndex, devA->nzIndex,
                                     devA->nzValues, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32I))
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, B->rows, B->columns, B->nz,
                                     devB->rowIndex, devB->nzIndex,
                                     devB->nzValues, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32I))
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, A->rows, B->columns, 0,
                                     NULL, NULL, NULL, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32I))

    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))

    float alpha = 1.0f;
    float beta = 0.0f;
    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA,
                                                 matB, &beta, matC, CUDA_R_32I,
                                                 CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDesc, &bufferSize1, NULL))
    // check memory requirment of next step
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA,
                                                 matB, &beta, matC, CUDA_R_32I,
                                                 CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDesc, &bufferSize1,
                                                 dBuffer1))
    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA,
                                                 matB, &beta, matC, CUDA_R_32I,
                                                 CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDesc, &bufferSize2,
                                                 dBuffer2))

    // A * B product
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB,
                                          &beta, matC, CUDA_R_32I,
                                          CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                          &bufferSize2, dBuffer2))
    // get matrix C non-zero entries C_nnz1
    int64_t tempRows, tempCols, tempnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &tempRows, &tempCols,
                                        &tempnz))
    // allocate matrix C
    CHECK_CUDA(cudaMalloc((void **)&devC->nzIndex, tempCols * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devC->nzValues, tempnz * sizeof(float)))

    // update matC
    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, devC->rowIndex, devC->nzIndex,
                                          devC->nzValues))
    // copy results
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB,
                                       &beta, matC, CUDA_R_32I, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc))

    // destroy descriptors
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroySpMat(matB))
    CHECK_CUSPARSE(cusparseDestroySpMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    // copy to host
    C = new CSRMatrix(tempRows, tempCols, tempnz);
    CHECK_CUDA(cudaMemcpy(C->rowIndex, devC->rowIndex, C->rows * sizeof(int),
                          cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(C->nzIndex, devC->nzIndex, C->nz * sizeof(int),
                          cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(C->nzValues, devC->nzValues, C->nz * sizeof(int),
                          cudaMemcpyDeviceToHost))

    CHECK_CUDA(cudaFree(dBuffer1))
    CHECK_CUDA(cudaFree(dBuffer2))
    CHECK_CUDA(cudaFree(devA->rowIndex))
    CHECK_CUDA(cudaFree(devA->nzIndex))
    CHECK_CUDA(cudaFree(devA->nzValues))
    CHECK_CUDA(cudaFree(devB->rowIndex))
    CHECK_CUDA(cudaFree(devB->rowIndex))
    CHECK_CUDA(cudaFree(devB->nzIndex))
    CHECK_CUDA(cudaFree(devC->nzValues))
    CHECK_CUDA(cudaFree(devC->nzIndex))
    CHECK_CUDA(cudaFree(devC->nzValues))
}

__global__ void calcdZero(int *e, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
        e[i] = 1;
}

__global__ void calcdOne(CSRMatrix *A, int *p1)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < A->rows)
        p1[i] = A->rowIndex[i + 1] - A->rowIndex[i];
}

__global__ void calcdTwo(CSRMatrix *A, int *p1, int *p2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < A->rows)
    {
        for (int j = A->rowIndex[i]; j < A->rowIndex[i + 1]; j++)
            p2[i] += p1[A->nzIndex[j]];

        p2[i] -= p1[i];
    }
}