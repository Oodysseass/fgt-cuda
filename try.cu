#include <iostream>
#include <cuda_runtime.h>


__global__ void vectorAdd(const float* a, const float* b, float* c, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        c[i] = a[i] + b[i];
}

int main(int argc, char* argv[])
{
    const int N = 512;
    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    cudaMallocHost(&a, N * sizeof(float));
    cudaMallocHost(&b, N * sizeof(float));
    cudaMallocHost(&c, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        a[i] = (float)i;
        b[i] = (float)i * 2;
    }

    cudaMalloc(&dev_a, N * sizeof(float));
    cudaMalloc(&dev_b, N * sizeof(float));
    cudaMalloc(&dev_c, N * sizeof(float));

    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);

    cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int i = 0; i < N; i++)
    {
        if (c[i] != a[i] + b[i])
        {
            passed = false;
            break;
        }
    }
    std::cout << "Vector addition: " << (passed ? "PASSED" : "FAILED") << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    return 0;
}
