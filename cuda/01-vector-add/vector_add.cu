#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <numeric>

__global__ void vectorAdd(
    const float* const a, 
    const float* const b, 
    float* result,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        result[i] = a[i] + b[i];
    }

    return;
}

int main() {
    constexpr std::size_t n = 1'000'000;

    std::vector<float> a(n);
    std::vector<float> b(n);

    iota(a.begin(), a.end(), 0);
    iota(b.begin(), b.end(), 0);

    std::vector<float> result(n);

    float* deviceA = nullptr;
    cudaError_t allocStatusA = cudaMalloc(&deviceA, n * sizeof(float));
    if (allocStatusA != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(allocStatusA) << "\n";
        return 1;
    }
    cudaError_t copyStatusA = cudaMemcpy(deviceA, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (copyStatusA != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(copyStatusA) << "\n";
        cudaFree(deviceA);
        return 1;
    }
    
    float* deviceB = nullptr;
    cudaError_t allocStatusB = cudaMalloc(&deviceB, n * sizeof(float));
    if (allocStatusB != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(allocStatusB) << "\n";
        cudaFree(deviceA);
        return 1;
    }
    cudaError_t copyStatusB = cudaMemcpy(deviceB, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (copyStatusB != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(copyStatusB) << "\n";
        cudaFree(deviceA);
        cudaFree(deviceB);
        return 1;
    }
    
    float* deviceResult = nullptr;
    cudaError_t allocStatusResult = cudaMalloc(&deviceResult, n * sizeof(float));
    if (allocStatusResult != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(allocStatusResult) << "\n";
        cudaFree(deviceA);
        cudaFree(deviceB);
        return 1;
    }

    constexpr int threads = 256;
    constexpr int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(deviceA, deviceB, deviceResult, n);

    cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
        std::cerr << "vectorAdd failed: " << cudaGetErrorString(launchStatus) << "\n";
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceResult);
        return 1;
    }

    cudaError_t copyStatusResult =  cudaMemcpy(result.data(), deviceResult, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (copyStatusResult != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(copyStatusResult) << "\n";
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceResult);
        return 1;
    }

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceResult);

    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] + b[i] != result[i]) {
            std::cerr << "Mistake: " << a[i] + b[i] << " != " << result[i] << "\n";
            return 1;
        }
    }

    std::cout << "Done!" << "\n";
    return 0;
}
