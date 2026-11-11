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
    cudaError_t statusA = cudaMalloc(&deviceA, n * sizeof(float));
    if (statusA != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(statusA) << "\n";
        return 1;
    }
    cudaMemcpy(deviceA, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    float* deviceB = nullptr;
    cudaError_t statusB = cudaMalloc(&deviceB, n * sizeof(float));
    if (statusB != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(statusB) << "\n";
        return 1;
    }
    cudaMemcpy(deviceB, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    float* deviceResult = nullptr;
    cudaError_t statusResult = cudaMalloc(&deviceResult, n * sizeof(float));
    if (statusResult != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(statusResult) << "\n";
        return 1;
    }

    constexpr int threads = 256;
    constexpr int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(deviceA, deviceB, deviceResult, n);
    cudaMemcpy(result.data(), deviceResult, n * sizeof(float), cudaMemcpyDeviceToHost);

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
