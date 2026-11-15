#include "device_buffer.hpp"

#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>

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
    std::vector<float> result(n);

    iota(a.begin(), a.end(), 0);
    iota(b.begin(), b.end(), 0);

    try {
        DeviceBuffer deviceA(n, "DeviceA");
        DeviceBuffer deviceB(n, "DeviceB");
        DeviceBuffer deviceResult(n, "DeviceResult");

        for (int epoch = 0; epoch < 10; epoch++) {
            auto cpuStart = std::chrono::steady_clock::now();

            cudaError_t copyStatusA = cudaMemcpy(deviceA.data(), a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            if (copyStatusA != cudaSuccess) {
                std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(copyStatusA) << "\n";
                return 1;
            }

            cudaError_t copyStatusB = cudaMemcpy(deviceB.data(), b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            if (copyStatusB != cudaSuccess) {
                std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(copyStatusB) << "\n";
                return 1;
            }

            constexpr int threads = 256;
            constexpr int blocks = (n + threads - 1) / threads;

            vectorAdd<<<blocks, threads>>>(deviceA.data(), deviceB.data(), deviceResult.data(), n);

            cudaError_t launchStatus = cudaGetLastError();
            if (launchStatus != cudaSuccess) {
                std::cerr << "vectorAdd failed: " << cudaGetErrorString(launchStatus) << "\n";
                return 1;
            }

            // cudaEvent_t start, stop;
            // cudaEventCreate(&start);
            // cudaEventCreate(&stop);

            // cudaEventRecord(start);
            // for (int epoch = 0; epoch < 100; epoch++) {
            //     vectorAdd<<<blocks, threads>>>(deviceA, deviceB, deviceResult, n);
            //     launchStatus = cudaGetLastError();
            //     if (launchStatus != cudaSuccess) {
            //         std::cerr << "vectorAdd failed: " << cudaGetErrorString(launchStatus) << "\n";
            //         return 1;
            //     }
            // }
            // cudaEventRecord(stop);

            // cudaEventSynchronize(stop);
            // float elapsedMs = 0.0f;
            // cudaEventElapsedTime(&elapsedMs, start, stop);
            // std::cout << "vectorAdd kernel total elapsedTime: " << elapsedMs << "ms" << "\n";
            // std::cout << "vectorAdd kernel average elapsedTime: " << elapsedMs / 100 << "ms" << "\n";

            // cudaEventDestroy(start);
            // cudaEventDestroy(stop);

            cudaError_t copyStatusResult = cudaMemcpy(result.data(), deviceResult.data(), n * sizeof(float), cudaMemcpyDeviceToHost);
            if (copyStatusResult != cudaSuccess) {
                std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(copyStatusResult) << "\n";
                return 1;
            }

            auto cpuStop = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = cpuStop - cpuStart;
            std::cout << "CPU + GPU elapsed time: " << elapsed.count() << "\n";
        }

        for (std::size_t i = 0; i < n; ++i) {
            if (a[i] + b[i] != result[i]) {
                std::cerr << "Mistake: " << a[i] + b[i] << " != " << result[i] << "\n";
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Done!" << "\n";
    return 0;
}
