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

__global__ void matrixAdd(
    const float* const a,
    const float* const b,
    float* const result,
    std::size_t height,
    std::size_t width
) {
    std::size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width) {
        std::size_t i = r * width + c;
        result[i] = a[i] + b[i];
    }
}

int main() {
    constexpr std::size_t height = 8193;
    constexpr std::size_t width = 17;
    constexpr std::size_t n = height * width;

    // constexpr dim3 threads(32, 8);
    constexpr dim3 threads(16, 16);
    // constexpr dim3 threads(7, 5);
    constexpr dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    std::cout << "Threads per block (x, y): " << threads.x << " x " << threads.y << "\n";
    std::cout << "Grid size (x, y): " << blocks.x << " x " << blocks.y << "\n";

    std::vector<float> a(n);
    std::vector<float> b(n);
    std::vector<float> result(n);

    iota(a.begin(), a.end(), 0);
    iota(b.begin(), b.end(), 0);

    try {
        DeviceBuffer deviceA(n, "DeviceA");
        DeviceBuffer deviceB(n, "DeviceB");
        DeviceBuffer deviceResult(n, "DeviceResult");

        constexpr int repeats = 1;
        std::chrono::duration<double, std::milli> totalDuration{};
        for (int epoch = 0; epoch < repeats; epoch++) {
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

            matrixAdd<<<blocks, threads>>>(deviceA.data(), deviceB.data(), deviceResult.data(), height, width);
            cudaError_t warmupStatus = cudaGetLastError();
            if (warmupStatus != cudaSuccess) {
                std::cerr << "matrixAdd failed: " << cudaGetErrorString(warmupStatus) << "\n";
                return 1;
            }

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            constexpr int runs = 100;
            for (int run = 0; run < runs; run++) {
                matrixAdd<<<blocks, threads>>>(deviceA.data(), deviceB.data(), deviceResult.data(), height, width);
                cudaError_t launchStatus = cudaGetLastError();
                if (launchStatus != cudaSuccess) {
                    std::cerr << "matrixAdd failed: " << cudaGetErrorString(launchStatus) << "\n";
                    return 1;
                }
            }

            cudaEventRecord(stop);
            cudaError_t syncStatus  = cudaEventSynchronize(stop);
            if (syncStatus  != cudaSuccess) {
                std::cerr << "cudaEventSynchronize failed: " << cudaGetErrorString(syncStatus ) << "\n";
                return 1;
            }

            float elapsedMs = 0.0f;
            cudaError_t elapsedStatus  = cudaEventElapsedTime(&elapsedMs, start, stop);
            if (elapsedStatus  != cudaSuccess) {
                std::cerr << "cudaEventElapsedTime failed: " << cudaGetErrorString(elapsedStatus ) << "\n";
                return 1;
            }
            std::cout << "Kernel total: " << elapsedMs << "ms" << "\n";
            std::cout << "Kernel average: " << elapsedMs / runs << "ms" << "\n";

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            // constexpr int threads = 256;
            // constexpr int blocks = (n + threads - 1) / threads;

            // vectorAdd<<<blocks, threads>>>(deviceA.data(), deviceB.data(), deviceResult.data(), n);
            // cudaError_t warmupStatus = cudaGetLastError();
            // if (warmupStatus != cudaSuccess) {
            //     std::cerr << "vectorAdd failed: " << cudaGetErrorString(warmupStatus) << "\n";
            //     return 1;
            // }

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
            totalDuration += cpuStop - cpuStart;
            std::chrono::duration<double, std::milli> elapsed = cpuStop - cpuStart;
            std::cout << "CPU + GPU elapsed time: " << elapsed.count() << "\n";
        }

        std::cout << "Total total: " << totalDuration.count() << "ms" << "\n";
        std::cout << "Total average: " << (totalDuration/ repeats).count() << "ms" << "\n";

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
