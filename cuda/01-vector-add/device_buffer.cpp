#include "device_buffer.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <iostream>

DeviceBuffer::DeviceBuffer(std::size_t count) {
    cudaError_t allocStatus = cudaMalloc(&ptr_, count * sizeof(float));
    if (allocStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(allocStatus) << "\n";
        throw std::runtime_error(cudaGetErrorString(allocStatus));
    }
}

DeviceBuffer::~DeviceBuffer() {
    cudaFree(ptr_);
}

float* DeviceBuffer::data() const {
    return ptr_;
}
