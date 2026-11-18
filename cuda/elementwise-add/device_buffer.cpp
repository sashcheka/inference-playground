#include "device_buffer.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <string>

DeviceBuffer::DeviceBuffer(std::size_t count, std::string id) : id_(id) {
    cudaError_t allocStatus = cudaMalloc(&ptr_, count * sizeof(float));
    if (allocStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(allocStatus) << "\n";
        throw std::runtime_error(cudaGetErrorString(allocStatus));
    }
    std::cout << id_ << " has constructed" << "\n";
}

DeviceBuffer::~DeviceBuffer() {
    cudaFree(ptr_);
    std::cout << id_ << " has destoyed" << "\n";
}

float* DeviceBuffer::data() const {
    return ptr_;
}
