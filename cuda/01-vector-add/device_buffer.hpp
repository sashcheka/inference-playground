#pragma once

#include <cstddef>

class DeviceBuffer{
public:
    explicit DeviceBuffer(std::size_t);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* data() const;

private:
    float* ptr_ = nullptr;    
};
