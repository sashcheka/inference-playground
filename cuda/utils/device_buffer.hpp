#pragma once

#include <cstddef>
#include <string>

class DeviceBuffer{
public:
    explicit DeviceBuffer(std::size_t, std::string);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* data() const;

private:
    float* ptr_ = nullptr;
    std::string id_;
};
