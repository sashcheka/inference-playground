cmake_minimum_required(VERSION 3.24)

project(inference_playground LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

enable_testing()

add_executable(elementwise_add
    cuda/elementwise-add/elementwise_add.cu
    cuda/elementwise-add/device_buffer.cpp
)
target_link_libraries(elementwise_add PRIVATE CUDA::cudart)
set_target_properties(elementwise_add PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
)

add_test(NAME elementwise_add COMMAND elementwise_add)
