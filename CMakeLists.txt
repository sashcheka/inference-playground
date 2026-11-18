cmake_minimum_required(VERSION 3.24)

project(inference_playground LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

enable_testing()

add_library(cuda_utils STATIC cuda/utils/device_buffer.cpp)
target_include_directories(cuda_utils PUBLIC cuda/utils)
target_link_libraries(cuda_utils PRIVATE CUDA::cudart)

add_executable(elementwise_add cuda/elementwise-add/elementwise_add.cu)
target_link_libraries(elementwise_add PRIVATE cuda_utils)
set_target_properties(elementwise_add PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
)

add_test(NAME elementwise_add COMMAND elementwise_add)

add_executable(memory_coalescing cuda/memory-coalescing/memory_coalescing.cu)
target_link_libraries(memory_coalescing PRIVATE cuda_utils)
set_target_properties(memory_coalescing PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
)
