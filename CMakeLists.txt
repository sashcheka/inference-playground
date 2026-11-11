cmake_minimum_required(VERSION 3.24)

project(inference_playground LANGUAGES CXX CUDA)

enable_testing()

add_executable(vector_add cuda/01-vector-add/vector_add.cu)
set_target_properties(vector_add PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
)

add_test(NAME vector_add COMMAND vector_add)
