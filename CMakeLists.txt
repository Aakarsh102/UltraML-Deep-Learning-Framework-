cmake_minimum_required(VERSION 3.18)
project(ultraml LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Pick a reasonable default arch; override with -DCMAKE_CUDA_ARCHITECTURES=xx
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
endif()

find_package(CUDAToolkit REQUIRED)

set(ULTRAML_SOURCES
    core/tensor.cu
    core/context.cu
    activations/activations.cu
    norms/batchnorm.cu
    norms/layernorm.cu
    norms/rmsnorm.cu
    norms/groupnorm.cu
    layers/linear.cu
    layers/conv.cu
    layers/pool.cu
    layers/matmul.cu
    layers/dropout.cu
    layers/embedding.cu
    losses/losses.cu
    autograd/autograd.cu
    autograd/ops.cu
    nn/nn.cu
    nn/init.cu
    optim/optim.cu
)

add_library(ultraml STATIC ${ULTRAML_SOURCES})

target_include_directories(ultraml PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

set_target_properties(ultraml PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)

target_link_libraries(ultraml
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cudnn
)

# ---- examples --------------------------------------------------------
option(ULTRAML_BUILD_EXAMPLES "Build examples" ON)
if(ULTRAML_BUILD_EXAMPLES)
    add_executable(mlp_example examples/mlp.cpp)
    target_link_libraries(mlp_example PRIVATE ultraml)

    add_executable(train_mlp_example examples/train_mlp.cpp)
    target_link_libraries(train_mlp_example PRIVATE ultraml)

    add_executable(transformer_block_example examples/transformer_block.cpp)
    target_link_libraries(transformer_block_example PRIVATE ultraml)
endif()
