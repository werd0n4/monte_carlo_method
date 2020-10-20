cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(MonteCarlo LANGUAGES CXX CUDA)


add_subdirectory(parallel)
add_subdirectory(sequence)

add_executable(Main main.cpp)


target_link_libraries(Main PUBLIC parallel)
target_link_libraries(Main PUBLIC sequence)

target_include_directories(Main PUBLIC
    ./sequence
    ./parallel
)   

# set_property(TARGET main 
#              PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET Main PROPERTY CUDA_ARCHITECTURES 61)
