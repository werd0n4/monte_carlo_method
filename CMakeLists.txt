cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(MonteCarlo LANGUAGES CUDA CXX)



add_subdirectory(parallel)
add_subdirectory(sequence)

add_executable(Main main.cu)




target_link_libraries(Main PUBLIC parallel)
target_link_libraries(Main PUBLIC sequence)



target_include_directories(Main PUBLIC
    ./sequenc
	./parallel
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)   

set_property(TARGET Main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET Main PROPERTY CUDA_ARCHITECTURES 61)
set_property(TARGET Main PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)



