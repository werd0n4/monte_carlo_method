enable_language(CUDA)

add_library(parallel
    monte_carlo.cu
    findMinMax.cu
)
find_library(CURAND_LIBRARY parallel ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(parallel ${CUDA_LIBRARIES})
target_link_libraries(parallel ${CURAND_LIBRARY})



set_property(TARGET parallel PROPERTY CUDA_ARCHITECTURES 61)
set_property(TARGET parallel PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET parallel PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

