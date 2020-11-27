enable_language(CUDA)

add_library(parallel
    monte_carlo.cu
    findMinMax.cu
)

find_package(CUDA)
target_link_libraries(parallel ${CUDA_LIBRARIES})
target_link_libraries(parallel ${CUDA_curand_LIBRARY})

set_property(TARGET parallel PROPERTY CUDA_ARCHITECTURES 61)
set_property(TARGET parallel PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET parallel PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

