enable_language(CUDA)

add_library(parallel
    #headers
    monte_carlo.cu
    #src
    
)

set_property(TARGET parallel PROPERTY CUDA_ARCHITECTURES 61)
