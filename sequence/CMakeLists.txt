enable_language(CUDA)

add_library(sequence
    #headers
    

    #src
    monte_carlo.cpp
)

set_property(TARGET sequence PROPERTY CUDA_ARCHITECTURES 61)