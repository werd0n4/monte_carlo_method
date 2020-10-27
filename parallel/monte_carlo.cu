
#include "monte_carlo.cuh"


// template<typename return_type, typename arg_type>
__global__ void monte_carlo_parallel(unsigned long seed,
                                    double A, double B,
                                    double min_Y, double max_Y,
                                    int* array, int threads_amount, int gpu_size,
                                    double(*f)(double))
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (gid < threads_amount) {
        int addToScore = 0;
        double X, Y;
		curandState_t state;
		curand_init(seed, gid, 0, &state);

		for(int j = 0; j < gpu_size; ++j){
            X = ((double)(curand(&state) / (double)(0x0FFFFFFFFUL) * (B - A))) + A;
			Y = ((double)(curand(&state) / (double)(0x0FFFFFFFFUL) * (max_Y - min_Y))) + min_Y;

			if ((Y > 0) && (Y <= f(X))) {
				++addToScore;
			}
			else if ((Y < 0) && (Y >= f(X))) {
				--addToScore;
			}
		}
		array[gid] = addToScore;
	}
}

__host__ void print_device_info(){
	int device_number = 0;
    cudaDeviceProp iProp;
    
    cudaGetDeviceProperties(&iProp, device_number);
    
	printf("Device %d: %s\n", device_number, iProp.name);
	printf("Number of multiprocessors:   %d\n", iProp.multiProcessorCount);
	printf("Clockrate: %d\n", iProp.clockRate);
    printf("maxThreadsPerBlock: %d\n", iProp.maxThreadsPerBlock);
    
	printf("maxThreadsDim max dimension of a block x %d\n", iProp.maxThreadsDim[0]);
	printf("maxThreadsDim max dimension of a block y %d\n", iProp.maxThreadsDim[1]);
    printf("maxThreadsDim max dimension of a block z %d\n", iProp.maxThreadsDim[2]);
    
	printf("maxGridSize[3] max size of grid dimension x : %d\n", iProp.maxGridSize[0]);
	printf("maxGridSize[3] max size of grid dimension y : %d\n", iProp.maxGridSize[1]);
	printf("maxGridSize[3] max size of grid dimension z : %d\n", iProp.maxGridSize[2]);
}