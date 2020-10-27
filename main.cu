#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "sequence/monte_carlo.hpp"
#include "parallel/monte_carlo.cuh"


// __host__ double linear_func(double x){
//     return 2*x - 4;
// }

void run_parallel_version(){
	unsigned long cuRand_seed = time(NULL);
	// cudaError error;
	int size = 10000;
    int block_size = 512;
    int score = 0;
    double result;
    int gpu_size = 100000;

    //zahardcodowane przedziały
    double user_A = 1;
    double user_B = 4;
    double randomY_min = -2;
    double randomY_max = 4;
    
    //hostp pointers
	int* gpu_results;
	//device pointers
    int* d_c;
    
	int number_of_bytes = size * sizeof(int);
	//allocate memory for host pointers
    gpu_results = (int*)malloc(number_of_bytes);
    
    //wypełnij gpu_results zerami, o rozmiarze nobytes
	memset(gpu_results, 0, number_of_bytes);

    //allocate device memory
	gpuErrchk(cudaMalloc((int**)&d_c, number_of_bytes));
	
	//start grid
	dim3 block(block_size);
	dim3 grid((size / block.x) + 1);

    //uruchom funkcje z zahardcodowanymi parametrami
	monte_carlo_parallel<<<grid, block>>> (cuRand_seed, user_A, user_B, randomY_min, randomY_max, d_c, size, gpu_size);
	cudaDeviceSynchronize();//wait for all threads to finish

	//memory transfer back to host
	cudaMemcpy(gpu_results, d_c, number_of_bytes, cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; ++i) {
		score += gpu_results[i];
	}
	result = (score / ((double)size*gpu_size)) *
		((user_B - user_A) * (randomY_max - randomY_min));
	
	printf("\nWartosc calki wynosi w przyblizeniu %f\n", result);
    printf("Wartosc stosunku %f\n", (score / ((double)size*gpu_size)));
    
	cudaFree(d_c);
	free(gpu_results);
}

int main()
{
    print_device_info();
    run_parallel_version();
    
    // cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}