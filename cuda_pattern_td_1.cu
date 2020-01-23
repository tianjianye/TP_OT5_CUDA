#include "wb.h"
#define NUM_BINS 4096
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
        file, line);
        if (abort)
            exit(code);
    }
}

__global__ void histo_kernel(unsigned int* buffer, long size, unsigned int* histo)
{
	__shared__ unsigned int histo_private[7];
	if(threadIdx.x<7) {
		histo_private[threadIdx.x]=0;
	}
	__syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride=blockDim.x * gridDim.x;
    while(i<size) {
		atomicAdd(&histo_private[buffer[i]/4],1);
		i+=stride;
	}
	__syncthreads();
	if(threadIdx.x<7){
		atomicAdd(&(histo[threadIdx.x]),histo_private[threadIdx.x]);
	}
}

int main(int argc, char *argv[]) {
    wbArg_t args;
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *deviceInput;
    unsigned int *deviceBins;
    args = wbArg_read(argc, argv);
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0), &inputLength, "Integer");
    hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    wbLog(TRACE, "The input length is ", inputLength);
    wbLog(TRACE, "The number of bins is ", NUM_BINS);
    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	
	size_t bytes = inputLength*sizeof(float);
	cudaMalloc(&deviceInput, bytes);
    cudaMalloc(&deviceBins, bytes);
	
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Allocating GPU memory.");
    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	
	cudaMemcpy( deviceInput, hostInput, bytes, cudaMemcpyHostToDevice);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    // Launch kernel
    // ----------------------------------------------------------
	dim3 block(32);
    dim3 grid ((inputLength + block.x - 1)/block.x);
	histo_kernel<<<grid, block>>>(deviceInput, inputLength, hostBins);
	
    wbLog(TRACE, "Launching kernel");
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Perform kernel computation here
	
	
	
    wbTime_stop(Compute, "Performing CUDA computation");
    wbTime_start(Copy, "Copying output memory to the CPU");

    //@@ Copy the GPU memory back to the CPU here
	
	cudaMemcpy( hostBins, deviceBins, bytes, cudaMemcpyDeviceToHost );
	
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(Copy, "Copying output memory to the CPU");
    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	
	cudaFree(deviceInput);
	cudaFree(deviceBins);
	
    wbTime_stop(GPU, "Freeing GPU Memory");
 
    free(hostBins);
    free(hostInput);
    return 0;
}
