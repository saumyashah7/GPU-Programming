#include <wb.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 512 

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *aux, int len) {
    //@@ Modify the body of this kernel to generate the scanned blocks
    //@@ Make sure to use the workefficient version of the parallel scan
    //@@ Also make sure to store the block sum to the aux array 
	__shared__ float XY[2 * BLOCK_SIZE];

	//loading data from global memory to shared memory
	int i = 2 * blockIdx.x*blockDim.x + threadIdx.x;
	if (i<len)
		XY[threadIdx.x] = input[i];
	else
		XY[threadIdx.x] = 0;


	if (i + blockDim.x<len)
		XY[threadIdx.x + blockDim.x] = input[i + blockDim.x];
	else
		XY[threadIdx.x + blockDim.x] = 0;
		
	__syncthreads();
	
	//Reduction Phase
	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
	{
		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
			XY[index] += XY[index - stride];
		__syncthreads();
	}

	//Post Reduction Phase
	for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index + stride < 2 * BLOCK_SIZE) 
			XY[index + stride] += XY[index];
		
		__syncthreads();
	}
	
	//writing data to output
	if (i < len) 
		output[i] = XY[threadIdx.x];
	if (i + blockDim.x < len) 
		output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
	if(aux!=NULL && threadIdx.x==0)
		aux[blockIdx.x] = XY[2 * blockDim.x - 1];

     
}

__global__ void addScannedBlockSums(float *input, float *aux, int len) {
	//@@ Modify the body of this kernel to add scanned block sums to 
	//@@ all values of the scanned blocks
	int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

	if (blockIdx.x > 0) {

		if (i < len)
			input[i] += aux[blockIdx.x - 1];

		if (i + blockDim.x < len)
			input[i + blockDim.x] += aux[blockIdx.x - 1];
	}

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output 1D list
  float *deviceInput;
  float *deviceOutput;
  float *deviceAuxArray, *deviceAuxScannedArray;
  int numElements; // number of elements in the input/output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating device memory.");
  //@@ Allocate device memory
  //you can assume that aux array size would not need to be more than BLOCK_SIZE*2 (i.e., 1024)
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxScannedArray, 2 * BLOCK_SIZE  * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxArray, 2 * BLOCK_SIZE * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating device memory.");

  wbTime_start(GPU, "Clearing output device memory.");
  //@@ zero out the deviceOutput using cudaMemset() by uncommenting the below line
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output device memory.");

  wbTime_start(GPU, "Copying input host memory to device.");
  //@@ Copy input host memory to device	
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input host memory to device.");

  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  //@@ You need to launch scan kernel twice: 1) for generating scanned blocks 
  //@@ (hint: pass deviceAuxArray to the aux parameter)
  //@@ and 2) for generating scanned aux array that has the scanned block sums. 
  //@@ (hint: pass NULL to the aux parameter)
  //@@ Then you should call addScannedBlockSums kernel.
  int numBlocks = ceil(numElements / float(BLOCK_SIZE * 2));

  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  scan<<<dimGrid, dimBlock >> >(deviceInput, deviceOutput, deviceAuxArray, numElements);
  cudaDeviceSynchronize();

  dim3 singleGrid(1, 1, 1);
  scan<<<singleGrid, dimBlock >> >(deviceAuxArray, deviceAuxScannedArray, NULL, BLOCK_SIZE * 2);
  cudaDeviceSynchronize();

  addScannedBlockSums <<<dimGrid, dimBlock >> >(deviceOutput, deviceAuxScannedArray, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output device memory to host");
  //@@ Copy results from device to host	
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output device memory to host");

  wbTime_start(GPU, "Freeing device memory");
  //@@ Deallocate device memory
  cudaFree(deviceInput);
  cudaFree(deviceAuxScannedArray);
  cudaFree(deviceAuxArray);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing device memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
