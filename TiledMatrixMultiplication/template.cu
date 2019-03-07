#include <wb.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#define TILE_WIDTH 3

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
	int numARows, int numAColumns,
	int numBColumns) {


	int numBRows = numAColumns;
	int numCRows = numARows;
	int numCColumns = numBColumns;

	float CValue = 0;

	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	for (int k = 0; k < (TILE_WIDTH + numAColumns - 1) / TILE_WIDTH; k++) {

		if (k*TILE_WIDTH + threadIdx.x < numAColumns && Row < numARows)
			ds_A[threadIdx.y][threadIdx.x] = A[Row*numAColumns + k*TILE_WIDTH + threadIdx.x];
		else
			ds_A[threadIdx.y][threadIdx.x] = 0.0;

		if (k*TILE_WIDTH + threadIdx.y < numBRows && Col < numBColumns)
			ds_B[threadIdx.y][threadIdx.x] = B[(k*TILE_WIDTH + threadIdx.y)*numBColumns + Col];
		else
			ds_B[threadIdx.y][threadIdx.x] = 0.0;

		__syncthreads();

		for (int n = 0; n < TILE_WIDTH; ++n)
			CValue += ds_A[threadIdx.y][n] * ds_B[n][threadIdx.x];

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns)
		C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns) +
		(blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
}

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostA; // The A matrix
	float *hostB; // The B matrix
	float *hostC; // The output C matrix
	float *deviceA;
	float *deviceB;
	float *deviceC;
	int numARows;    // number of rows in the matrix A
	int numAColumns; // number of columns in the matrix A
	int numBRows;    // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	int numCRows;    // number of rows in the matrix C (you have to set this)
	int numCColumns; // number of columns in the matrix C (you have to set
					 // this)

	hostC = NULL;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
	hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
	//@@ Set numCRows and numCColumns
	if (numAColumns != numBRows)
	{
		printf("The Matrix multiplication is not possible because numAColumns != numBRows\n");
		return -1;
	}
	numCRows = numARows;
	numCColumns = numBColumns;

	int soa = numARows*numAColumns * sizeof(float);
	int sob = numBRows*numBColumns * sizeof(float);
	int soc = numCRows*numCColumns * sizeof(float);
	//@@ Allocate the hostC matrix
	hostC = (float *)malloc(soc);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here
	cudaMalloc((void **)&deviceA, soa);
	cudaMalloc((void **)&deviceB, sob);
	cudaMalloc((void **)&deviceC, soc);
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	//@@ Copy memory to the GPU here
	cudaMemcpy(deviceA, hostA, soa, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, sob, cudaMemcpyHostToDevice);
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	//@@ Initialize the grid and block dimensions here

	dim3 dimGrid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

	//dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	//dim3 dimGrid((float)ceil(numCColumns / TILE_WIDTH), (float)ceil(numCRows / TILE_WIDTH), 1);

	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Launch the GPU Kernel here
	matrixMultiplyShared <<<dimGrid, dimBlock >>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns);
	//cudaDeviceSynchronize();
	cudaThreadSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	//@@ Copy the GPU memory back to the CPU here
	cudaMemcpy(hostC, deviceC, soc, cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	//@@ Free the GPU memory here
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostC, numCRows, numCColumns);

	free(hostA);
	free(hostB);
	free(hostC);

	return 0;
}
