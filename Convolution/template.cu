#include <wb.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MASK_WIDTH 5
#define MASK_RADIUS (MASK_WIDTH / 2)
#define O_TILE_WIDTH 16
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE 
//implement the tiled 2D convolution kernel with adjustments for channels
//use shared memory to reduce the number of global accesses, handle the boundary conditions when loading input list elements into the shared memory
//clamp your output values
__global__
void convolution2D(float * I, const float * __restrict__ M, float * P,
	int channels, int width, int height)
{
	__shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	for (int k = 0; k < channels; ++k) {
		int out = ty * O_TILE_WIDTH + tx;
		int col_o = out % BLOCK_WIDTH;
		int row_o = out / BLOCK_WIDTH;
		int row_i = by * O_TILE_WIDTH + row_o - MASK_RADIUS;
		int col_i = bx * O_TILE_WIDTH + col_o - MASK_RADIUS;
		int in = (row_i * width + col_i) * channels + k;

		if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
			N_ds[row_o][col_o] = I[in];
		else
			N_ds[row_o][col_o] = 0.0;

		out = ty * O_TILE_WIDTH + tx + O_TILE_WIDTH * O_TILE_WIDTH;
		row_o = out / BLOCK_WIDTH;
		col_o = out % BLOCK_WIDTH;
		row_i = by * O_TILE_WIDTH + row_o - MASK_RADIUS;
		col_i = bx * O_TILE_WIDTH + col_o - MASK_RADIUS;
		in = (row_i * width + col_i) * channels + k;

		if (row_o < BLOCK_WIDTH) {
			if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
				N_ds[row_o][col_o] = I[in];
			else
				N_ds[row_o][col_o] = 0.0;
		}
		__syncthreads(); // Ensuring that all the elements are loaded before we start computation

		float accum = 0;
		for (int y = 0; y < MASK_WIDTH; ++y)
			for (int x = 0; x < MASK_WIDTH; ++x)
				accum += N_ds[ty + y][tx + x] * M[y * MASK_WIDTH + x];

		int y = by * O_TILE_WIDTH + ty;
		int x = bx * O_TILE_WIDTH + tx;
		if (y < height && x < width)
			P[(y * width + x) * channels + k] = clamp(accum);

		__syncthreads();
	}



}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
  assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  //allocate device memory
  cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  //copy host memory to device
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  //initialize thread block and kernel grid dimensions
  //invoke CUDA kernel	
  dim3 dimBlock(O_TILE_WIDTH, O_TILE_WIDTH, 1);
  dim3 dimGrid((imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1, 1);
  convolution2D <<<dimGrid, dimBlock >>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, imageChannels, imageWidth, imageHeight);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  //copy results from device to host	
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ INSERT CODE HERE
  //deallocate device memory	
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
