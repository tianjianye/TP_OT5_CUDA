#include "wb.h"

#define wbCheck(stmt)\
    do {\
    cudaError_t err = stmt;\
    if (err != cudaSuccess) {\
        wbLog(ERROR, "Failed to run stmt ", #stmt);\
        wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));\
        return -1;\
    }\
} while (0)\

#define BLUR_SIZE 5

//@@ INSERT CODE HERE

__global__ void blurKernel(int height, int width,float *input, float *output) {    
	int Row=threadIdx.x+blockIdx.x*blockDim.x;	
	int Col=threadIdx.y+blockIdx.y*blockDim.y;
	if(Col<width && Row<height){
		for(int k=0;k<3;++k){
			float pixVal=0;
			int pixels=0;
			for(int blurRow=-BLUR_SIZE;blurRow<BLUR_SIZE+1;++blurRow){	
				for(int blurCol=-BLUR_SIZE;blurCol<BLUR_SIZE+1;++blurCol){	
					int curRow=Row+blurRow;
					int curCol=Col+blurCol;
					if(curRow>-1 && curRow<height && curCol>-1 && curCol<width){
						pixVal+=input[(curRow*width+curCol)*3+k];
						pixels++;			
					}
				}
			}	
			output[(Row*width+Col)*3+k]=pixVal/pixels;	
		}
	}
}

int main(int argc, char *argv[]) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    char *inputImageFile;
	char *outputImageFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    float *deviceInputImageData;
    float *deviceOutputImageData;
    args = wbArg_read(argc, argv); /* parse the input arguments */
    inputImageFile = wbArg_getInputFile(args, 0);
	outputImageFile = wbArg_getInputFile(args, 1);
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, 3);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **)&deviceInputImageData,
    imageWidth * imageHeight * sizeof(float)*3);
    cudaMalloc((void **)&deviceOutputImageData,
    imageWidth * imageHeight * sizeof(float)*3);
    wbTime_stop(GPU, "Doing GPU memory allocation");
    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData,
    imageWidth * imageHeight * sizeof(float) * 3,
    cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");
    ///////////////////////////////////////////////////////
    wbTime_start(Compute, "Doing the computation on the GPU");

	dim3 DimGrid((imageHeight-1.0)/16.0+1.0,(imageWidth-1.0)/16.0+1.0,1);
	dim3 DimBlock(16,16,1);
	blurKernel<<<DimGrid, DimBlock>>>(imageHeight, imageWidth, deviceInputImageData, deviceOutputImageData);	
	
    wbTime_stop(Compute, "Doing the computation on the GPU");
    ///////////////////////////////////////////////////////
    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
    imageWidth * imageHeight * sizeof(float) * 3,
    cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");
    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
    
    int i, j;
    FILE *fp = fopen(outputImageFile, "wb"); /* b - binary mode */
    (void) fprintf(fp, "P6\n%d %d\n255\n", imageWidth, imageHeight);
        for (i = 0; i < imageHeight; ++i)
    {
    for (j = 0; j < imageWidth; ++j)
        {
            static unsigned char color[3];
            color[0] = hostOutputImageData[(i*imageWidth+j)*3]*255;  /* red */
            color[1] = hostOutputImageData[(i*imageWidth+j)*3+1]*255;  /* green */
            color[2] = hostOutputImageData[(i*imageWidth+j)*3+2]*255;  /* blue */

            (void) fwrite(color, 1, 3, fp);
        }
    }
    (void) fclose(fp);
    
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);
    return 0;
}
