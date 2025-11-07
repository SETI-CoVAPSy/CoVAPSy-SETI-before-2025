/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Template project which demonstrates the basics on how to setup a project 
 * example application.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <iostream>
using namespace std;

// includes CUDA
#include <cuda_runtime.h>

#include "seuillage.h"

__global__ void seuillage_kernel(float d_image_in[][SIZE_J][SIZE_I],float d_image_out[][SIZE_J][SIZE_I]) {
    // Calculate the global thread indices
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    // Check if the thread indices are within the image bounds
    if (i < SIZE_J && j < SIZE_I) {
        // Extract the RGB values
        float r = d_image_in[0][i][j]; // Red channel
        float g = d_image_in[1][i][j]; // Green channel
        float b = d_image_in[2][i][j]; // Blue channel

        // Calculate the intensity I
        float denominator = sqrt(r * r + g * g + b * b);
        float I = 0.0f; // Initialize I

        // Avoid division by zero
        if (denominator > 0.0f) {
            I = r / denominator;
        }

        // Create the red mask
        if (I > 0.7) {
            // If the mask condition is met, modify the output image
            d_image_out[0][i][j] = r; // Red channel remains the same
            d_image_out[1][i][j] = r; // Green channel gets red value
            d_image_out[2][i][j] = b; // Blue channel remains the same
        } else {
            // If the mask condition is not met, keep the original values
            d_image_out[0][i][j] = r;
            d_image_out[1][i][j] = g;
            d_image_out[2][i][j] = b;
        }
    }
}



////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);




////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	runTest( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
	cudaError_t error;

	if (argc<2)
		printf("indiquer le chemin du repertoire contenant les images\n");

	const unsigned int mem_size = sizeof(float) * 3* SIZE_J * SIZE_I;
	// allocate host memory
	float* h_image_in = (float*) malloc(mem_size);
	
	printf("memsize: %u B\n", mem_size);


	//Initilaisation du volume d'entr�e
	FILE *file_ptr;
	char name_file_in[512];
	sprintf(name_file_in,"%s/base_image.raw",argv[1]);
	file_ptr=fopen(name_file_in,"rb");
	fread(h_image_in,sizeof(float),3*SIZE_J*SIZE_I,file_ptr);
	fclose(file_ptr);


	////////////////////////////////////////////////////////////////////////////////
	// EXECUTION SUR LE CPU
	///////////////////////////////////////////////////////////////////////


	// Image traite sur le CPU
	float* h_image_out_CPU = (float*) malloc( mem_size);

	// printf("Seuillage CPU d'une image couleur \n");

	cudaEvent_t start,stop;
	error = cudaEventCreate(&start);
	error = cudaEventCreate(&stop);

	//! Record the start event
	error = cudaEventRecord(start, NULL);
	error = cudaEventSynchronize(start);
	//Seuillage sur CPU
	seuillage_C( (float (*)[SIZE_J][SIZE_I])h_image_out_CPU, (float (*)[SIZE_J][SIZE_I])h_image_in);

	//! Record the start event
	error = cudaEventRecord(stop, NULL);
	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);


	printf("cpu_time: %f ms\n",msecTotal);

	//Sauvegarde de l'image resultat
	char name_file_out_CPU[512];
	sprintf(name_file_out_CPU,"%s/base_image_out_CPU.raw",argv[1]);
	file_ptr=fopen(name_file_out_CPU,"wb");
	fwrite(h_image_out_CPU,sizeof(float),3*SIZE_J*SIZE_I,file_ptr);
	fclose(file_ptr);


	////////////////////////////////////////////////////////////////////////////////
	// EXECUTION SUR LE GPU
	///////////////////////////////////////////////////////////////////////

	cudaEvent_t start_mem,stop_mem;
	error = cudaEventCreate(&start_mem);
	error = cudaEventCreate(&stop_mem);

	//! Wait for the start event to complete
	error = cudaEventRecord(start, NULL);
	error = cudaEventSynchronize(start);


	float* h_image_out_GPU = (float*) malloc(mem_size);

	// images on device memory
	float* d_image_in;
	float* d_image_out;

	// Allocate memory for d_image_in and d_image_out on the GPU
	error = cudaMalloc((void**)&d_image_in, mem_size);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error allocating device memory for d_image_in: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void**)&d_image_out, mem_size);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error allocating device memory for d_image_out: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy host memory to device
	error = cudaMemcpy(d_image_in, h_image_in, mem_size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error copying data from host to device: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaEventRecord(stop_mem, NULL);
	//! Wait for the stop event to complete
	error = cudaEventSynchronize(stop_mem);
	float msecMem = 0.0f;
	error = cudaEventElapsedTime(&msecMem, start, stop_mem);

	// setup execution parameters -> découpage en threads

	// Define grid and block sizes
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // Example block size
	dim3 gridSize((SIZE_I + blockSize.x - 1) / blockSize.x, (SIZE_J + blockSize.y - 1) / blockSize.y);

	// Launch the kernel
	seuillage_kernel<<<gridSize, blockSize>>>((float (*)[SIZE_J][SIZE_I])d_image_in, (float (*)[SIZE_J][SIZE_I])d_image_out);
	// Check for errors during kernel launch
	// cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "Error launching kernel: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the GPU to finish before accessing the results
	cudaDeviceSynchronize();

	// INDICATION : pour les parametres de la fonction kernel seuillage_kernel, vous ferez un changement de type (float *) vers  (float (*)[SIZE_J][SIZE_I])
	// inspirez vous du lancement de la fonction seuillage_C dans le main.

	// Record the start event

	error = cudaEventRecord(start_mem, NULL);
	error = cudaEventSynchronize(start_mem);

	// Copy result from device to host
	error = cudaMemcpy(h_image_out_GPU, d_image_out, mem_size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		fprintf(stderr, "Error copying data from device to host: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// At this point, h_image_out_GPU contains the processed image data

	// cleanup device memory
	//ENLEVEZ LES COMMENTAIRES
	cudaFree(d_image_in);
	cudaFree(d_image_out);


	error = cudaEventRecord(stop, NULL);
	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);
	msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	float msecMem2 =0.0f;
	error = cudaEventElapsedTime(&msecMem2, start_mem, stop);
	msecMem+=msecMem2;

	printf("gpu_time: %f ms\nmem_percentage: %2.2f percent\nmemtime: %f ms\n",msecTotal,(msecMem)/(msecTotal)*100, msecMem);

	// Enregistrement de l'image de sortie sur un fichier
	char name_file_out_GPU[512];
	sprintf(name_file_out_GPU,"%s/base_image_out_GPU.raw",argv[1]);
	file_ptr=fopen(name_file_out_GPU,"wb");
	fwrite(h_image_out_GPU,sizeof(float),3*SIZE_J*SIZE_I,file_ptr);
	fclose(file_ptr);


	// cleanup memory
	free(h_image_in);
	free(h_image_out_GPU);
	free(h_image_out_CPU);


}
