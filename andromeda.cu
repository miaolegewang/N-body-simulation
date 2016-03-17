/*
 *  This program is a CUDA C program simulating the N-body system
 *    of two galaxies as PHY 241 FINAL PROJECTS
 *
 */

/*
 *  TODO:(*for final project)
 *    1. andromeda
 *    2. report
 *    3. presentation
 *	  *4. N-body galaxy code-generat 10^11 particles
 *	  *5. MatLab write a function to track the distance between Milkway and Andromeda
 *	  *6. change accel function to the N-body one.
 *	  *7. print mass[i]. because the halo is dark matter. Or better way distinguish dark matter and rings?
 */


#include <cuda.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include "dataDeclaration.h"
#include "functionDeclaration.h"
#include "otherfunctions.c"


/**     Main function     **/
int main(int argc, char *argv[])
{
  /*
   *  Handling commandline inputs and setting initial value of the arguments
   *    1. number of steps (mstep)
   *    2. warp (nout)
   *    3. offset (start printing position)
   *    4. timestamp (dt)
   *
   */
  int mstep, nout, offset, tnow = 0, n;
  double dt, *x, *y, *z, *vx, *vy, *vz, *mass;
  mstep = argc > 1 ? atoi(argv[1]) : 100;
  nout = argc > 2 ? atoi(argv[2]) : 1;
  offset = argc > 3 ? atoi(argv[3]) : 0;
  dt = argc > 4 ? atof(argv[4]) : 2 * PI * RMIN * RMIN /sqrt(G * MASS_1) / 200.0;
//   dt = argc > 4 ? atof(argv[4]) : 0.1;
  initialCondition_host_file("milky_way.dat", "andromeda.dat", &x, &y, &z, &vx, &vy, &vz, &mass, &n);
  int grids = ceil((double)n / BLOCKSIZE), threads = BLOCKSIZE;
  /*
   *  Use cudaDeviceSetLimit() to change the buffer size of printf
   *   used in kernel functions to solve the problem encountered before:
   *    cannot print more than 4096 lines of data using printf
   *
   */
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, n * BUFFERSIZE);

  /*  Start looping steps from first step to mstep  */
  for(int i = 0; i < offset; i++, tnow++){
    cudaDeviceSynchronize();
    accel<<< grids, threads >>>(n, x, y, z, vx, vy, vz, mass, dt);
    cudaDeviceSynchronize();
    leapstep<<< grids, threads >>>(n, x, y, z, vx, vy, vz, dt);
    cudaDeviceSynchronize();
    accel<<< grids, threads >>>(n, x, y, z, vx, vy, vz, mass, dt);
    cudaDeviceSynchronize();
  }
  for(int i = offset; i < mstep; i++, tnow++){
    if(i % nout == 0)
      printstate<<< grids, threads >>>(n, x, y, z, vx, vy, vz, tnow);
    cudaDeviceSynchronize();

    /*
     *  Use cudaDeviceSynchronize() to wait till all blocks of threads
     *   finish running the kernel functions
     *  Since between each accel() is called, the position of each particle
     *   is updated, which affect the second accel() calls, we need sychronize
     *   in the middle
     *
     */
    accel<<< grids, threads >>>(n, x, y, z, vx, vy, vz, mass, dt);
    cudaDeviceSynchronize();
    leapstep<<< grids, threads >>>(n, x, y, z, vx, vy, vz, dt);
    cudaDeviceSynchronize();
    accel<<< grids, threads >>>(n, x, y, z, vx, vy, vz, mass, dt);
    cudaDeviceSynchronize();
  }
  if(mstep % nout == 0)
    printstate<<< grids, threads >>>(n, x, y, z, vx, vy, vz, tnow);
  cudaDeviceSynchronize();

  // After finishing, free the allocated memory
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  cudaFree(vx);
  cudaFree(vy);
  cudaFree(vz);
  cudaFree(mass);

  // Exit the current thread
  cudaThreadExit();
  return 0;
}

/*
 *  Functions Implmenetation Section
 *
 */
__global__ void initialConditions(int n, double* x, double* y, double* z, double* vx, double* vy, double* vz, double* mass){
  /*  TODO    */
}

void rotate(double* x, double* y, double *z, double n1, double n2, double n3, double theta){

   double tmpx, tmpy, tmpz;
   double a, c, s, sigma;

   sigma = -theta;
   c = cos(sigma);
   s = sin(sigma);
   a = 1 - cos(sigma);


  tmpx = ( a * n1 * n1 + c ) * (*x) + ( a * n1 * n2 - s * n3) * (*y) + ( a * n1 * n3 + s * n2 ) * (*z);
  tmpy = ( a * n1 * n2 + s * n3) * (*x) + ( a * n2 * n2 + c) * (*y) + ( a * n2 * n3 - s * n1 ) * (*z);
  tmpz = ( a * n1 * n3 - s * n2) * (*x) + ( a * n2 * n3 + s * n1) * (*y) + ( a * n3 * n3 + c) * (*z);

  (*x) = tmpx;
  (*y) = tmpy;
  (*z) = tmpz;

}

__global__ void leapstep(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt){
  const unsigned int serial = blockIdx.x * BLOCKSIZE + threadIdx.x;
  if(serial < n){
    x[serial] += dt * vx[serial];
    y[serial] += dt * vy[serial];
    z[serial] += dt * vz[serial];
  }
}


__global__ void accel(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double* mass, double dt){
  const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int tdx = threadIdx.x;
  __shared__ double lx[BLOCKSIZE];
  __shared__ double ly[BLOCKSIZE];
  __shared__ double lz[BLOCKSIZE];
  __shared__ double lm[BLOCKSIZE];

  double ax = 0.0, ay = 0.0, az = 0.0, norm, thisX = x[serial], thisY = y[serial], thisZ = z[serial];
  for(int i = 0; i < gridDim.x; i++){
    // Copy data from main memory
    lx[tdx] = x[i * blockDim.x + tdx];
    lz[tdx] = y[i * blockDim.x + tdx];
    ly[tdx] = z[i * blockDim.x + tdx];
    __syncthreads();
    // Accumulates the acceleration
    for(int j = 0; j < blockDim.x; j++){
      norm = pow(SOFTPARAMETER + pow(thisX - lx[j], 2) + pow(thisY - ly[j], 2) + pow(thisZ - lz[j], 2), 1.5);
      if(i * BLOCKSIZE + j != serial){
        ax += - G * lm[i * blockDim.x + j] * (thisX - lx[j]) / norm;
        ay += - G * lm[i * blockDim.x + j] * (thisY - ly[j]) / norm;
        az += - G * lm[i * blockDim.x + j] * (thisZ - lz[j]) / norm;
      }
    }
  }
  if(serial < n){
    vx[serial] += 0.5 * dt * ax;
    vy[serial] += 0.5 * dt * ay;
    vz[serial] += 0.5 * dt * az;
  }
}

__global__ void printstate(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz, int tnow){
  const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
  if(serial < n){
    printf("%d, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %d\n", serial, x[serial], y[serial], z[serial], vx[serial], vy[serial], vz[serial], tnow);
  }
}

void initialCondition_host_file(char *input1, char *input2, double **x, double **y, double **z, double **vx, double **vy, double **vz, double **mass, int *size){
  FILE *fp1 = fopen(input1, "r");
  FILE *fp2 = fopen(input2, "r");
  if(fp1 == NULL || fp2 == NULL){
    printf("Error: fail to open a file.\n");
    exit(-1);
  }
  int s1, s2;
  double unknown;
  read_size_from_file(input1, &s1);
  (*size) = s1;
  read_size_from_file(input2, &s2);
  (*size) += s2;
  s1 = (*size) - s2;

  // Initial local data array
  double *lx, *ly, *lz, *lvx, *lvy, *lvz, *lm;
  lx = (double*)malloc((*size) * 7 * sizeof(double));
  ly = lx + (*size);
  lz = ly + (*size);
  lvx = lz + (*size);
  lvy = lvx + (*size);
  lvz = lvy + (*size);
  lm = lvz + (*size);

  // Read data from file1
  FILE *fp = fopen(input1, "r");
  if(fp == NULL){
    printf("Error: fail to open file 1\n");
    exit(-1);
  }
  int junk1;
  double junk2;
  int count = 0;
  fscanf(fp, "%lu %lf", &junk1, &junk2);    // skip first line
  while(!feof(fp)){
    fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", lm + count, lx + count, ly + count, lz + count, lvx + count, lvy + count, lvz + count);
    *(lx + count) += MilkwayXOffset;
    *(ly + count) += MilkwayYOffset;
    *(lz + count) += MilkwayZOffset;
    count++;
  }
  fclose(fp);

  // Read data from file2
  fp = fopen(input2, "r");
  if(fp == NULL){
    printf("Error: fail to open file 2\n");
    exit(-1);
  }
  fscanf(fp, "%lu %lf", &junk1, &junk2);    // skip first line
  while(!feof(fp)){
    fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", lm + count, lx + count, ly + count, lz + count, lvx + count, lvy + count, lvz + count);
    *(lx + count) += AndromedaXOffset;
    *(ly + count) += AndromedaYOffset;
    *(lz + count) += AndromedaZOffset;
    count++;
  }
  fclose(fp);
  // Allocate device memory
  int numOfBlocks = ceil((double)(*size) / BLOCKSIZE);
  cudaMalloc((void**)x, numOfBlocks * BLOCKSIZE * 7 * sizeof(double));
  *y = *x + numOfBlocks * BLOCKSIZE;
  *z = *y + numOfBlocks * BLOCKSIZE;
  *vx = *z + numOfBlocks * BLOCKSIZE;
  *vy = *vx + numOfBlocks * BLOCKSIZE;
  *vz = *vy + numOfBlocks * BLOCKSIZE;
  *mass = *vz + numOfBlocks * BLOCKSIZE;
  cudaMemcpy((void**)x, lx, numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy((void**)y, ly, numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy((void**)z, lz, numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy((void**)vx, lvx, numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy((void**)vy, lvy, numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy((void**)vz, lvz, numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy((void**)mass, lm, numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  free(lx);
  fclose(fp1);
  fclose(fp2);
}

void read_size_from_file(char *input, int *size){
  FILE *fp = fopen(input, "r");
  double unknown;
  fscanf(fp, "%lu", size);
  (*size)++;
  fclose(fp);
}
