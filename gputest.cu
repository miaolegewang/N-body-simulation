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

#define PI 3.14159265
#define BUFFERSIZE 256
#ifndef BLOCKSIZE
  #define BLOCKSIZE 256
#endif
//#define SOFTPARAMETER 0.2 * RMIN
// #define AU 149597870700.0
// #define R (77871.0 * 1000.0 / AU)
// #define G (4.0 * pow(PI, 2))
#define G 1.0
#define MASS_1 38.2352941              // Center mass of Milky Way
#define MASS_2 38.2352941                // Center mass of Andromeda(M31)
#define NUM_OF_RING_1 12         // Number of rings in 1st galaxy
#define NUM_OF_RING_2 12          // Number of rings in 2nd galaxy
// #define RING_BASE_1 (R * 0.2)       // Radius of first ring in 1st galaxy
// #define RING_BASE_2 (R * 0.2)       // Radius of first ring in 2nd galaxy
#define NUM_P_BASE 12             // Number of particles in the first ring
#define INC_NUM_P 3               // increment of number of particles each step
// #define INC_R_RING (0.5 * R)      // increment of radius of rings each step
#define PMASS 1             // mass of each particle
#define V_PARAMTER 1            // Parameter adding to initial velocity to make it elliptic
//#define RMIN (172.5 / 25)
#define RMIN (7.733/4.5)
#define ECCEN 0.5
#define RMAX ((1.0 + ECCEN) * RMIN / (1.0 - ECCEN))
#define RING_BASE_1 (RMIN * 0.2)       // Radius of first ring in 1st galaxy
#define RING_BASE_2 (RMIN * 0.2)       // Radius of first ring in 2nd galaxy
#define INC_R_RING (RMIN * 0.05)      // increment of radius of rings each step
#define SOFTPARAMETER 0.000001
#define AndromedaXOffsetP -41.0882
#define AndromedaYOffsetP 68.3823
#define AndromedaZOffsetP -33.8634
#define AndromedaXOffsetV 0.2001
#define AndromedaYOffsetV -0.1741
#define AndromedaZOffsetV 0.0864
#define MilkwayXOffsetP 41.0882
#define MilkwayYOffsetP -68.3823
#define MilkwayZOffsetP 33.8634
#define MilkwayXOffsetV -0.2001
#define MilkwayYOffsetV 0.1741
#define MilkwayZOffsetV -0.0864





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
  dt = 0.2;
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
    accel<<< grids, threads >>>(n, x, y, z, vx, vy, vz, mass, dt);
    cudaDeviceSynchronize();
    leapstep<<< grids, threads >>>(n, x, y, z, vx, vy, vz, dt);
    cudaDeviceSynchronize();
    accel<<< grids, threads >>>(n, x, y, z, vx, vy, vz, mass, dt);
    cudaDeviceSynchronize();
    //counter++;
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
  const unsigned int serial = blockIdx.x * BLOCKSIZE + threadIdx.x;
  const unsigned int tdx = threadIdx.x;
  __shared__ double lx[BLOCKSIZE];
  __shared__ double ly[BLOCKSIZE];
  __shared__ double lz[BLOCKSIZE];
  __shared__ double lm[BLOCKSIZE];
  double ax = 0.0, ay = 0.0, az = 0.0, norm, thisX = x[serial], thisY = y[serial], thisZ = z[serial];
  for(int i = 0; i < gridDim.x; i++){
    // Copy data from main memory
    lx[tdx] = x[i * BLOCKSIZE + tdx];
    lz[tdx] = y[i * BLOCKSIZE + tdx];
    ly[tdx] = z[i * BLOCKSIZE + tdx];
    lm[tdx] = mass[i * BLOCKSIZE + tdx];
    __syncthreads();
    // Accumulates the acceleration
    #pragma unroll
    for(int j = 0; j < BLOCKSIZE;j++){
      // loop unrolling
      if(serial == i * BLOCKSIZE + j)
        continue;
      norm = pow(SOFTPARAMETER + pow(thisX - lx[j], 2) + pow(thisY - ly[j], 2) + pow(thisZ - lz[j], 2), 1.5);
      ax += - G * lm[j] * (thisX - lx[j]) / norm;
      ay += - G * lm[j] * (thisY - ly[j]) / norm;
      az += - G * lm[j] * (thisZ - lz[j]) / norm;
    }
    __syncthreads();
  }
  if(serial < n){
    //printf("%d\n", serial);
    vx[serial] += 0.5 * dt * ax;
    vy[serial] += 0.5 * dt * ay;
    vz[serial] += 0.5 * dt * az;
  }
}

__global__ void printstate(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz, int tnow){
  const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
  if(serial < 10000 || (serial > 44000 && serial < 54001)){
    printf("%d, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %d\n", serial, x[serial], y[serial], z[serial], vx[serial], vy[serial], vz[serial], tnow);
  }
}

void initialCondition_host_file(char *input1, char *input2, double **x, double **y, double **z, double **vx, double **vy, double **vz, double **mass, int *size){
  int s1, s2;
  double unknown;
  read_size_from_file(input1, &s1);
  (*size) = s1;
  read_size_from_file(input2, &s2);
  (*size) += s2;
  s1 = (*size) - s2;
  int numOfBlocks = ceil((double)(*size) / BLOCKSIZE);
  // Initial local data array
  double *lx, *ly, *lz, *lvx, *lvy, *lvz, *lm;
  lx = (double*)malloc(numOfBlocks * BLOCKSIZE * 7 * sizeof(double));
  ly = lx + numOfBlocks * BLOCKSIZE;
  lz = ly + numOfBlocks * BLOCKSIZE;
  lvx = lz + numOfBlocks * BLOCKSIZE;
  lvy = lvx + numOfBlocks * BLOCKSIZE;
  lvz = lvy + numOfBlocks * BLOCKSIZE;
  lm = lvz + numOfBlocks * BLOCKSIZE;

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
  double omega = 0.0, sigma = -PI / 2.0;
  while(!feof(fp) && count < s1){
    fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", lm + count, lx + count, ly + count, lz + count, lvx + count, lvy + count, lvz + count);
    rotate(lx + count, ly + count, lz + count, cos(omega), sin(omega), 0, sigma);
    rotate(lvx + count, lvy + count, lvz + count, cos(omega), sin(omega), 0, sigma);
    *(lx + count) += MilkwayXOffsetP;
    *(ly + count) += MilkwayYOffsetP;
    *(lz + count) += MilkwayZOffsetP;
    *(lvx + count) += MilkwayXOffsetV;
    *(lvy + count) += MilkwayYOffsetV;
    *(lvz + count) += MilkwayZOffsetV;
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
  omega = -2 * PI / 3;
  sigma = PI / 6;
  while(!feof(fp) && count < (*size)){
    fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf", lm + count, lx + count, ly + count, lz + count, lvx + count, lvy + count, lvz + count);
    rotate(lx + count, ly + count, lz + count, cos(omega), sin(omega), 0, sigma);
    rotate(lvx + count, lvy + count, lvz + count, cos(omega), sin(omega), 0, sigma);
    *(lx + count) += AndromedaXOffsetP;
    *(ly + count) += AndromedaYOffsetP;
    *(lz + count) += AndromedaZOffsetP;
    *(lvx + count) += AndromedaXOffsetV;
    *(lvy + count) += AndromedaYOffsetV;
    *(lvz + count) += AndromedaZOffsetV;
    count++;
  }
  fclose(fp);
  size_t extra = numOfBlocks * BLOCKSIZE - (*size);
  memset(lx + count, 0, extra * sizeof(double));
  memset(ly + count, 0, extra * sizeof(double));
  memset(lz + count, 0, extra * sizeof(double));
  memset(lvx + count, 0, extra * sizeof(double));
  memset(lvy + count, 0, extra * sizeof(double));
  memset(lvz + count, 0, extra * sizeof(double));
  memset(lm + count, 0, extra * sizeof(double));
  // Allocate device memory
  cudaMalloc((void**)x, numOfBlocks * BLOCKSIZE * 7 * sizeof(double));
  *y = *x + numOfBlocks * BLOCKSIZE;
  *z = *y + numOfBlocks * BLOCKSIZE;
  *vx = *z + numOfBlocks * BLOCKSIZE;
  *vy = *vx + numOfBlocks * BLOCKSIZE;
  *vz = *vy + numOfBlocks * BLOCKSIZE;
  *mass = *vz + numOfBlocks * BLOCKSIZE;
  cudaMemcpy(*x, lx, (size_t)numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(*y, ly, (size_t)numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(*z, lz, (size_t)numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(*vx, lvx, (size_t)numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(*vy, lvy, (size_t)numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(*vz, lvz, (size_t)numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(*mass, lm, (size_t)numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  free(lx);
}

void read_size_from_file(char *input, int *size){
  FILE *fp = fopen(input, "r");
  double unknown;
  fscanf(fp, "%lu", size);
  (*size)++;
  fclose(fp);
}
