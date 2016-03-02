/*
 *  This program is a CUDA C program simulating the N-body system
 *  TODO: This program will fail if particles number exceeds 4096
 *   Reason Unknown
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
//#include "kernel.cu"
#define PI 3.14159265
#define G 1.0	/*  4*pi^2*AU^3*hr(^(-1)	AU=1 */
#ifndef BLOCKSIZE
  #define BLOCKSIZE 256
#endif

__global__ void leapstep(int, double*, double*, double*, double*, double*, double*, double);
__global__ void printstate(int, double*, double*, double*, double*, double*, double*);
__global__ void initialConditions(int, double*, double*, double*, double*, double*, double*);
void initialCondition_host(int, double*, double*, double*, double*, double*, double*);
void setGrid();

int main(int argc, char *argv[])
{
  // Handling commandline inputs and setting initial value of the arguments
  int n, mstep, nout;
  double dt, *x, *y, *z, *vx, *vy, *vz;
  n = argc > 1 ? atoi(argv[1]) : 10000;
  mstep = argc > 2 ? atoi(argv[2]) : 200;
  nout = argc > 3 ? atoi(argv[3]) : 1;
  dt = argc > 4 ? atof(argv[4]) : 0.05;

  // setup execution configuration
  //dim3 threads(BLOCKSIZE);
  int numOfBlocks = n / BLOCKSIZE + (n % BLOCKSIZE != 0);
  //dim3 grids(numOfBlocks);
  int threads = BLOCKSIZE, grids = numOfBlocks;
  //setGrid(n, threads, grids);

  // Setup initial conditions for each of the particles
  cudaMalloc((void**) &x, (size_t)(n * sizeof(double)));
  //cudaMemset((void**) x, 0, (size_t)(n * sizeof(double)));
  cudaMalloc((void**) &y, (size_t)(n * sizeof(double)));
  //cudaMemset((void**) y, 0, (size_t)(n * sizeof(double)));
  cudaMalloc((void**) &z, (size_t)(n * sizeof(double)));
  //cudaMemset((void**) z, 0, (size_t)(n * sizeof(double)));
  cudaMalloc((void**) &vx, (size_t)(n * sizeof(double)));
  ///cudaMemset((void**) vx, 0, (size_t)(n * sizeof(double)));
  cudaMalloc((void**) &vy, (size_t)(n * sizeof(double)));
  //cudaMemset((void**) vy, 0, (size_t)(n * sizeof(double)));
  cudaMalloc((void**) &vz, (size_t)(n * sizeof(double)));
  //cudaMemset((void**) vz, 0, (size_t)(n * sizeof(double)));

  initialConditions<<< grids, threads >>>(n, x, y, z, vx, vy, vz);
  //initialConditions(n, x, y, z, vx, vy, vz);
  cudaDeviceSynchronize();

  // Start looping performing integration
  for(int i = 0; i < mstep; i++){
    if(i % nout == 0)
      //printstate(n, x, y, z, vx, vy, vz);
      printstate<<< grids, threads >>>(n, x, y, z, vx, vy, vz);
    leapstep<<< grids, threads >>>(n, x, y, z, vx, vy, vz, dt);
    //leapstep(n, x, y, z, vx, vy, vz, dt);
    cudaDeviceSynchronize();
  }
  if(mstep % nout == 0)
    printstate<<< grids, threads >>>(n, x, y, z, vx, vy, vz);

  cudaDeviceSynchronize();
    //printstate(n, x, y, z, vx, vy, vz);
  // After finishing, free the allocated memory
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  cudaFree(vx);
  cudaFree(vy);
  cudaFree(vz);

  // Exit the current thread
  cudaThreadExit();
  return 0;
}


/*
 *  setGrid set the dimension of grid and dimension of block
 */
void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
}


/* TODO Need to add random variable */
__device__ double randomX(){
  curandState state;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(0, idx, 0, &state);
  return curand_uniform_double(&state);
}

__global__ void initialConditions(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz)
{
  const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
  if(serial < n){
    // if serial number is smaller than n
    // then need to initial data
    double r, rx3;
    r = 1.0 / sqrt(pow(randomX(), -1.5) - 1);
    rx3 = randomX();
    z[serial] = (1 - 2 * randomX()) * r;
    x[serial] = sqrt(r * r - z[serial] * z[serial]) * cos(2 * PI * rx3);
    y[serial] = sqrt(r * r - z[serial] * z[serial]) * sin(2 * PI * rx3);
#ifndef UNSTABLE
    double rx4, rx5, rx7, Ve, V;
    Ve = sqrt(2.0) / sqrt(sqrt(1 + r * r));
    rx4 = randomX();
    rx5 = randomX();
    while(0.1 * rx5 >= rx4 * rx4 * pow(1 - rx4 * rx4, 3.5)){
      rx4 = randomX();
      rx5 = randomX();
    }
    V = Ve * rx4;
    vz[serial] = (1 - 2 * randomX()) * V;
    rx7 = randomX();
    vx[serial] = sqrt(V * V - vz[serial] * vz[serial]) * cos(2 * PI * rx7);
    vy[serial] = sqrt(V * V - vz[serial] * vz[serial]) * sin(2 * PI * rx7);
#else
    vx[serial] = vy[serial] = vz[serial] = 0.0;
#endif
  }
}

__global__ void leapstep(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt){
  __shared__ double lx[BLOCKSIZE];
  __shared__ double ly[BLOCKSIZE];
  __shared__ double lz[BLOCKSIZE];
  double aX = 0.0, aY = 0.0, aZ = 0.0, norm;
  const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
  if(serial < n){
    double lxcp = x[serial], lycp = y[serial], lzcp = z[serial], mass = 1.0 / n;
    double lvx = vx[serial], lvy = vy[serial], lvz = vz[serial];
    const unsigned int tx = threadIdx.x;
    const unsigned int numOfBlocks = n / BLOCKSIZE + (n % BLOCKSIZE != 0);
    for(unsigned int i = 0; i < numOfBlocks; i++){
      __syncthreads();
      /* Copy smaller blocks of coordinates into shared memory  */
      lx[tx] = x[i * BLOCKSIZE + tx];
      ly[tx] = y[i * BLOCKSIZE + tx];
      lz[tx] = z[i * BLOCKSIZE + tx];
      /*  Wait until all threads finish copying   */
      __syncthreads();

      /*  Calculate the accelation of each particle
       *  accel()
       */
      for(unsigned int j = 0; j < BLOCKSIZE; j++){
        if(serial != i * BLOCKSIZE + j && i * BLOCKSIZE + j < n){
          norm = pow(0.000001 + pow(lxcp - lx[j], 2) + pow(lycp - ly[j], 2) + pow(lzcp - lz[j], 2), 1.5);
          aX += - G * mass * (lxcp - lx[j]) / norm;
          aY += - G * mass * (lycp - ly[j]) / norm;
          aZ += - G * mass * (lzcp - lz[j]) / norm;
        }
      }
    }
    /*  Update the position and velocity after accelation */
    lvx += 0.5 * dt * aX;
    lvy += 0.5 * dt * aY;
    lvz += 0.5 * dt * aZ;
    lxcp += dt * lvx;
    lycp += dt * lvy;
    lzcp += dt * lvz;
    /*  Update position in main memory  */
    x[serial] = lxcp;
    y[serial] = lycp;
    z[serial] = lzcp;
    aX = aY = aZ = 0.0;
    /*  Recalculate the acceleration   */
    for(unsigned int i = 0; i < numOfBlocks; i++){
      __syncthreads();
      /* Copy smaller blocks of coordinates into shared memory  */
      lx[tx] = x[i * BLOCKSIZE + tx];
      ly[tx] = y[i * BLOCKSIZE + tx];
      lz[tx] = z[i * BLOCKSIZE + tx];
      /*  Wait until all threads finish copying   */
      __syncthreads();

      /*  Calculate the accelation of each particle
       *  accel()
       */
      for(unsigned int j = 0; j < BLOCKSIZE; j++){
        if(serial != i * BLOCKSIZE + j && i * BLOCKSIZE + j < n){
          norm = pow(0.000001 + pow(lx[tx] - lx[j], 2) + pow(ly[tx] - ly[j], 2) + pow(lz[tx] - lz[j], 2), 1.5);
          aX += - G * mass * (lxcp - lx[j]) / norm;
          aY += - G * mass * (lycp - ly[j]) / norm;
          aZ += - G * mass * (lzcp - lz[j]) / norm;
        }
      }
    }

    /*  Copy memory back to the main memory */
    vx[serial] += 0.5 * dt * aX;
    vy[serial] += 0.5 * dt * aY;
    vz[serial] += 0.5 * dt * aZ;
  }
}

__global__ void accel(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz){
  __shared__ double lx[BLOCKSIZE];
  __shared__ double ly[BLOCKSIZE];
  __shared__ double lz[BLOCKSIZE];

}

__global__ void printstate(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz){
  const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
  if(serial < n){
    printf("%d, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f\n", serial, x[serial], y[serial], z[serial], vx[serial], vy[serial], vz[serial]);
  }
}
