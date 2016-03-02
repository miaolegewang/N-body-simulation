/*
 *  This program is a CUDA C program simulating the N-body system
 *
 */
#include <cuda.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
//#include "kernel.cu"
#define PI 3.14159265
#define G 1.0	/*  4*pi^2*AU^3*hr(^(-1)	AU=1 */
#define BUFFERSIZE 256
#ifndef BLOCKSIZE
  #define BLOCKSIZE 256
#endif
#define SOFTPARAMETER 0.0000001

__global__ void leapstep(int, double*, double*, double*, double*, double*, double*, double);
__global__ void accel(int, double*, double*, double*, double*, double*, double*, double);
__global__ void printstate(int, double*, double*, double*, double*, double*, double*);
void printstate_host(int, double*, double*, double*, double*, double*, double*);
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
  MASS = 1.0 / n;
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
#ifndef SINGLECORE
  initialConditions<<< grids, threads >>>(n, x, y, z, vx, vy, vz);
  cudaDeviceSynchronize();
#else
  initialCondition_host(n, x, y, z, vx, vy, vz);
#endif
  // Start looping performing integration
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, n * BUFFERSIZE);
  for(int i = 0; i < mstep; i++){
    if(i % nout == 0)
      printstate<<< grids, threads >>>(n, x, y, z, vx, vy, vz);
    cudaDeviceSynchronize();
    /* TODO: rewrite leapstep function to be a host function,
     *  add a kernel function accel to calculate the acceleration
     */
    accel<<< grids, threads >>>(n, x, y, z, vx, vy, vz, dt);
    cudaDeviceSynchronize();
    leapstep<<< grids, threads >>>(n, x, y, z, vx, vy, vz, dt);
    cudaDeviceSynchronize();
    accel<<< grids, threads >>>(n, x, y, z, vx, vy, vz, dt);
    cudaDeviceSynchronize();
  }
  if(mstep % nout == 0)
    printstate<<< grids, threads >>>(n, x, y, z, vx, vy, vz);
  cudaDeviceSynchronize();

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


/* Generate random double-type number within 0 ~ 1.0 */
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

double random_host(){
  return (double)rand() / RAND_MAX;
}

void initialCondition_host(int n, double* x, double* y, double* z, double* vx, double* vy, double* vz){
  srand(time(0));
  double *lx = (double*)malloc(n * sizeof(double));
  double *ly = (double*)malloc(n * sizeof(double));
  double *lz = (double*)malloc(n * sizeof(double));
  double *lvx = (double*)malloc(n * sizeof(double));
  double *lvy = (double*)malloc(n * sizeof(double));
  double *lvz = (double*)malloc(n * sizeof(double));

  double mp = 1.0;
  double centerx = 0.0;
  double centery = 0.0;
  double centervx = 0.0;
  double centervy = 0.0;
  double Rmin = 10 * R;
  double rmin = 0.2 * Rmin;
  double rd = 0.05 * Rmin;
  double nummin = 12;
  double numd = 3;
  double ring = 1;

  revolution = 3;
  nout = 20;
  mp = strtod(argv[1], NULL) * 2.858860e-12;
  n = atoi(argv[2]);
  dt = sqrt(pow(2 * PI, 2) * pow(R, 3) / GMSATURN) / 200;
  double parameter = 0.8;
  if(argc >= 4)
    revolution = atoi(argv[3]);
  if(argc >= 6)
    dt = strtod(argv[5], NULL);
  if(argc >= 5)
    nout = atoi(argv[4]);
  double piece = 2 * PI / n;
  double velocity = sqrt(GMSATURN / R);

  numOfPartical[0] = nummin;
  radiusOfPartical[0] = rmin;

  for(i = 1; i < ring; i++){
      numOfPartical[i] = numOfPartical[i-1] + numd;
      radiusOfPartical[i] = radiusOfPartical[i-1] + rd;
  }

  count = 0;

   for(j = 0; j < ring; j++){
      double piece = 2 * PI / numOfPartical[j];
      double velocity = sqrt(GMSATURN / radiusOfPartical[j]);
      for(i = 0; i < numOfPartical[j]; i++){
        lx[count] = centerx + radiusOfPartical[j] * cos(piece * i);
        ly[count] = centery + radiusOfPartical[j] * sin(piece * i);
        lz[count] = 0;
        lvx[count] = centervx - velocity * sin(piece * i) * parameter;
        lvy[count] = centervy + velocity * cos(piece * i) * parameter;
        lvz[count] = 0;
        count++;
      }
  }

//   for(int i = 0; i < n; i++){
//     double r, rx3;
//     r = 1.0 / sqrt(pow(random_host(), -1.5) - 1);
//     rx3 = random_host();
//     lz[i] = (1 - 2 * random_host()) * r;
//     lx[i] = sqrt(r * r - lz[i] * lz[i]) * cos(2 * PI * rx3);
//     ly[i] = sqrt(r * r - lz[i] * lz[i]) * sin(2 * PI * rx3);
// #ifndef UNSTABLE
//     double rx4, rx5, rx7, Ve, V;
//     Ve = sqrt(2.0) / sqrt(sqrt(1 + r * r));
//     rx4 = random_host();
//     rx5 = random_host();
//     while(0.1 * rx5 >= rx4 * rx4 * pow(1 - rx4 * rx4, 3.5)){
//       rx4 = random_host();
//       rx5 = random_host();
//     }
//     V = Ve * rx4;
//     lvz[i] = (1 - 2 * random_host()) * V;
//     rx7 = random_host();
//     lvx[i] = sqrt(V * V - lvz[i] * lvz[i]) * cos(2 * PI * rx7);
//     lvy[i] = sqrt(V * V - lvz[i] * lvz[i]) * sin(2 * PI * rx7);
// #else
//     lvx[i] = lvy[i] = lvz[i] = 0.0;
// #endif
//   }
  cudaMemcpy(x, lx, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y, ly, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(z, lz, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vx, lvx, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vy, lvy, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vz, lvz, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  free(lx);
  free(ly);
  free(lz);
  free(lvx);
  free(lvy);
  free(lvz);
}


/**
__global__ void leapstep(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt){
  __shared__ double lx[BLOCKSIZE];
  __shared__ double ly[BLOCKSIZE];
  __shared__ double lz[BLOCKSIZE];
  double aX = 0.0, aY = 0.0, aZ = 0.0, norm;
  const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
  if(serial < n){
    double lxcp = x[serial], lycp = y[serial], lzcp = z[serial], MASS = 1.0 / n;
    double lvx = vx[serial], lvy = vy[serial], lvz = vz[serial];
    const unsigned int tx = threadIdx.x;
    const unsigned int numOfBlocks = n / BLOCKSIZE + (n % BLOCKSIZE != 0);
    for(unsigned int i = 0; i < numOfBlocks; i++){
      __syncthreads();
      // Copy smaller blocks of coordinates into shared memory
      lx[tx] = x[i * BLOCKSIZE + tx];
      ly[tx] = y[i * BLOCKSIZE + tx];
      lz[tx] = z[i * BLOCKSIZE + tx];
      //  Wait until all threads finish copying
      __syncthreads();

      //  Calculate the accelation of each particle
       //  accel()

      for(unsigned int j = 0; j < BLOCKSIZE; j++){
        if(serial != i * BLOCKSIZE + j && i * BLOCKSIZE + j < n){
          norm = pow(0.000001 + pow(lxcp - lx[j], 2) + pow(lycp - ly[j], 2) + pow(lzcp - lz[j], 2), 1.5);
          aX += - G * MASS * (lxcp - lx[j]) / norm;
          aY += - G * MASS * (lycp - ly[j]) / norm;
          aZ += - G * MASS * (lzcp - lz[j]) / norm;
        }
      }
    }
    //  Update the position and velocity after accelation
    lvx += 0.5 * dt * aX;
    lvy += 0.5 * dt * aY;
    lvz += 0.5 * dt * aZ;
    lxcp += dt * lvx;
    lycp += dt * lvy;
    lzcp += dt * lvz;
    //  Update position in main memory
    x[serial] = lxcp;
    y[serial] = lycp;
    z[serial] = lzcp;
    aX = aY = aZ = 0.0;
    //  Recalculate the acceleration
    for(unsigned int i = 0; i < numOfBlocks; i++){
      __syncthreads();
      // Copy smaller blocks of coordinates into shared memory
      lx[tx] = x[i * BLOCKSIZE + tx];
      ly[tx] = y[i * BLOCKSIZE + tx];
      lz[tx] = z[i * BLOCKSIZE + tx];
      //  Wait until all threads finish copying
      __syncthreads();

      //  Calculate the accelation of each particle
      // accel()

      for(unsigned int j = 0; j < BLOCKSIZE; j++){
        if(serial != i * BLOCKSIZE + j && i * BLOCKSIZE + j < n){
          norm = pow(0.000001 + pow(lx[tx] - lx[j], 2) + pow(ly[tx] - ly[j], 2) + pow(lz[tx] - lz[j], 2), 1.5);
          aX += - G * MASS * (lxcp - lx[j]) / norm;
          aY += - G * MASS * (lycp - ly[j]) / norm;
          aZ += - G * MASS * (lzcp - lz[j]) / norm;
        }
      }
    }

    //  Copy memory back to the main memory
    vx[serial] += 0.5 * dt * aX;
    vy[serial] += 0.5 * dt * aY;
    vz[serial] += 0.5 * dt * aZ;
  }
}
*/

__global__ void leapstep(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt){
  const unsigned int serial = blockIdx.x * BLOCKSIZE + threadIdx.x;
  if(serial < n){
    x[serial] += dt * vx[serial];
    y[serial] += dt * vy[serial];
    z[serial] += dt * vz[serial];
  }
}

__global__ void accel(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt){
  const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int tdx = threadIdx.x;
  __shared__ double lx[BLOCKSIZE];
  __shared__ double ly[BLOCKSIZE];
  __shared__ double lz[BLOCKSIZE];
  double MASS = 1.0 / n;

  if(serial < n){
    double ax = 0.0, ay = 0.0, az = 0.0, norm, thisX = x[serial], thisY = y[serial], thisZ = z[serial];
    for(int i = 0; i < gridDim.x; i++){
      // Copy data from main memory
      lx[tdx] = x[i * BLOCKSIZE + tdx];
      lz[tdx] = y[i * BLOCKSIZE + tdx];
      ly[tdx] = z[i * BLOCKSIZE + tdx];
      __syncthreads();

      // Accumulates the acceleration
      int itrSize = min(BLOCKSIZE, n - i * BLOCKSIZE);
      for(int j = 0; j < itrSize; j++){
        norm = pow(SOFTPARAMETER + pow(thisX - lx[j], 2) + pow(thisY - ly[j], 2) + pow(thisZ - lz[j], 2), 1.5);
        if(i * BLOCKSIZE + j != serial){
          ax += - G * MASS * (thisX - lx[j]) / norm;
          ay += - G * MASS * (thisY - ly[j]) / norm;
          az += - G * MASS * (thisZ - lz[j]) / norm;
        }
      }
    }

    // Updates velocities in each directions
    vx[serial] += 0.5 * dt * ax;
    vy[serial] += 0.5 * dt * ay;
    vz[serial] += 0.5 * dt * az;
  }
}

__global__ void printstate(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz){
  const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
  if(serial < n){
    printf("%d, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f\n", serial, x[serial], y[serial], z[serial], vx[serial], vy[serial], vz[serial]);
  }
}

void printstate_host(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz){
  double *lx = (double *)malloc(n * sizeof(double));
  double *ly = (double *)malloc(n * sizeof(double));
  double *lz = (double *)malloc(n * sizeof(double));
  double *lvx = (double *)malloc(n * sizeof(double));
  double *lvy = (double *)malloc(n * sizeof(double));
  double *lvz = (double *)malloc(n * sizeof(double));
  cudaMemcpy(lx, x, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(ly, y, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(lz, z, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(lvx, vx, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(lvy, vy, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(lvz, vz, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost);
  for(int i = 0; i < n; i++){
    printf("%d, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f\n", i, lx[i], ly[i], lz[i], lvx[i], lvy[i], lvz[i]);
  }
  free(lx);
  free(ly);
  free(lz);
  free(lvx);
  free(lvy);
  free(lvz);
}
