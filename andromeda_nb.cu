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
#define BUFFERSIZE 500
#define BLOCKSIZE 256
#define G 1.0
#define MASS_1 38.2352941
#define RMIN (7.733/4.5)
#define SOFTPARAMETER 0.000001
#define AndromedaXOffsetP -41.0882
#define AndromedaYOffsetP 68.3823
#define AndromedaZOffsetP -33.8634
#define AndromedaXOffsetV 0.0420
#define AndromedaYOffsetV -0.2504
#define AndromedaZOffsetV 0.1240
#define MilkwayXOffsetP 41.0882
#define MilkwayYOffsetP -68.3823
#define MilkwayZOffsetP 33.8634
#define MilkwayXOffsetV -0.0420
#define MilkwayYOffsetV 0.2504
#define MilkwayZOffsetV -0.1240

// Headers
void rotate(double* x, double* y, double *z, double n1, double n2, double n3, double theta);
__global__ void leapstep(unsigned long n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt);
__global__ void accel(unsigned long n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double* mass, double dt);
__global__ void printstate(double *x, double *y, double *z, unsigned long tnow);
void initialCondition_host_file(char *input1, char *input2, double **x, double **y, double **z, double **vx, double **vy, double **vz, double **mass, unsigned long *size);
void read_size_from_file(char *input, unsigned long *size) ;

/**     Main function     **/
int main(int argc, char *argv[]) {
  /*
   *  Handling commandline inputs and setting initial value of the arguments
   *    1. number of steps (mstep)
   *    2. warp (nout)
   *    3. offset (start printing position)
   *    4. timestamp (dt)
   *
   */
  unsigned long mstep, nout, offset, tnow = 0, n;
  double dt, *x, *y, *z, *vx, *vy, *vz, *mass;

  mstep = (argc > 1) ? atoi(argv[1]) : 100;
  nout = (argc > 2) ? atoi(argv[2]) : 1;
  offset = (argc > 3) ? atoi(argv[3]) : 0;
  dt = (argc > 4) ? atof(argv[4]) : (2.0 * PI * RMIN * RMIN) / (sqrt(G * MASS_1) * 40.0);

  initialCondition_host_file("milky_way.dat", "andromeda.dat", &x, &y, &z, &vx, &vy, &vz, &mass, &n);

  unsigned long grids = ceil((double)n / BLOCKSIZE), threads = BLOCKSIZE;

  /*
   *  Use cudaDeviceSetLimit() to change the buffer size of printf
   *   used in kernel functions to solve the problem encountered before:
   *    cannot print more than 4096 lines of data using printf
   *
   */
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, n * BUFFERSIZE);

  /*  Start looping steps from first step to mstep  */
  for (unsigned long i = 0; i < offset; i++, tnow++){
    accel<<<grids, BLOCKSIZE>>> (n, x, y, z, vx, vy, vz, mass, dt);
    cudaDeviceSynchronize();
    leapstep<<<grids, BLOCKSIZE>>> (n, x, y, z, vx, vy, vz, dt);
    cudaDeviceSynchronize();
    accel<<<grids, BLOCKSIZE>>> (n, x, y, z, vx, vy, vz, mass, dt);
    cudaDeviceSynchronize();
  }
  for (unsigned long i = offset; i < mstep; i++, tnow++) {
    if(i % nout == 0) {
      printstate<<<grids, threads>>> (x, y, z, tnow);
      cudaDeviceSynchronize();
    }

    accel<<<grids, BLOCKSIZE>>> (n, x, y, z, vx, vy, vz, mass, dt);
    cudaDeviceSynchronize();
    leapstep<<<grids, BLOCKSIZE>>> (n, x, y, z, vx, vy, vz, dt);
    cudaDeviceSynchronize();
    accel<<<grids, BLOCKSIZE>>> (n, x, y, z, vx, vy, vz, mass, dt);
    cudaDeviceSynchronize();
  }

  if(mstep % nout == 0) {
    printstate<<<grids, BLOCKSIZE>>>(x, y, z, tnow);
  }
  cudaDeviceSynchronize();

  // After finishing, free the allocated memory
  cudaFree(x);

  // Exit the current thread
  return 0;
}

void rotate(double* x, double* y, double *z, double n1, double n2, double n3, double theta) {
  double sigma = -theta;
  double c = cos(sigma);
  double s = sin(sigma);
  double a = 1 - cos(sigma);

  double tmpx = ( a * n1 * n1 + c ) * (*x) + ( a * n1 * n2 - s * n3 ) * (*y) + ( a * n1 * n3 + s * n2 ) * (*z);
  double tmpy = ( a * n1 * n2 + s * n3 ) * (*x) + ( a * n2 * n2 + c ) * (*y) + ( a * n2 * n3 - s * n1 ) * (*z);
  double tmpz = ( a * n1 * n3 - s * n2 ) * (*x) + ( a * n2 * n3 + s * n1 ) * (*y) + ( a * n3 * n3 + c ) * (*z);

  (*x) = tmpx;
  (*y) = tmpy;
  (*z) = tmpz;
}

__global__ void leapstep(unsigned long n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt) {
  const unsigned long serial = blockIdx.x * blockDim.x + threadIdx.x;
  if (serial < n){
    x[serial] += dt * vx[serial];
    y[serial] += dt * vy[serial];
    z[serial] += dt * vz[serial];
  }
}

__global__ void accel(unsigned long n, double *x, double *y, double *z, double *vx, double *vy, double *vz, double* mass, double dt) {
  const unsigned long serial = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long tdx = threadIdx.x;

  __shared__ double lx[BLOCKSIZE];
  __shared__ double ly[BLOCKSIZE];
  __shared__ double lz[BLOCKSIZE];
  __shared__ double lm[BLOCKSIZE];

  double ax = 0.0, ay = 0.0, az = 0.0;
  double norm;
  double thisX, thisY, thisZ;
  if (serial < n) {
    thisX = x[serial];
    thisY = y[serial];
    thisZ = z[serial];
  }

  for (unsigned long i = 0; i < gridDim.x; i++) {
    unsigned long index = i * blockDim.x + tdx;
    if (index < n) {
      // Copy data from main memory
      lx[tdx] = x[index];
      lz[tdx] = y[index];
      ly[tdx] = z[index];
      lm[tdx] = mass[index];
    }
    __syncthreads();

    // Accumulates the acceleration
    #pragma unroll
    for (unsigned long j = 0; j < BLOCKSIZE; j++) {
      unsigned long pos = i * blockDim.x + j;
      if (pos >= n) {
        continue;
      }

      norm = pow(SOFTPARAMETER + pow(thisX - lx[j], 2) + pow(thisY - ly[j], 2) + pow(thisZ - lz[j], 2), 1.5);
      ax += - G * lm[j] * (thisX - lx[j]) / norm;
      ay += - G * lm[j] * (thisY - ly[j]) / norm;
      az += - G * lm[j] * (thisZ - lz[j]) / norm;
    }

    __syncthreads();
  }

  if (serial < n) {
    vx[serial] += 0.5 * dt * ax;
    vy[serial] += 0.5 * dt * ay;
    vz[serial] += 0.5 * dt * az;
  }
}

__global__ void printstate(double *x, double *y, double *z, unsigned long tnow) {
  const unsigned long serial = blockIdx.x * blockDim.x + threadIdx.x;
  if(serial < 10000 || (serial >= 44000 && serial < 54000)){
    printf("%d,%12.6lf,%12.6lf,%12.6lf,%d\n", serial, x[serial], y[serial], z[serial], tnow);
  }
}

void initialCondition_host_file(char *input1, char *input2, double **x, double **y, double **z, double **vx, double **vy, double **vz, double **mass, unsigned long *size) {
  unsigned long s1, s2;
  read_size_from_file(input1, &s1);
  (*size) = s1;
  read_size_from_file(input2, &s2);
  (*size) += s2;
  unsigned long numOfBlocks = ceil(((double)(*size)) / BLOCKSIZE);

  // Initial local data array
  double *lx, *ly, *lz, *lvx, *lvy, *lvz, *lm;
  lx = (double*) malloc(7 * numOfBlocks * BLOCKSIZE * sizeof(double));
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
  unsigned long count = 0;

  // Skip first galaxy
  unsigned long junk1;
  double junk2;
  fscanf(fp, "%lu %lf\n", &junk1, &junk2);

  double omega = 0.0;
  double sigma = PI / 2.0;

  while((!feof(fp)) && (count < s1)){
    fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf\n", lm + count, lx + count, ly + count, lz + count, lvx + count, lvy + count, lvz + count);
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

  // Skip first line
  fscanf(fp, "%lu %lf\n", &junk1, &junk2);

  omega = - 2.0 * PI / 3.0;
  sigma = PI / 6.0;

  while((!feof(fp)) && (count < (*size))){
    fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf\n", lm + count, lx + count, ly + count, lz + count, lvx + count, lvy + count, lvz + count);
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

  // Allocate device memory
  cudaMalloc(x, 7 * numOfBlocks * BLOCKSIZE * sizeof(double));
  (*y) = (*x) + numOfBlocks * BLOCKSIZE;
  (*z) = (*y) + numOfBlocks * BLOCKSIZE;
  (*vx) = (*z) + numOfBlocks * BLOCKSIZE;
  (*vy) = (*vx) + numOfBlocks * BLOCKSIZE;
  (*vz) = (*vy) + numOfBlocks * BLOCKSIZE;
  (*mass) = (*vz) + numOfBlocks * BLOCKSIZE;
  cudaMemcpy((*x), lx, 7 * numOfBlocks * BLOCKSIZE * sizeof(double), cudaMemcpyHostToDevice);
  free(lx);
}

void read_size_from_file(char *input, unsigned long *size) {
  FILE *fp = fopen(input, "r");
  fscanf(fp, "%lu", size);
  fclose(fp);
}
