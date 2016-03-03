/*
**  This program is a CUDA C program simulating the N-body system
**    of two galaxies as PHY 241 FINAL PROJECTS
**
*/
#include <cuda.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

/*
**  Modify the constant parameters if neccessary
**    Constant Section
*/
#define PI 3.14159265
#define BUFFERSIZE 256
#ifndef BLOCKSIZE
  #define BLOCKSIZE 256
#endif
#define SOFTPARAMETER 0.0000001
#define AU 149597870700.0
#define R (77871.0 * 1000.0 / AU)
#define G (4.0 * pow(PI, 2) / pow(365.0, 2) * 285.8860e-06)
#define MASS_1 1.0                // Center mass of 1st galaxy
#define MASS_2 1.0                // Center mass of 2nd galaxy
#define NUM_OF_RING_1 20          // Number of rings in 1st galaxy
#define NUM_OF_RING_2 20          // Number of rings in 2nd galaxy
#define RING_BASE_1 (R * 2)       // Radius of first ring in 1st galaxy
#define RING_BASE_2 (R * 2)       // Radius of first ring in 2nd galaxy
#define NUM_P_BASE 12             // Number of particles in the first ring
#define INC_NUM_P 3               // increment of number of particles each step
#define INC_R_RING (0.5 * R)      // increment of radius of rings each step
#define PMASS 0.1e-06             // mass of each particle
#define V_PARAMTER 0.8            // Parameter adding to initial velocity to make it elliptic

/**    Function Declarations Section    **/
__global__ void leapstep(int, double*, double*, double*, double*, double*, double*, double);
/*
 *  leapstep - kernel function: update positions using leapfrog algorithm
 *    parameters:
 *      [#particles] [x position] [y position] [x position]
 *                   [x velocity] [y velocity] [z velocity] [dt]
 *
 */
__global__ void accel(int, double*, double*, double*, double*, double*, double*, double*, double);
/*
 *  accel - kernel function: update velocity using leapfrog algorithm
 *    parameters:
 *      [#particles] [x position] [y position] [x position]
 *                   [x velocity] [y velocity] [z velocity]
 *                   [mass] [dt]
 *
 */
__global__ void printstate(int, double*, double*, double*, double*, double*, double*);
/*
 *  printstate - kernel function: print position and velocity
 *    parameters:
 *      [#particles] [x position] [y position] [x position]
 *                   [x velocity] [y velocity] [z velocity]
 *
 */
void printstate_host(int, double*, double*, double*, double*, double*, double*);
/*
 *  printstate_host - host function: print position and velocity
 *    parameters:
 *      [#particles] [x position] [y position] [x position]
 *                   [x velocity] [y velocity] [z velocity]
 *
 */
__global__ void initialConditions(int, double*, double*, double*, double*, double*, double*, double*);
/*
 *  initialConditions - kernel function: setup initial conditions
 *    parameters:
 *      [#particles] [x position] [y position] [x position]
 *                   [x velocity] [y velocity] [z velocity] [mass]
 *
 */
void initialCondition_host(int, double*, double*, double*, double*, double*, double*, double*);
/*
 *  initialCondition_host - host function: setup initial conditions
 *    parameters:
 *      [#particles] [x position] [y position] [x position]
 *                   [x velocity] [y velocity] [z velocity] [mass]
 *
 */


/**     Main function     **/
int main(int argc, char *argv[])
{
  /*
   *  Handling commandline inputs and setting initial value of the arguments
   *    1. number of steps (mstep)
   *    2. warp (nout)
   *    3. timestamp (dt)
   *
   */
  int mstep, nout;
  double dt, *x, *y, *z, *vx, *vy, *vz, *mass;
  mstep = argc > 1 ? atoi(argv[1]) : 100;
  nout = argc > 2 ? atoi(argv[2]) : 1;
  dt = argc > 3 ? atof(argv[3]) : 0.05;
  int n = (NUM_P_BASE + NUM_P_BASE + (NUM_OF_RING_1 - 1) * INC_NUM_P) * NUM_OF_RING_1 / 2 + (NUM_P_BASE + NUM_P_BASE + (NUM_OF_RING_2 - 1) * INC_NUM_P) * NUM_OF_RING_2 / 2 + 2;
  /*
   *  setup execution configuration
   */
  int numOfBlocks = n / BLOCKSIZE + (n % BLOCKSIZE != 0);
  int threads = BLOCKSIZE, grids = numOfBlocks;

  /*
  ** Allocate device memory for kernel functions
  **  May not need to allocate memory for host function
  **  because we print using kernel function
  */
  cudaMalloc((void**) &x, (size_t)(n * sizeof(double)));
  cudaMalloc((void**) &y, (size_t)(n * sizeof(double)));
  cudaMalloc((void**) &z, (size_t)(n * sizeof(double)));
  cudaMalloc((void**) &vx, (size_t)(n * sizeof(double)));
  cudaMalloc((void**) &vy, (size_t)(n * sizeof(double)));
  cudaMalloc((void**) &vz, (size_t)(n * sizeof(double)));
  cudaMalloc((void**) &mass, (size_t)(n * sizeof(double)));

  /*
   *  If MCORE is defined, use kernel function to setup
   *    initial conditions
   *  Otherwise, use host function to setup initial conditions
   *
   */
#ifdef MCORE
  initialConditions<<< grids, threads >>>(n, x, y, z, vx, vy, vz, mass);
  cudaDeviceSynchronize();
#else
  initialCondition_host(n, x, y, z, vx, vy, vz, mass);
#endif

  /*
   *  Use cudaDeviceSetLimit() to change the buffer size of printf
   *   used in kernel functions to solve the problem encountered before:
   *    cannot print more than 4096 lines of data using printf
   *
   */
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, n * BUFFERSIZE);

  /*  Start looping steps from first step to mstep  */
  for(int i = 0; i < mstep; i++){
    if(i % nout == 0)
      printstate<<< grids, threads >>>(n, x, y, z, vx, vy, vz);
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
    printstate<<< grids, threads >>>(n, x, y, z, vx, vy, vz);
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


void initialCondition_host(int n, double* x, double* y, double* z, double* vx, double* vy, double* vz, double* mass){
  srand(time(0));
  double *lx = (double*)malloc(n * sizeof(double));
  double *ly = (double*)malloc(n * sizeof(double));
  double *lz = (double*)malloc(n * sizeof(double));
  double *lvx = (double*)malloc(n * sizeof(double));
  double *lvy = (double*)malloc(n * sizeof(double));
  double *lvz = (double*)malloc(n * sizeof(double));
  double *lmass = (double*)malloc(n * sizeof(double));

  /*
   *  Setup mass of each particles (including center mass)
   *
   */
  lmass[0] = MASS_1;    // Set the mass of center mass of 1st galaxy
  for(int i = 1; i < NUM_OF_RING_1 + 1; i++){
    lmass[i] = PMASS;
  }
  lmass[NUM_OF_RING_1 + 1] = MASS_2;
  for(int i = NUM_OF_RING_1 + 2; i < n; i++){
    lmass[i] = PMASS;
  }

  /*
   *  Setup position of each particles
   *
   */
   lx[0] = (double)rand() / RAND_MAX;
   ly[0] = (double)rand() / RAND_MAX;
   lz[0] = (double)rand() / RAND_MAX;
   lvx[0] = lvy[0] = lvz[0] = 0.0;
   double cx = lx[0], cy = ly[0], cz = lz[0], cvx = lvx[0], cvy = lvy[0], cvz = lvz[0];
   double radius = RING_BASE_1;
   int count = 1;

   double norm, tmpx, tmpy, tmpz, normv, tmpvx, tmpvy, tmpvz;
   double a, c, sigma, sigma1;

   sigma = -2.0 * PI/3;
   sigma1 = -PI/3.0;
   c = cos(sigma1);
   s = sin(sigma1);
   a = 1 - cos(sigma);

   for(int i = 0; i < NUM_OF_RING_1; i++){
     int numOfP = NUM_P_BASE + INC_NUM_P * i;
     double piece = 2.0 * PI / numOfP;
     double velocity = sqrt(G * MASS_1 / radius);
     for(int j = 0; j < numOfP; j++){
       lx[count] = cx + radius * cos(piece * j);
       ly[count] = cy + radius * sin(piece * j);
       lz[count] = cz;
       lvx[count] = cvx - velocity * sin(piece * j) * V_PARAMTER;
       lvy[count] = cvy + velocity * cos(piece * j) * V_PARAMTER;
       lvz[count] = cvz;

       /*[vx' vy' vz'] = R [vx vy vz]*/
        norm = sqrt(lx[count] * lx[count] + ly[count] * ly[count] + lz[count] * lz[count])
        lx[count] /= norm;
        ly[count] /= norm;
        lz[count] /= norm;

        tmpx = ( a * lx[count] * lx[count] + c ) * lx[count] + ( a * lx[count] * ly[count] -sigma * lz[count]) * ly[count] + ( a * lx[count] * lz[count] + s * ly[count] ) * lz[count];
        tmpy = ( a * lx[count] * ly[count] + s * lz[count]) * lx[count] + ( a * ly[count] * ly[count] + c) * ly[count] + ( a * ly[count] * lz[count] - s * lx[count] ) * lz[count];
        tmpz = ( a * lx[count] * lz[count] - s * ly[count]) * lx[count] + ( a * ly[count] * lz[count] + s * lx[count]) * ly[count] + ( a * lz[count] * lz[count] + c) * lz[count];

        lx[count] = tmpx * norm;
        ly[count] = tmpy * norm;
        lz[count] = tmpz * norm;

        /*[vx' vy' vz'] = R [vx vy vz]*/
         normv = sqrt(lvx[count] * lvx[count] + lvy[count] * lvy[count] + lvz[count] * lvz[count])
         lvx[count] /= normv;
         lvy[count] /= normv;
         lvz[count] /= normv;

         tmpvx = ( a * lvx[count] * lvx[count] + c ) * lvx[count] + ( a * lvx[count] * lvy[count] -sigma * lvz[count]) * lvy[count] + ( a * lvx[count] * lvz[count] + s * lvy[count] ) * lvz[count];
         tmpvy = ( a * lvx[count] * lvy[count] + s * lvz[count]) * lvx[count] + ( a * lvy[count] * lvy[count] + c) * lvy[count] + ( a * lvy[count] * lvz[count] - s * lvx[count] ) * lvz[count];
         tmpvz = ( a * lvx[count] * lvz[count] - s * lvy[count]) * lvx[count] + ( a * lvy[count] * lvz[count] + s * lvx[count]) * lvy[count] + ( a * lvz[count] * lvz[count] + c) * lvz[count];

         lvx[count] = tmpvx * normv;
         lvy[count] = tmpvy * normv;
         lvz[count] = tmpvz * normv;


        count++;
     }
     radius += INC_R_RING;
   }


    sigma = 2.0 * PI/3;
    sigma1 = -PI/3.0;
    c = cos(sigma1);
    s = sin(sigma1);
    a = 1 - cos(sigma);

   lx[count] = lx[0] + radius * 3.0;
   ly[count] = ly[0] + radius * 4.0;
   lz[count] = lz[0];
   lvx[count] = lvy[count] = lvz[count] = 0.0;
   cx = lx[count];
   cy = ly[count];
   cz = lz[count];
   cvx = lvx[count];
   cvy = lvy[count];
   cvz = lvz[count];
   count++;
   radius = RING_BASE_2;
   for(int i = 0; i < NUM_OF_RING_2; i++){
     int numOfP = NUM_P_BASE + INC_NUM_P * i;
     double velocity = sqrt(G * MASS_2 / radius);
     double piece = 2.0 * PI / numOfP;
     for(int j = 0; j < numOfP; j++){
       lx[count] = cx + radius * cos(piece * j);
       ly[count] = cy + radius * sin(piece * j);
       lz[count] = cz;
       lvx[count] = cvx - velocity * sin(piece * j) * V_PARAMTER;
       lvy[count] = cvy + velocity * cos(piece * j) * V_PARAMTER;
       lvz[count] = cvz;

       norm = sqrt(lx[count] * lx[count] + ly[count] * ly[count] + lz[count] * lz[count])
       lx[count] /= norm;
       ly[count] /= norm;
       lz[count] /= norm;

       tmpx = ( a * lx[count] * lx[count] + c ) * lx[count] + ( a * lx[count] * ly[count] -sigma * lz[count]) * ly[count] + ( a * lx[count] * lz[count] + s * ly[count] ) * lz[count];
       tmpy = ( a * lx[count] * ly[count] + s * lz[count]) * lx[count] + ( a * ly[count] * ly[count] + c) * ly[count] + ( a * ly[count] * lz[count] - s * lx[count] ) * lz[count];
       tmpz = ( a * lx[count] * lz[count] - s * ly[count]) * lx[count] + ( a * ly[count] * lz[count] + s * lx[count]) * ly[count] + ( a * lz[count] * lz[count] + c) * lz[count];

       lx[count] = tmpx * norm;
       ly[count] = tmpy * norm;
       lz[count] = tmpz * norm;

       /*[vx' vy' vz'] = R [vx vy vz]*/
        normv = sqrt(lvx[count] * lvx[count] + lvy[count] * lvy[count] + lvz[count] * lvz[count])
        lvx[count] /= normv;
        lvy[count] /= normv;
        lvz[count] /= normv;

        tmpvx = ( a * lvx[count] * lvx[count] + c ) * lvx[count] + ( a * lvx[count] * lvy[count] -sigma * lvz[count]) * lvy[count] + ( a * lvx[count] * lvz[count] + s * lvy[count] ) * lvz[count];
        tmpvy = ( a * lvx[count] * lvy[count] + s * lvz[count]) * lvx[count] + ( a * lvy[count] * lvy[count] + c) * lvy[count] + ( a * lvy[count] * lvz[count] - s * lvx[count] ) * lvz[count];
        tmpvz = ( a * lvx[count] * lvz[count] - s * lvy[count]) * lvx[count] + ( a * lvy[count] * lvz[count] + s * lvx[count]) * lvy[count] + ( a * lvz[count] * lvz[count] + c) * lvz[count];

        lvx[count] = tmpvx * normv;
        lvy[count] = tmpvy * normv;
        lvz[count] = tmpvz * normv;

       count++;
     }
     radius += INC_R_RING;
   }


  cudaMemcpy(x, lx, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y, ly, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(z, lz, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vx, lvx, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vy, lvy, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(vz, lvz, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mass, lmass, (size_t) n * sizeof(double), cudaMemcpyHostToDevice);
  free(lx);
  free(ly);
  free(lz);
  free(lvx);
  free(lvy);
  free(lvz);
  free(lmass);
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
          ax += - G * mass[i * BLOCKSIZE + j] * (thisX - lx[j]) / norm;
          ay += - G * mass[i * BLOCKSIZE + j] * (thisY - ly[j]) / norm;
          az += - G * mass[i * BLOCKSIZE + j] * (thisZ - lz[j]) / norm;
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
