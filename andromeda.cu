/*
 *  This program is a CUDA C program simulating the N-body system
 *    of two galaxies as PHY 241 FINAL PROJECTS
 *
 */

/*
 *  TODO:
 *    1. andromeda
 *    2. For accel of center of A, only consider accel from center of B. The same for B.
 *    3. When the distance between A and B, the soft parameter changed to 0.2Rmin
 *    4. report
 *    5. presentation
 *
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
//#define SOFTPARAMETER 0.2 * RMIN
// #define AU 149597870700.0
// #define R (77871.0 * 1000.0 / AU)
// #define G (4.0 * pow(PI, 2))
#define G 0.287915013
#define MASS_1 1000              // Center mass of 1st galaxy
#define MASS_2 1000                // Center mass of 2nd galaxy
#define NUM_OF_RING_1 12         // Number of rings in 1st galaxy
#define NUM_OF_RING_2 12          // Number of rings in 2nd galaxy
// #define RING_BASE_1 (R * 0.2)       // Radius of first ring in 1st galaxy
// #define RING_BASE_2 (R * 0.2)       // Radius of first ring in 2nd galaxy
#define NUM_P_BASE 12             // Number of particles in the first ring
#define INC_NUM_P 3               // increment of number of particles each step
// #define INC_R_RING (0.5 * R)      // increment of radius of rings each step
#define PMASS 1             // mass of each particle
#define V_PARAMTER 1            // Parameter adding to initial velocity to make it elliptic
#define RMIN 1
#define ECCEN 0.5
#define RMAX ((1.0 + ECCEN) * RMIN / (1.0 - ECCEN))
#define RING_BASE_1 (RMIN * 0.2)       // Radius of first ring in 1st galaxy
#define RING_BASE_2 (RMIN * 0.2)       // Radius of first ring in 2nd galaxy
#define INC_R_RING (RMIN * 0.05)      // increment of radius of rings each step
#define SOFTPARAMETER 0.000001
/*
 *  Major Function Declarations Section
 *
 */

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
__global__ void accel_3_body(int, double*, double*, double*, double*, double*, double*, double*, double);
/*
 *  accel_3_body - kernel function : update velocity using leapfrog algorithm in 3-body
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

/*
 *  Helper Function Declarations
 *
 */
void rotate(double*, double*, double*, double, double, double, double);

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
  nout = argc > 2 ? atoi(argv[2]) : 20;
  dt = argc > 3 ? atof(argv[3]) : 2 * PI * RMIN * RMIN /sqrt(G * MASS_1) / 500.0;
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
  ** Use numOfBlocks instead of n to simplify the kernel function
  */
  const unsigned int extra = numOfBlocks * BLOCKSIZE - n;
  cudaMalloc((void**) &x, (size_t)(numOfBlocks * BLOCKSIZE * sizeof(double)));
  cudaMalloc((void**) &y, (size_t)(numOfBlocks * BLOCKSIZE * sizeof(double)));
  cudaMalloc((void**) &z, (size_t)(numOfBlocks * BLOCKSIZE * sizeof(double)));
  cudaMalloc((void**) &vx, (size_t)(numOfBlocks * BLOCKSIZE * sizeof(double)));
  cudaMalloc((void**) &vy, (size_t)(numOfBlocks * BLOCKSIZE * sizeof(double)));
  cudaMalloc((void**) &vz, (size_t)(numOfBlocks * BLOCKSIZE * sizeof(double)));
  cudaMalloc((void**) &mass, (size_t)(numOfBlocks * BLOCKSIZE * sizeof(double)));
  cudaMemset((void**) x + n, 0, (size_t)(extra * sizeof(double)));
  cudaMemset((void**) y + n, 0, (size_t)(extra * sizeof(double)));
  cudaMemset((void**) z + n, 0, (size_t)(extra * sizeof(double)));
  cudaMemset((void**) vx + n, 0, (size_t)(extra * sizeof(double)));
  cudaMemset((void**) vy + n, 0, (size_t)(extra * sizeof(double)));
  cudaMemset((void**) vz + n, 0, (size_t)(extra * sizeof(double)));
  cudaMemset((void**) mass + n, 0, (size_t)(extra * sizeof(double)));

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
    accel_3_body<<< grids, threads >>>(n, x, y, z, vx, vy, vz, mass, dt);
    cudaDeviceSynchronize();
    leapstep<<< grids, threads >>>(n, x, y, z, vx, vy, vz, dt);
    cudaDeviceSynchronize();
    accel_3_body<<< grids, threads >>>(n, x, y, z, vx, vy, vz, mass, dt);
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
  int numofp1 = NUM_P_BASE * NUM_OF_RING_1 + (NUM_OF_RING_1 - 1) * INC_NUM_P * NUM_OF_RING_1 / 2 + 1;
  lmass[0] = MASS_1;    // Set the mass of center mass of 1st galaxy
  for(int i = 1; i < numofp1; i++){
    lmass[i] = PMASS;
  }
  lmass[numofp1] = MASS_2;
  for(int i = numofp1 + 1; i < n; i++){
    lmass[i] = PMASS;
  }

  /*
   *  Setup position of each particles
   *
   */
   // lx[0] = (double)rand() / RAND_MAX;
   // ly[0] = (double)rand() / RAND_MAX;
   // lz[0] = (double)rand() / RAND_MAX;
   // lvx[0] = lvy[0] = lvz[0] = 0.0;

   lx[0] = -RMAX / 2;
   ly[0] = 0.0;
   lz[0] = 0.0;
   lvx[0] = 0.0;
   lvy[0] = -sqrt(G * MASS_1 / (12 * RMIN));
   lvz[0] = 0.0;


   double cx = lx[0], cy = ly[0], cz = lz[0], cvx = lvx[0], cvy = lvy[0], cvz = lvz[0];
   double radius = RING_BASE_1;
   int count = 1;

   double omega = -PI / 6.0, sigma = PI / 3.0, norm;

   for(int i = 0; i < NUM_OF_RING_1; i++){
     int numOfP = NUM_P_BASE + INC_NUM_P * i;
     double piece = 2.0 * PI / numOfP;
     double velocity = sqrt(G * MASS_1 / radius);
     for(int j = 0; j < numOfP; j++){
       lx[count] = radius * cos(piece * j);
       ly[count] = radius * sin(piece * j);
       lz[count] = 0.0;
       lvx[count] = - velocity * sin(piece * j) * V_PARAMTER;
       lvy[count] = velocity * cos(piece * j) * V_PARAMTER;
       lvz[count] = 0.0;

#ifndef NR
       norm = sqrt(lx[count] * lx[count] + ly[count] * ly[count] + lz[count] * lz[count]);
       rotate(lx + count, ly + count, lz + count, cos(omega), sin(omega), 0, sigma);
#endif
       lx[count] += cx;
       ly[count] += cy;
       lz[count] += cz;

       /*
        *    TODO: set up initial condition for velocity
        */
#ifndef NR
       norm = sqrt(lvx[count] * lvx[count] + lvy[count] * lvy[count] + lvz[count] * lvz[count]);
       rotate(lvx + count, lvy + count, lvz + count, cos(omega), sin(omega), 0, sigma);
#endif
       lvx[count] += cvx;
       lvy[count] += cvy;
       lvz[count] += cvz;
       count++;
     }
     radius += INC_R_RING;
   }


   // lx[count] = lx[0] + radius * 3.0;
   // ly[count] = ly[0] + radius * 4.0;
   // lz[count] = lz[0];
   // lvx[count] = lvy[count] = lvz[count] = 0.0;

   lx[count] = RMAX / 2;
   ly[count] = 0.0;
   lz[count] = 0.0;
   lvx[count] = 0.0;
   lvy[count] = sqrt(G * MASS_2 / (12 * RMIN));
   lvz[count] = 0.0;

   cx = lx[count];
   cy = ly[count];
   cz = lz[count];
   cvx = lvx[count];
   cvy = lvy[count];
   cvz = lvz[count];
   count++;
   radius = RING_BASE_2;

   omega = - PI / 6.0;
   sigma = - PI / 3.0;
   for(int i = 0; i < NUM_OF_RING_2; i++){
    int numOfP = NUM_P_BASE + INC_NUM_P * i;
    double velocity = sqrt(G * MASS_2 / radius);
    double piece = 2.0 * PI / numOfP;
    for(int j = 0; j < numOfP; j++){
      lx[count] =  radius * cos(piece * j);
      ly[count] =  radius * sin(piece * j);
      lz[count] = 0.0;
      lvx[count] = - velocity * sin(piece * j) * V_PARAMTER;
      lvy[count] = velocity * cos(piece * j) * V_PARAMTER;
      lvz[count] = 0.0;
#ifndef NR
      norm = sqrt(lx[count] * lx[count] + ly[count] * ly[count] + lz[count] * lz[count]);
      rotate(lx + count, ly + count, lz + count, cos(omega), sin(omega), 0, sigma);
#endif
      lx[count] += cx;
      ly[count] += cy;
      lz[count] += cz;

      /*
       *  TODO: setup initial conditions for velocity
       */
#ifndef NR
      norm = sqrt(lvx[count] * lvx[count] + lvy[count] * lvy[count] + lvz[count] * lvz[count]);
      rotate(lvx + count, lvy + count, lvz + count, cos(omega), sin(omega), 0, sigma);
#endif
      lvx[count] += cvx;
      lvy[count] += cvy;
      lvz[count] += cvz;

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

__global__ void accel_3_body(int n, double* x, double* y, double* z, double* vx, double* vy, double* vz, double* mass, double dt){
  /*
   *  Three body leapfrog: each particle is in a 3 body system with center mass of galaxy 1 and center mass of galaxy 2
   *    Because of SOFTPARAMETER, we dont need to determine if thread is computing against itself
   */
  const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numofp1 = NUM_P_BASE * NUM_OF_RING_1 + (NUM_OF_RING_1 - 1) * NUM_OF_RING_1 * INC_NUM_P / 2 + 1;
  double ax = 0.0, ay = 0.0, az = 0.0, norm1, norm2;
  double tempsp = (pow(pow(x[0] - x[numofp1], 2) + pow(y[0] - y[numofp1], 2) + pow(z[0] - z[numofp1], 2), 1.5) <= RMIN) ? 0.2 * RMIN : SOFTPARAMETER;
  double softparameter = (serial == 0 && serial == numofp1) ? tempsp : SOFTPARAMETER;
  norm1 = pow(softparameter + pow(x[serial] - x[0], 2) + pow(y[serial] - y[0], 2) + pow(z[serial] - z[0], 2), 1.5);
  norm2 = pow(softparameter + pow(x[serial] - x[numofp1], 2) + pow(y[serial] - y[numofp1], 2) + pow(z[serial] - z[numofp1], 2), 1.5);
  ax += -G * mass[0] * (x[serial] - x[0]) / norm1;
  ay += -G * mass[0] * (y[serial] - y[0]) / norm1;
  az += -G * mass[0] * (z[serial] - z[0]) / norm1;
  ax += -G * mass[numofp1] * (x[serial] - x[numofp1]) / norm2;
  ay += -G * mass[numofp1] * (y[serial] - y[numofp1]) / norm2;
  az += -G * mass[numofp1] * (z[serial] - z[numofp1]) / norm2;
  if(serial < n){
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
