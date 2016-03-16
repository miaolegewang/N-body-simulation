__global__ void accel_3_body(int, double*, double*, double*, double*, double*, double*, double*, double);
/*
 *  accel_3_body - kernel function : update velocity using leapfrog algorithm in 3-body
 *    parameters:
 *      [#particles] [x position] [y position] [x position]
 *                   [x velocity] [y velocity] [z velocity]
 *                   [mass] [dt]
 *
 */
 __global__ void accel_3_body_relative(int, double*, double*, double*, double*, double*, double*, double*, double);
 /*
  *  accel_3_body_relative - kernel function : update velocity using leapfrog algorithm in 3-body setting Milky Way at the origin
  *    parameters:
  *      [#particles] [x position] [y position] [x position]
  *                   [x velocity] [y velocity] [z velocity]
  *                   [mass] [dt]
  *
  */
void printstate_host(int, double*, double*, double*, double*, double*, double*, int);
/*
 *  printstate_host - host function: print position and velocity
 *    parameters:
 *      [#particles] [x position] [y position] [x position]
 *                   [x velocity] [y velocity] [z velocity] [tnow]
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

void printstate_host(int n, double *x, double *y, double *z, double *vx, double *vy, double *vz, int tnow){
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
    printf("%d, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %d\n", i, lx[i], ly[i], lz[i], lvx[i], lvy[i], lvz[i], tnow);
  }
  free(lx);
  free(ly);
  free(lz);
  free(lvx);
  free(lvy);
  free(lvz);
}

__global__ void accel_3_body(int n, double* x, double* y, double* z, double* vx, double* vy, double* vz, double* mass, double dt){
  /*
   *  Three body leapfrog: each particle is in a 3 body system with center mass of galaxy 1 and center mass of galaxy 2
   *    Because of SOFTPARAMETER, we dont need to determine if thread is computing against itself
   */
  const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numofp1 = NUM_P_BASE * NUM_OF_RING_1 + (NUM_OF_RING_1 - 1) * NUM_OF_RING_1 * INC_NUM_P / 2 + 1;
  double ax = 0.0, ay = 0.0, az = 0.0, norm1, norm2;
#ifdef SOFTPARA
  double tempsp = (pow(pow(x[0] - x[numofp1], 2) + pow(y[0] - y[numofp1], 2) + pow(z[0] - z[numofp1], 2), 1.5) <= RMIN) ? 0.2 * RMIN : SOFTPARAMETER;
  double softparameter = (serial == 0 && serial == numofp1) ? tempsp : SOFTPARAMETER;
#else
  double softparameter = 0.0;
#endif
  norm1 = pow(softparameter + pow(x[serial] - x[0], 2) + pow(y[serial] - y[0], 2) + pow(z[serial] - z[0], 2), 1.5);
  norm2 = pow(softparameter + pow(x[serial] - x[numofp1], 2) + pow(y[serial] - y[numofp1], 2) + pow(z[serial] - z[numofp1], 2), 1.5);
  if(serial != 0){
    ax += -G * mass[0] * (x[serial] - x[0]) / norm1;
    ay += -G * mass[0] * (y[serial] - y[0]) / norm1;
    az += -G * mass[0] * (z[serial] - z[0]) / norm1;
  }
  if(serial != numofp1){
    ax += -G * mass[numofp1] * (x[serial] - x[numofp1]) / norm2;
    ay += -G * mass[numofp1] * (y[serial] - y[numofp1]) / norm2;
    az += -G * mass[numofp1] * (z[serial] - z[numofp1]) / norm2;
  }
  if(serial < n){
    vx[serial] += 0.5 * dt * ax;
    vy[serial] += 0.5 * dt * ay;
    vz[serial] += 0.5 * dt * az;
  }
}

__global__ void accel_3_body_relative(int n, double* x, double* y, double* z, double* vx, double* vy, double* vz, double* mass, double dt){
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
  if(serial < n && serial != 0){  // Not updating Mily Way
    vx[serial] += 0.5 * dt * ax;
    vy[serial] += 0.5 * dt * ay;
    vz[serial] += 0.5 * dt * az;
  }
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
  lmass[0] =   MASS_1;    // Set the mass of center mass of 1st galaxy
  for(int i = 1; i < numofp1; i++){
    lmass[i] = PMASS;
  }
  lmass[numofp1] =  MASS_2;
  for(int i = numofp1 + 1; i < n; i++){
    lmass[i] = PMASS;
  }

  /*
   *  Setup position of each particles
   *    First galaxy is M31
   *    Secon galaxy is Milky way
   *
   */

   // Milky Way setup
   lx[0] = 9.8165;
   ly[0] = -16.7203;
   lz[0] = 8.2814;
   lvx[0] = -0.1248;
   lvy[0] = 0.1057;
   lvz[0] = -0.0523;


   double cx = lx[0], cy = ly[0], cz = lz[0], cvx = lvx[0], cvy = lvy[0], cvz = lvz[0];
   double radius = RING_BASE_1;
   int count = 1;

   // omega = 240, sigma = -i
   double omega = 0.0, sigma = PI / 2.0, norm;

   for(int i = 0; i < NUM_OF_RING_1; i++){
     int numOfP = NUM_P_BASE + INC_NUM_P * i;
     double piece = 2.0 * PI / numOfP;
     double velocity = sqrt(G * lmass[0] / radius);
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

   // M31 way setup
   lx[numofp1] = -4.6355;
   ly[numofp1] = 7.8957;
   lz[numofp1] = -3.9106;
   lvx[numofp1] = 0.0589;
   lvy[numofp1] = -0.0499;
   lvz[numofp1] = 0.0247;

   cx = lx[count];
   cy = ly[count];
   cz = lz[count];
   cvx = lvx[count];
   cvy = lvy[count];
   cvz = lvz[count];
   count++;
   radius = RING_BASE_2;

   omega = -2 * PI / 3;
   sigma = PI / 6;
   for(int i = 0; i < NUM_OF_RING_2; i++){
    int numOfP = NUM_P_BASE + INC_NUM_P * i;
    double velocity = sqrt(G * lmass[numofp1] / radius);
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
