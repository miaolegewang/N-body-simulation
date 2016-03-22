#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/** Constant definition **/
#define PI 3.14159265
#define BUFFERSIZE 500
#ifndef BLOCKSIZE
#define BLOCKSIZE 256
#endif
#define G 1.0
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
#define NUMOFP 88000
#define DT 0.2
#define LEAPSTEPSIZE 1.0
#define MilkyWayTranslate 1


/**   Particle structure    **/
typedef struct particle{
  double x;
  double y;
  double z;
  double vx;
  double vy;
  double vz;
  double mass;
} particle;

typedef struct rotationMatrix{
  double n1, n2, n3;
  double theta;
} rotationMatrix;

/** Major function declaration **/
unsigned long create_galaxy_from_file(char *file, particle* p, unsigned long offset);
__global__ void accel(unsigned long n, particle* p, double dt);
__global__ void leapstep(unsigned long n, particle* p, double dt);
__global__ void printstate(unsigned long n, particle* p, unsigned long tnow);

/** Helper function declaration **/
void rotate(particle* p, double n1, rotationMatrix m1, rotationMatrix m2);
void translate(particle* p, int galaxy);
rotationMatrix make_rmatrix(double n1, double n2, double n3, double theta);

int main(int argc, char *argv[]){
  /* Input args:
   *  1. mstep
   *  2. nout
   *  3. offset
   *  4. dt
   *
   */
   unsigned long mstep, nout, offset, tnow = 0, n = NUMOFP, numOfBlocks, offset_file_read;
   double dt;
   particle* p;

   mstep = (argc > 1) ? atoi(argv[1]) : 1200;
   nout = (argc > 2) ? atoi(argv[2]) : 1;
   offset = (argc > 3) ? atoi(argv[3]) : 0;
   dt = (argc > 4) ? atof(argv[4]) : DT;
   numOfBlocks = ceil((double)n / BLOCKSIZE);

   cudaMalloc(p, numOfBlocks * BLOCKSIZE * sizeof(particle));
   offset_file_read = create_galaxy_from_file("milky_way.dat", p, 0);
   create_galaxy_from_file("andromeda.dat", p, tnow, offset_file_read);
   cudaMemset(p + n, 0, (size_t)(numOfBlocks * BLOCKSIZE - n) * sizeof(particle));
   printstate(n, p, 0);

   cudaFree(p);
}

/** Function implementation **/
/* Helper functions */
void rotate(particle* p, double n1, rotationMatrix m1, rotationMatrix m2){
  double sigma, c, s, a, tx, ty, tz, n1, n2, n3;
  /*  rotate position  */
  sigma = - m1.theta;
  c = cos(sigma);
  s = sin(sigma);
  a = 1 - cos(sigma);
  n1 = m1.n1;
  n2 = m1.n2;
  n3 = m1.n3;
  tx = ( a * n1 * n1 + c ) * (p->x) + ( a * n1 * n2 - s * n3 ) * (p->y) + ( a * n1 * n3 + s * n2 ) * (p->z);
  ty = ( a * n1 * n2 + s * n3 ) * (p->x) + ( a * n2 * n2 + c ) * (p->y) + ( a * n2 * n3 - s * n1 ) * (p->z);
  tz = ( a * n1 * n3 - s * n2 ) * (p->x) + ( a * n2 * n3 + s * n1 ) * (p->y) + ( a * n3 * n3 + c ) * (p->z);
  p->x = tx;
  p->y = ty;
  p->z = tz;
  /*  rotate velocity */
  sigma = - m2.theta;
  c = cos(sigma);
  s = sin(sigma);
  a = 1 - cos(sigma);
  n1 = m2.n1;
  n2 = m2.n2;
  n3 = m2.n3;
  tx = ( a * n1 * n1 + c ) * (p->vx) + ( a * n1 * n2 - s * n3 ) * (p->vy) + ( a * n1 * n3 + s * n2 ) * (p->vz);
  ty = ( a * n1 * n2 + s * n3 ) * (p->vx) + ( a * n2 * n2 + c ) * (p->vy) + ( a * n2 * n3 - s * n1 ) * (p->vz);
  tz = ( a * n1 * n3 - s * n2 ) * (p->vx) + ( a * n2 * n3 + s * n1 ) * (p->vy) + ( a * n3 * n3 + c ) * (p->vz);
  p->vx = tx;
  p->vy = ty;
  p->vz = tz;
}

void translate(particle* p, int galaxy){
  if(galaxy == MilkyWayTranslate){
    p->x += MilkwayXOffsetP;
    p->y += MilkwayYOffsetP;
    p->z += MilkwayZOffsetP;
    p->vx += MilkwayXOffsetV;
    p->vy += MilkwayYOffsetV;
    p->vz += MilkwayZOffsetV;
  } else {
    p->x += AndromedaXOffsetP;
    p->y += AndromedaYOffsetP;
    p->z += AndromedaZOffsetP;
    p->vx += AndromedaXOffsetV;
    p->vy += AndromedaYOffsetV;
    p->vz += AndromedaZOffsetV;
  }

}

/* Major functions  */
unsigned long create_galaxy_from_file(char *file, particle* p, unsigned long offset){
  FILE *f = fopen(file, 'r');
  unsigned long size, counter = 0, numOfBlocks = ceil(NUMOFP / BLOCKSIZE);
  double junk;
  fscanf(f, "%lu %lf\n", &size, &junk);
  particle* lp = malloc(size * sizeof(double));
  while(!feof(f) && counter < size){
    fscanf(f, "%lf %lf %lf %lf %lf %lf %lf\n", &(p[counter].mass), &(p[counter].x), &(p[counter].y), &(p[counter].z),&(p[counter].vx), &(p[counter].vy, &(p[counter].vz)));
    counter++;
  }
  fclose(f);
  cudaMemcpy(p, lp, (size_t)size * sizeof(particle), cudaMemcpyHostToDevice);
  free(lp);
  return size;
}

__global__ void accel(unsigned long n, particle* p, double dt){
  unsigned long id = blockIdx.x * blockDim.x + threadIdx.x, tdx = threadIdx.x;
  double thisX = p[id].x, thisY = p[id].y, thisZ = p[id].z, ax = 0.0, ay = 0.0, az = 0.0, xdist, ydist, zdist, norm;
  __shared__ double x[BLOCKSIZE];
  __shared__ double y[BLOCKSIZE];
  __shared__ double z[BLOCKSIZE];
  __shared__ double m[BLOCKSIZE];
  for(unsigned long i = 0; i < gridDim.x; i++){
    x[tdx] = p[i * BLOCKSIZE + tdx].x;
    y[tdx] = p[i * BLOCKSIZE + tdx].y;
    z[tdx] = p[i * BLOCKSIZE + tdx].z;
    m[tdx] = p[i * BLOCKSIZE + tdx].mass;
    __syncthreads();
    for(unsigned long j = 0; j < BLOCKSIZE; j++){
      xdist = thisX - x[j];
      ydist = thisY - y[j];
      zdist = thisZ - z[j];
      norm = rsqrt(SOFTPARAMETER + xdist * xdist + ydist * ydist + zdist * zdist);
      ax += G * m[j] * (x[j] - thisX) * norm * norm * norm;
      ay += G * m[j] * (y[j] - thisY) * norm * norm * norm;
      az += G * m[j] * (z[j] - thisZ) * norm * norm * norm;
    }
    __syncthreads();
  }
  if(id < n){
    p[id].vx += LEAPSTEPSIZE * dt * ax;
    p[id].vy += LEAPSTEPSIZE * dt * ay;
    p[id].vz += LEAPSTEPSIZE * dt * az;
  }
}

__global__ void printstate(unsigned long n, particle* p, unsigned long tnow){
  unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < 10000 || (i >= 44000 && i < 54000))
    printf("%lu, %f, %f, %f, %f, %d\n", id, p[id].x, p[id].y, p[id].z, tnow);
}
