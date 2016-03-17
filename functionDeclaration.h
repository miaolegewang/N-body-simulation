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

__global__ void printstate(int, double*, double*, double*, double*, double*, double*, int);
/*
 *  printstate - kernel function: print position and velocity
 *    parameters:
 *      [#particles] [x position] [y position] [x position]
 *                   [x velocity] [y velocity] [z velocity] [tnow]
 *
 */

void initialCondition_host_file(char *, char *, double **, double **, double **, double **, double **, double **, double **, int *);
/*
 *  initialCondition_host_file - host function: create both galaxies by extracting data from files
 *    parameters:
 *      [filename galaxy 1] [filename galaxy 2]
 *                 [x position array addr] [y position array addr] [z position array addr]
 *                 [x velocity array addr] [y velocity array addr] [z velocity array addr]
 *                 [mass array addr] [size addr]
 *
 */
void create_galaxy_file(FILE *, double*, double*, double*, double*, double*, double*, double*);
/*
 *  create_galaxy_file - host function: create galaxy by reading data from file
 *    parameters:
 *      [filename galaxy] [x position] [y position] [z position]
 *                        [x velocity] [y velocity] [z velocity]
 *                        [mass]
 *
 */
void initialCondition_host_file(char *, char *, double **, double **, double **, double **, double **, double **, double **, int *);
/*
 *  initialCondition_host_file - host function: create both galaxies by extracting data from files
 *    parameters:
 *      [filename galaxy 1] [filename galaxy 2]
 *                 [x position array addr] [y position array addr] [z position array addr]
 *                 [x velocity array addr] [y velocity array addr] [z velocity array addr]
 *                 [mass array addr] [size addr]
 *
 */
void create_galaxy_file(FILE *, double*, double*, double*, double*, double*, double*, double*);
/*
 *  create_galaxy_file - host function: create galaxy by reading data from file
 *    parameters:
 *      [filename galaxy] [x position] [y position] [z position]
 *                        [x velocity] [y velocity] [z velocity]
 *                        [mass]
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

/*
 *  Helper Function Declarations
 *
 */
void rotate(double*, double*, double*, double, double, double, double);
void read_size_from_file(FILE*, int*, double*);
