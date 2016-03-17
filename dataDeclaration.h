#ifndef CONSTANTDECLARATION
#define CONSTANTDECLARATION
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
#define MASS_1 19.5              // Center mass of 1st galaxy
#define MASS_2 19.5                // Center mass of 2nd galaxy
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
#define RMIN (7.733/25)
#define ECCEN 0.5
#define RMAX ((1.0 + ECCEN) * RMIN / (1.0 - ECCEN))
#define RING_BASE_1 (RMIN * 0.2)       // Radius of first ring in 1st galaxy
#define RING_BASE_2 (RMIN * 0.2)       // Radius of first ring in 2nd galaxy
#define INC_R_RING (RMIN * 0.05)      // increment of radius of rings each step
#define SOFTPARAMETER 0.000001

#endif
