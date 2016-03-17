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
#define AndromedaXOffsetV 0.0420
#define AndromedaYOffsetV -0.2504
#define AndromedaZOffsetV 0.1240
#define MilkwayXOffsetP 41.0882
#define MilkwayYOffsetP -68.3823
#define MilkwayZOffsetP 33.8634
#define MilkwayXOffsetV -0.0420
#define MilkwayYOffsetV 0.2504
#define MilkwayZOffsetV -0.1240

#endif
