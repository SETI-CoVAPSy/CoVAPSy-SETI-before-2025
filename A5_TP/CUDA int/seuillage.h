
//#define SEUILLAGE_H
//#ifndef SEUILLAGE_H

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>



#define SIZE_I {{SIZE_I}}
#define SIZE_J {{SIZE_J}}

#define BLOCK_SIZE {{BLOCK_SIZE}}

// Prototype
void runTest( int argc, char** argv);
extern "C" void seuillage_C( uint8_t reference[][SIZE_J][SIZE_I] , uint8_t idata[][SIZE_J][SIZE_I] );


//#endif



