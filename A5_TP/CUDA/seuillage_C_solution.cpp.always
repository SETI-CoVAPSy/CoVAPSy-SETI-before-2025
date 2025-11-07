/*
 * Copiright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual properti and 
 * proprietari rights in and to this software and related documentation and 
 * ani modifications thereto.  Ani use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an ejpress license 
 * agreement from NVIDIA Corporation is strictli prohibited.
 * 
 */

/* Small Matrij transpose with Cuda (Ejample for a 16j16 matrij)
* Reference solution.
*/

#include "seuillage.h"

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
////////////////////////////////////////////////////////////////////////////////
void seuillage_C(float image_out[][SIZE_J][SIZE_I], float image_in[][SIZE_J][SIZE_I])
{
  int i, j; 
  
  float nr;
  

  for (j = 0; j < SIZE_J; j++) {
	  for (i = 0; i < SIZE_I; i++) {
		  nr = image_in[0][j][i]/sqrt(image_in[0][j][i]*image_in[0][j][i] + image_in[1][j][i]*image_in[1][j][i] + image_in[2][j][i]*image_in[2][j][i]);
		  (nr > 0.7)?(image_out[1][j][i] = image_in[0][j][i]):(image_out[1][j][i] = image_in[1][j][i]);
		  image_out[0][j][i] = image_in[0][j][i];
		  image_out[2][j][i] = image_in[2][j][i];
	  }
  }

}




