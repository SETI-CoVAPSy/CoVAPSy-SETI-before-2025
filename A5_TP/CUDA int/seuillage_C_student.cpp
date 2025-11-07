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
void seuillage_C(uint8_t image_out[][SIZE_J][SIZE_I], uint8_t image_in[][SIZE_J][SIZE_I]) 
{
    for(int i = 0; i < SIZE_I ; i++){
        for(int j = 0; j < SIZE_J; j++){
            // Extract the RGB values
            uint8_t r = image_in[0][i][j]; // Red channel
            uint8_t g = image_in[1][i][j]; // Green channel
            uint8_t b = image_in[2][i][j]; // Blue channel

            // Calculate the intensity I
            float denominator = sqrt(r * r + g * g + b * b);
            float I = 0.0f; // Initialize I

            // Avoid division by zero
            if (denominator > 0.0f) {
                I = r / denominator;
            }

            // Create the red mask
            if (I > 0.7) {
                // If the mask condition is met, modify the output image
                image_out[0][i][j] = r; // Red channel remains the same
                image_out[1][i][j] = r; // Green channel gets red value
                image_out[2][i][j] = b; // Blue channel remains the same
            } else {
                // If the mask condition is not met, keep the original values
                image_out[0][i][j] = r;
                image_out[1][i][j] = g;
                image_out[2][i][j] = b;
            }
        }
    }

}




