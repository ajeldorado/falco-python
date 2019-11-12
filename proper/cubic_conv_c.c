#include <stdio.h>
#include <math.h>

//#include "idl_export.h"

/* based on the traditional separable derivation described in "Two-Dimensional Cubic Convolution" by
 * Reichenbach & Geng, IEEE Transactions on Image Processing, v.12, 857 (2003).  This code is intended
 * to provide the same results as the CUBIC=-0.5 option to IDL's INTERPOLATE function.  */
 
/*--------------------------------------------------------------------------------------*/
double roundval( double x )
{
	if ( x > 0.0 ) 
		return( floor(x+0.5) );
	else
		return( -floor(-x+0.5) );
}

/*--------------------------------------------------------------------------------------*/
//void  cubic_conv_c( int argc, void *argv[] )
void cubic_conv_c(const void * image_input, int size_in_x, int size_in_y, void * image_output, int size_out_x, int size_out_y, void * x_input, void * y_input)
{
	double  a, k, d, sum, xc, yc, kx0[5], kx1[5], ky0[5], ky1[5];
	int	i, j, xoff, yoff, x_pix, x_out, y_pix, y_out;
	size_t	ii;

	const double * image_in = (double *)image_input;
	double * image_out = (double *)image_output;
	double * x_in = (double *)x_input;	/* vector of X coordinates in input image */
	double * y_in = (double *)y_input;	/* vector of Y coordinates in input image */
	a = -0.5;

	for ( y_out = 0; y_out < size_out_y; ++y_out )
	{
		for ( x_out = 0; x_out < size_out_x; ++x_out )
		{
			ii = y_out * (size_t)size_out_x + x_out;

	        	if ( y_in[ii] < 0 || y_in[ii] >= size_in_y )
	           		continue;
		    	if ( x_in[ii] < 0 || x_in[ii] >= size_in_x )
		       		continue;

			y_pix = roundval( y_in[y_out] );
			yc = roundval(y_in[y_out]) - y_in[y_out];

			for ( j = -2; j <= 2; ++j )
			{
				d = fabs(yc + j);
				if ( d <= 1 )
				{
					ky0[j+2] = d*d*(2*d - 3) + 1;
					ky1[j+2] = d*d*(d - 1);
				}
				else if ( d <= 2 )
				{
					ky0[j+2] = 0;
					ky1[j+2] = d*(d*d - 5*d + 8) - 4;
				}
				else
				{
					ky0[j+2] = 0;
					ky1[j+2] = 0;
				}
			}
			 
		    	x_pix = roundval( x_in[x_out] );
		    	xc = roundval(x_in[x_out]) - x_in[x_out];

		    	for ( j = -2; j <= 2; ++j )
		    	{
				d = fabs(xc + j);
				if ( d <= 1 )
				{
			   		kx0[j+2] = d*d*(2*d - 3) + 1;
			   		kx1[j+2] = d*d*(d - 1);
				}
				else if ( d <= 2 )
				{
			   		kx0[j+2] = 0;
			   		kx1[j+2] = d*(d*d - 5*d + 8) - 4;
				}
				else
				{
			   		kx0[j+2] = 0;
			   		kx1[j+2] = 0;
				}
		    	}

		    	sum = 0;
		    	for ( j = 0; j <= 4; ++j )
		    	{
				yoff = y_pix + j - 2;
				if ( yoff < 0 || yoff >= size_in_y )
			     		continue;
			     
				for ( i = 0; i <= 4; ++i )
				{
			     		xoff = x_pix + i - 2;
			     		if ( xoff < 0 || xoff >= size_in_x )
						continue;
					
			     		k = kx0[i] * ky0[j] + 
						a * ( kx0[i]*ky1[j] + kx1[i]*ky0[j] ) +
						a*a * kx1[i]*ky1[j];
			     		sum = sum + k * image_in[yoff*(long)size_in_x+xoff];
				}
		    	}

		    	image_out[ii] = sum;
	        }
       } 
}
