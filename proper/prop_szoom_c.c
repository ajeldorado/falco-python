#include <stdio.h>
#include <math.h>
//#include <malloc.h>
#include <stdlib.h> 
#define  K   13
#define  DK   6

/*--------------------------------------------------------------------------------------*/
double roundval( double x )
{
	if ( x > 0.0 ) 
		return( floor(x+0.5) );
	else
		return( -floor(-x+0.5) );
}

/*--------------------------------------------------------------------------------------*/
int prop_szoom_c(const void *image_input, int size_in, void *image_output, int size_out, double mag )
{
	double val, x, y, x_in, x_phase, y_in;	
	double  **sinc_table;
	int	i, ik, ikx, iky, ix, iy, x1, x2, x_pix, x_out, y1, y2, y_pix, y_out;

	const double *image_in = (double *)image_input;
	double *image_out = (double *)image_output;

	/* Precompute table of sinc kernel coefficients.  Because this routine *
	 * only expands or contracts the square image symmetrically about the  *
	 * center, just the kernel components for one axis are needed.	       */

	sinc_table = (double **)malloc( size_out * sizeof(double *) );
	for ( i = 0; i < size_out; ++i )
		sinc_table[i] = (double *)malloc( K * sizeof(double) );

	for ( x_out = 0; x_out < size_out; ++x_out )
	{
		x_in = (x_out - size_out/2) / mag;
		x_phase = x_in - roundval(x_in);
		for ( ik = 0; ik < K; ++ik )
		{
			x = (ik - K/2) - x_phase;
			if ( fabs(x) <= DK )
			{
				if ( x != 0.0 )
				{
					x = x * 3.141592653589793;
					sinc_table[x_out][ik] = sin(x)/x * sin(x/DK)/(x/DK);
				}
				else
					sinc_table[x_out][ik] = 1.0;
			}
			else
			{
				sinc_table[x_out][ik] = 0.0;
			}
		}
	}
			
	for ( y_out = 0; y_out < size_out; ++y_out )
	{
		y_in = (y_out - size_out/2) / mag;
		y_pix = roundval(y_in) + size_in/2;
		y1 = y_pix - K/2;
		y2 = y_pix + K/2;
		if ( (y1 < 0) || (y2 >= size_in) )
			continue;

		for ( x_out = 0; x_out < size_out; ++x_out )
		{
			x_in = (x_out - size_out/2) / mag;
			x_pix = roundval(x_in) + size_in/2;
			x1 = x_pix - K/2;
			x2 = x_pix + K/2;
			if ( (x1 < 0) || (x2 >= size_in) )
				continue;

			val = 0.0;
			iky = 0;
			for ( iy = y1; iy <= y2; ++iy )
			{
				ikx = 0;
				for ( ix = x1; ix <= x2; ++ix )
				{
					val = val + image_in[iy*(long)size_in+ix] * sinc_table[y_out][iky] * sinc_table[x_out][ikx];
					++ikx;
				}
				++iky;
			}
			image_out[y_out*(long)size_out+x_out] = val;
		}
	}

	for ( i = 0; i < size_out; ++i )
		free( sinc_table[i] );
	free( sinc_table );

	return( 0 );

} /* prop_szoom_c */

