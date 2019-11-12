#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>



/* based on the traditional separable derivation described in "Two-Dimensional Cubic Convolution" by
 * Reichenbach & Geng, IEEE Transactions on Image Processing, v.12, 857 (2003).  This code is intended
 * to provide the same results as the CUBIC=-0.5 option to IDL's INTERPOLATE function.  */

static const double a = -0.5;


struct parstruct {
	double *image_in, *image_out;
	double *x_in, *y_in;
	long  x_size_in, y_size_in;
	long  x_size_out, y_size_out;
	double *kx0, *kx1;
};

struct threadstruct {
	struct parstruct *pars;
	int	thread, start_row, end_row;
};
 
/*--------------------------------------------------------------------------------------*/
static double roundval( double x )
{
	if ( x > 0.0 ) 
		return( floor(x+0.5) );
	else
		return( -floor(-x+0.5) );
}

/*--------------------------------------------------------------------------------------*/
static void *interp_thread( void *tpar )
{
	struct threadstruct *thread;
	struct parstruct *pars;

	double	x, y, mag;
	double  k, d, sum, xc, yc, ky0[5], ky1[5];
	int	i, j, ix, xoff, yoff, x_pix, x_out, y_pix, y_out;
	size_t	ii;

	thread = tpar;
	pars = thread->pars;

	for ( y_out = thread->start_row; y_out <= thread->end_row; ++y_out )
	{
		for ( x_out = 0; x_out < pars->x_size_out; ++x_out )
		{
			ii = y_out * (size_t)pars->x_size_out + x_out;

			y_pix = roundval( pars->y_in[ii] );
			yc = roundval(pars->y_in[ii]) - pars->y_in[ii];

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

			x_pix = roundval( pars->x_in[ii] );

			sum = 0;
			for ( j = 0; j <= 4; ++j )
			{
				yoff = y_pix + j - 2;
				if ( yoff < 0 || yoff >= pars->y_size_in )
					continue;
				for ( i = 0; i <= 4; ++i )
				{
					xoff = x_pix + i - 2;
					if ( xoff < 0 || xoff >= pars->x_size_in )
						continue;
					ix = x_out * 5 + i;
					k = pars->kx0[ix] * ky0[j] + 
					    a * ( pars->kx0[ix]*ky1[j] + pars->kx1[ix]*ky0[j] ) +
					    a*a * pars->kx1[ix]*ky1[j];
					sum = sum + k * pars->image_in[yoff*(long)pars->x_size_in+xoff];
				}
			}

			pars->image_out[ii] = sum;
		}
	}

	return( NULL );
}

/*--------------------------------------------------------------------------------------*/
int cubic_conv_c(const void * image_input, int x_size_in, int y_size_in, void * image_output, int x_size_out, int y_size_out, void * x_input, void * y_input, int num_threads )
{
	double	x, y, mag;
	double  k, d, sum, xc, yc, *kx0, *kx1;
	int	i, j, ix, xoff, yoff, x_pix, x_out, y_pix, y_out;
	struct threadstruct *threadpars;
	struct parstruct pars;
        pthread_t *threads;
        int     dy, ithread;
        void    *retval;

	const double *image_in = (double *)image_input;
	double *image_out = (double *)image_output;
	double *x_in = (double *)x_input;	/* vector of flattened X coordinates in input image */
	double *y_in = (double *)y_input;	/* vector of flattened Y coordinates in input image */

	threadpars = (struct threadstruct *)malloc( num_threads * sizeof(struct threadstruct) );
	threads = (pthread_t *)malloc( num_threads * sizeof(pthread_t) );

	/* precompute X axis kernels */

	kx0 = (double *)malloc( 5 * x_size_out * sizeof(double) );
	kx1 = (double *)malloc( 5 * x_size_out * sizeof(double) );
	for ( x_out = 0; x_out < x_size_out; ++x_out )
	{
		x_pix = roundval( x_in[x_out] );
		xc = roundval(x_in[x_out]) - x_in[x_out];

		for ( j = -2; j <= 2; ++j )
		{
			ix = x_out * 5 + j + 2;
			d = fabs(xc + j);
			if ( d <= 1 )
			{
				kx0[ix] = d*d*(2*d - 3) + 1;
				kx1[ix] = d*d*(d - 1);
			}
			else if ( d <= 2 )
			{
				kx0[ix] = 0;
				kx1[ix] = d*(d*d - 5*d + 8) - 4;
			}
			else
			{
				kx0[ix] = 0;
				kx1[ix] = 0;
			}
		}
	}

	/* create threads */

	pars.image_in = image_in;
	pars.image_out = image_out;
	pars.x_in = x_in;
	pars.y_in = y_in;
	pars.x_size_in = x_size_in;
	pars.y_size_in = y_size_in;
	pars.x_size_out = x_size_out;
	pars.y_size_out = y_size_out;
	pars.kx0 = kx0;
	pars.kx1 = kx1;

	dy = y_size_out / num_threads;
	if ( dy * num_threads != y_size_out )
		dy = dy + 1;

	for ( ithread = 0; ithread < num_threads; ++ithread )
	{
		threadpars[ithread].pars = &pars;
		threadpars[ithread].thread = ithread;
		threadpars[ithread].start_row = ithread * dy;
		threadpars[ithread].end_row = (ithread+1) * dy - 1;
		if ( threadpars[ithread].end_row >= y_size_out )
			threadpars[ithread].end_row = y_size_out - 1;

                if ( pthread_create(&threads[ithread], NULL, interp_thread, &threadpars[ithread]) )
                {
                        fprintf(stderr, "Cannot make thread %d\n", ithread);
			free( threadpars );
			free( threads );
			free( kx0 );
			free( kx1 );
                        exit(1);
                }
	}

        /* Join (collapse) the two threads */

        for ( ithread = 0; ithread < num_threads; ++ithread )
        {
                if ( pthread_join(threads[ithread], &retval) )
                {
                        fprintf(stderr, "Thread join failed\n");
			free( threadpars );
			free( threads );
			free( kx0 );
			free( kx1 );
                        exit(1);
                }
        }

	free( kx0 );
	free( kx1 );
	free( threadpars );
	free( threads );

	return( 0 );
} 
