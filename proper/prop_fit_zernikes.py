#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Revised 28 Oct 2019 - JEK - Optional parameters in upper case, previously
#   in lower case (inconsistent)
#   Modified Jan 2020 - JEK - fixed fitting errors due to incorrect indentations and
#     conditionals

import proper
import numpy as np
import math

def prop_fit_zernikes(wavefront0, pupil0, pupilradius0, nzer, **kwargs):
    """Fit circular Zernike polynomials to a 2D error map.  
    
    The user provides the error map, a 2D mask (zero or one) that defines valid 
    data values in the map, and teh radius of the aperture.

    Parameters
    ----------        
    wavefront : numpy ndarray
        A 2D array containing the aberration map. The returned Zernike 
        coefficients will have the same data units as this map.

    pupil : numpy ndarray
        A 2D array indicating which corresponding values in "wavefront" are valid. 
        A value of 1 indicates a valid point, 0 is a bad point or is obscured.

    aperture_radius : float
        Radius of the beam in the map in pixels. The Zernike polynomials are 
        normalized for a circle of this radius.

    nzer : int
        Maximum number of Zernike polynomials to fit (1 to nzer using the Noll 
        ordering scheme). This is arbitrary for unobscured polynomials. For 
        obscured ones, the max allowed is 22.  The polynomials are Noll ordered.

    
    Returns
    -------
    zcoeff : float
        Fitted Zernike coefficients, ordered such that zcoeff(0) is the first
        zernike polynomial (Z1, piston). The units are the same as those in 
        "wavefront".   

    fit : numpy ndarray, optional
        2D map of the zernike polynomials fitted to "wavefront". This map can
        be directly subtracted from "wavefront", for instance. This map is created
        at the sampling of the input wavefront.
    
    
    Other Parameters
    ----------------
    OBSCURATION_RATIO : float
        Ratio of the central obscuration to the aperture radius. Specifying this 
        value will cause Zernike polynomials normalized for an obscured 
        aperture to be fit rather than unobscured Zernikes, which is the default.

    XC, YC : int
        Specifies the center of the wavefront in the wavefront array in pixels,
        with the center of the lower-left pixel being (0.0,0.0). By default,
        the wavefront center is at the center of the array (nx/2,ny/2).
    """

    if not "OBSCURATION_RATIO" in kwargs: 
        eps = 0.0
    else:
        eps = kwargs["OBSCURATION_RATIO"] 

    if eps > 0 and nzer > 22:
        raise ValueError("PROP_FIT_ZERNIKES: ERROR: Limited to first 22 Obscured Zernikes.")            

    nx = wavefront0.shape[0]
    ny = wavefront0.shape[1]
    wavefront = wavefront0
    pupil = pupil0
    
    if not "XC" in kwargs: xc = int(nx / 2)
    if not "YC" in kwargs: yc = int(ny / 2)        

    if proper.switch_set("FIT",**kwargs):
        kval = [0,1]
    else: 
        kval = [0]

    for k in kval:
        x = np.arange(nx, dtype = np.float64) - xc
        y = np.arange(ny, dtype = np.float64) - yc
        xx,yy = np.meshgrid(x,y)
        r = np.sqrt(xx**2 + yy**2) / pupilradius0
        t = np.arctan2(yy, xx) 
        if k==0:
            #-- during fitting sta>, only use pixels in (possibly resized) pupil region
            ## create a mask
            r = r.ravel()
            t = t.ravel()
            wavefront = wavefront.ravel()
            pupil = pupil.ravel()
            w = (r < 1.0) & ( pupil != 0)
            nw = int(nx) * ny
            r =r[w] 
            t = t[w]
            wavefront = wavefront[w]
        else:
            nw = int(nx) * ny
            r = r.ravel()
            t = t.ravel()
            wavefront = wavefront.ravel()
        
        nw = len(r) 
        ab = np.zeros([nw,nzer], dtype = np.float64)
        ab[:,0] = 1.0

        if eps != 0:
            r2 = r * r
            r3 = r2 * r
            r4 = r3 * r
            r5 = r4 * r
            if ( nzer >= 2 ) : ab[:,1] =  (2 * r * np.cos(t))/np.sqrt(1 + eps**2)
            if ( nzer >= 3 ) : ab[:,2] =  (2 * r * np.sin(t))/np.sqrt(1 + eps**2)
            if ( nzer >= 4 ) : ab[:,3] =  (np.sqrt(3)*(1 + eps**2 - 2*r2))/(-1 + eps**2)
            if ( nzer >= 5 ) : ab[:,4] =  (np.sqrt(6) *r2 * np.sin(2*t))/np.sqrt(1 + eps**2 + eps**4)
            if ( nzer >= 6 ) : ab[:,5] =  (np.sqrt(6)*r2*np.cos(2*t))/np.sqrt(1 + eps**2 + eps**4)
            if ( nzer >= 7 ) : ab[:,6] =  (2*np.sqrt(2)*r*(2 + 2*eps**4 - 3*r2 + eps**2*(2 - 3*r2))* np.sin(t))/\
               ((-1 + eps**2)*np.sqrt(1 + 5*eps**2 + 5*eps**4 + eps**6))
            if ( nzer >= 8 ) : ab[:,7] =  (2*np.sqrt(2)*r*(2 + 2*eps**4 - 3*r2 + eps**2*(2 - 3*r2))*np.cos(t))/ \
               ((-1 + eps**2)*np.sqrt(1 + 5*eps**2 + 5*eps**4 + eps**6))
            if ( nzer >= 9 ):  ab[:,8] =  (2*np.sqrt(2)*r3*np.sin(3*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6)
            if ( nzer >= 10 ): ab[:,9] =  (2*np.sqrt(2)*r3*np.cos(3*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6)
            if ( nzer >= 11 ) : ab[:,10] =  (np.sqrt(5)*(1 + eps**4 - 6*r2 + 6*r4 + eps**2*(4 - 6*r2)))/ (-1 + eps**2)**2
            if ( nzer >= 12 ) : ab[:,11] =  (np.sqrt(10)*r2*(3 + 3*eps**6 - 4*r2 + eps**2*(3 - 4*r2) + \
                eps**4*(3 - 4*r2))*np.cos(2*t))/ ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4)*\
                (1 + 4*eps**2 + 10*eps**4 + 4*eps**6 + eps**8)))
            if ( nzer >= 13 ) : ab[:,12] =  (np.sqrt(10)*r2*(3 + 3*eps**6 - 4*r2 + eps**2*(3 - 4*r2) + \
                eps**4*(3 - 4*r2))*np.sin(2*t))/ ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4)* \
                (1 + 4*eps**2 + 10*eps**4 + 4*eps**6 + eps**8)))                                                
            if ( nzer >= 14 ) : ab[:,13] =  (np.sqrt(10)*r4*np.cos(4*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8)
            if ( nzer >= 15 ) : ab[:,14] =  (np.sqrt(10)*r4*np.sin(4*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8)
            if ( nzer >= 16 ) : ab[:,15] =  (2*np.sqrt(3)*r*(3 + 3*eps**8 - 12*r2 + 10*r4 - 12*eps**6*(-1 + r2) + \
                2*eps**4*(15 - 24*r2 + 5*r4) + 4*eps**2*(3 - 12*r2 + 10*r4))*np.cos(t))/ \
                ((-1 + eps**2)**2*np.sqrt((1 + 4*eps**2 + eps**4)* (1 + 9*eps**2 + 9*eps**4 + eps**6)))
            if ( nzer >= 17 ) : ab[:,16] =  (2*np.sqrt(3)*r*(3 + 3*eps**8 - 12*r2 + 10*r4 - 12*eps**6*(-1 + r2) + \
                2*eps**4*(15 - 24*r2 + 5*r4) + 4*eps**2*(3 - 12*r2 + 10*r4))*np.sin(t))/ \
                ((-1 + eps**2)**2*np.sqrt((1 + 4*eps**2 + eps**4)* (1 + 9*eps**2 + 9*eps**4 + eps**6)))
            if ( nzer >= 18 ) : ab[:,17] =  (2*np.sqrt(3)*r3*(4 + 4*eps**8 - 5*r2 + eps**2*(4 - 5*r2) + eps**4*(4 - 5*r2) + \
                eps**6*(4 - 5*r2))*np.cos(3*t))/ ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4 + eps**6)* \
                (1 + 4*eps**2 + 10*eps**4 + 20*eps**6 + 10*eps**8 + 4*eps**10 + eps**12)))
            if ( nzer >= 19 ) : ab[:,18] =  (2*np.sqrt(3)*r3*(4 + 4*eps**8 - 5*r2 + eps**2*(4 - 5*r2) + eps**4*(4 - 5*r2) + \
                eps**6*(4 - 5*r2))*np.sin(3*t))/ ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4 + eps**6)* \
                (1 + 4*eps**2 + 10*eps**4 + 20*eps**6 + 10*eps**8 + 4*eps**10 + eps**12)))
            if ( nzer >= 20 ) : ab[:,19] =  (2*np.sqrt(3)*r5*np.cos(5*t))/ np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8 + eps**10)
            if ( nzer >= 21 ) : ab[:,20] =  (2*np.sqrt(3)*r5*np.sin(5*t))/ np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8 + eps**10)
            if ( nzer == 22 ) : ab[:,21] =  (np.sqrt(7)*(1 + eps**6 - 12*r2 + 30*r4 - 20*r**6 + eps**4*(9 - 12*r2) + \
                eps**2*(9 - 36*r2 + 30*r4)))/ (-1 + eps**2)**3
        else:
            zca, maxrp, maxtc = proper.prop_noll_zernikes(nzer, COMPACT = True, EXTRA_VALUES = True)
            for i in range(2, maxrp+1):
                rps = str(i).strip()
                cmd = "r_pow_" + rps + " = r**i"
                exec(cmd)
            for i in range(1, maxtc+1):
                tcs = str(i).strip()
                cmd = "cos" + tcs + "t = np.cos(i*t)"
                exec(cmd)
                cmd = "sin" + tcs + "t = np.sin(i*t)"
                exec(cmd)
            #Create aberration map
            for iz in range(1,len(zca)-1): ab[:, iz] = eval(zca[iz+1])
    
        if k==0:
            zcoeff = np.linalg.lstsq(ab,wavefront,rcond=None)[0]
        else: 
            fitv = np.zeros((nw)) 
            for i in range(nzer): fitv += zcoeff[i] * ab[:,i]
            fit = np.reshape(fitv, (nx, ny))

    if k==1:
        return (zcoeff, fit) 
    else:
        return zcoeff

