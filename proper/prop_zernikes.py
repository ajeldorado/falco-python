#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Modified - JEK - 15 July 2019: fixed use of ^ instead of ** in obscured Zernikes


import math
import proper
import numpy as np


def prop_zernikes(a, zernike_num, zernike_val, eps = 0., **kwargs):
    """Add Zernike-polynomial wavefront errors to current wavefront. 
    
    Noll ordering is used and a circular system is assumed. An arbitrary number 
    of Zernikes normalized for an unobscured circular region can be computed, 
    but only the first 22 Zernikes can be computed normalized for a 
    centrally-obscured region.
    

    Parameters
    ----------
    a : object
        WaveFront class object
        
    zernike_num : numpy ndarray
        Scalar or 1D array specifying which Zernike polynomials to include
        
    zernike_val : numpy ndarray
        Scalar or 1D array containing Zernike coefficients (in meters of RMS 
        wavefront phase error or dimensionless RMS amplitude error) for Zernike 
        polynomials indexed by "zernike_num".
        
    eps : float
        Central obscuration ratio (0.0-1.0); default is 0.0
        

    Returns
    -------
        None 
        Adds wavefront errors to current wavefront array
        
    dmap : numpy ndarray
        Aberration map
        
    
    Other Parameteres
    -----------------
    AMPLITUDE : bool
        Optional keyword that specifies that the Zernike values in "zernike_val"
        represent the wavefront RMS amplitude (rather than phase) variation.  
        The current wavefront will be multipled by the generated map.
        
    NAME : str
        String containing name of surface that will be printed when executed.
        
    NO_APPLY : bool
        If set, the aberration map will be generated but will not be applied to
        the wavefront. This is useful if you just want to generate a map for
        your own use and modification (e.g. to create an error map for a multi-
        segmented system, each with its own aberration map).
        
    RADIUS : float
        Optional keyword specifying the radius to which the Zernike polynomials
        are normalized. If not specified, the pilot beam radius is used.
    
    
    Raises
    ------
    ValueError:
        Maximum index for an obscured Zernike polynomial is 22
    
    
    Notes
    -----
    The user specifies 1D arrays containing the Zernike polynomial coefficient
    indicies, the respective coefficients, and if an obstructed circular aperture
    the central obscuration ratio. A wavefront error map will be generated and 
    added to the current wavefront.
    
    Zernike index and corresponding aberration for 1st 22 zernikes
     1 : Piston
     2 : X tilt
     3 : Y tilt
     4 : Focus
     5 : 45 degree astigmatism
     6 : 0 degree astigmatism
     7 : Y coma
     8 : X coma
     9 : Y clover (trefoil)
    10 : X clover (trefoil)
    11 : 3rd order spherical
        12 : 5th order 0 degree astig
        13 : 5th order 45 degree astig
    14 : X quadrafoil
    15 : Y quadrafoil
    16 : 5th order X coma
    17 : 5th order Y coma
    18 : 5th order X clover
    19 : 5th order Y clover
    20 : X pentafoil
    21 : Y pentafoil
    22 : 5th order spherical

    """
    zernike_num = np.asarray(zernike_num)
    zernike_val = np.asarray(zernike_val)
    n = proper.n
    
    if proper.print_it and not proper.switch_set("NO_APPLY",**kwargs):
        if "NAME" in kwargs:
            print("Applying aberrations at %s" %kwargs["NAME"])
        else:
            print("Applying aberrations")
            
    max_z = zernike_num.max()
    
    if eps != 0. and max_z > 22:
        raise ValueError("PROP_ZERNIKES: Maximum index for an obscured Zernike polynomial is 22.")
        
    dmap = np.zeros([n,n], dtype = np.float64)
    
    if not "RADIUS" in kwargs:
        beam_radius = proper.prop_get_beamradius(a)
    else:
        beam_radius = kwargs["RADIUS"]
        
    x = (np.arange(n, dtype = np.float64) - n//2) * proper.prop_get_sampling(a) / beam_radius
    x_pow_2 = x**2
    
    if (eps == 0.):
        # get list of executable equations defining Zernike polynomials
        zlist, maxrp, maxtc = proper.prop_noll_zernikes(max_z, COMPACT = True, EXTRA_VALUES = True)
        
        for j in range(n):
            ab = np.zeros(n, dtype = np.float64)
            y = (j - n//2) * proper.prop_get_sampling(a) / beam_radius
            r = np.sqrt(x_pow_2 + y**2)
            t = np.arctan2(y,x)
        
            # predefine r**power, cos(const*theta), sin(const*theta) vectors
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
           
            # assemble aberrations
            for iz in range(zernike_num.size):
                tmp = eval(zlist[zernike_num[iz]])
                ab += zernike_val[iz] * tmp
            
            dmap[j,:] += ab
    else:
        for j in range(n):
            y = (j-n//2) * proper.prop_get_sampling(a) / beam_radius
            r = np.sqrt(x_pow_2 + y**2)
            r2 = r**2
            r3 = r**3
            r4 = r**4
            r5 = r**5
            t = np.arctan2(y,x)
            
            for iz in range(len(zernike_num)):
                if zernike_num[iz] == 1:
                    ab = 1.
                elif zernike_num[iz] == 2:
                    ab = (2*r*np.cos(t))/np.sqrt(1 + eps**2)
                elif zernike_num[iz] == 3:
                    ab = (2*r*np.sin(t))/np.sqrt(1 + eps**2)
                elif zernike_num[iz] == 4:
                    ab = (np.sqrt(3)*(1 + eps**2 - 2*r2))/(-1 + eps**2)
                elif zernike_num[iz] == 5:
                    ab = (np.sqrt(6)*r2*np.sin(2*t))/np.sqrt(1 + eps**2 + eps**4)
                elif zernike_num[iz] == 6:
                    ab = (np.sqrt(6)*r2*np.cos(2*t))/np.sqrt(1 + eps**2 + eps**4)
                elif zernike_num[iz] == 7:
                    ab = (2*np.sqrt(2)*r*(2 + 2*eps**4 - 3*r2 + eps**2*(2 - 3*r2))*np.sin(t))/ \
                        ((-1 + eps**2)*np.sqrt(1 + 5*eps**2 + 5*eps**4 + eps**6))
                elif zernike_num[iz] == 8:
                    ab = (2*np.sqrt(2)*r*(2 + 2*eps**4 - 3*r2 + eps**2*(2 - 3*r2))*np.cos(t))/ \
                        ((-1 + eps**2)*np.sqrt(1 + 5*eps**2 + 5*eps**4 + eps**6))
                elif zernike_num[iz] == 9:
                    ab = (2*np.sqrt(2)*r3*np.sin(3*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6)
                elif zernike_num[iz] == 10:
                    ab = (2*np.sqrt(2)*r3*np.cos(3*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6)
                elif zernike_num[iz] == 11:
                    ab = (np.sqrt(5)*(1 + eps**4 - 6*r2 + 6*r4 + eps**2*(4 - 6*r2)))/ (-1 + eps**2)**2
                elif zernike_num[iz] == 12:
                    ab = (np.sqrt(10)*r2*(3 + 3*eps**6 - 4*r2 + eps**2*(3 - 4*r2) + eps**4*(3 - 4*r2))*np.cos(2*t))/ \
                        ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4)*(1 + 4*eps**2 + 10*eps**4 + 4*eps**6 + eps**8)))
                elif zernike_num[iz] == 13:
                    ab = (np.sqrt(10)*r2*(3 + 3*eps**6 - 4*r2 + eps**2*(3 - 4*r2) + eps**4*(3 - 4*r2))*np.sin(2*t))/ \
                        ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4)*(1 + 4*eps**2 + 10*eps**4 + 4*eps**6 + eps**8)))
                elif zernike_num[iz] == 14:
                    ab = (np.sqrt(10)*r4*np.cos(4*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8)                    
                elif zernike_num[iz] == 15:
                    ab = (np.sqrt(10)*r4*np.sin(4*t))/np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8)
                elif zernike_num[iz] == 16:
                    ab = (2*np.sqrt(3)*r*(3 + 3*eps**8 - 12*r2 + 10*r4 - 12*eps**6*(-1 + r2) + 2*eps**4*(15 - 24*r2 + 5*r4) + \
                        4*eps**2*(3 - 12*r2 + 10*r4))*np.cos(t))/((-1 + eps**2)**2*np.sqrt((1 + 4*eps**2 + eps**4)* (1 + 9*eps**2 + 9*eps**4 + eps**6)))
                elif zernike_num[iz] == 17:
                    ab = (2*np.sqrt(3)*r*(3 + 3*eps**8 - 12*r2 + 10*r4 - 12*eps**6*(-1 + r2) + 2*eps**4*(15 - 24*r2 + 5*r4) + \
                        4*eps**2*(3 - 12*r2 + 10*r4))*np.sin(t))/((-1 + eps**2)**2*np.sqrt((1 + 4*eps**2 + eps**4)* (1 + 9*eps**2 + 9*eps**4 + eps**6)))
                elif zernike_num[iz] == 18:
                    ab = (2*np.sqrt(3)*r3*(4 + 4*eps**8 - 5*r2 + eps**2*(4 - 5*r2) + eps**4*(4 - 5*r2) + eps**6*(4 - 5*r2))*np.cos(3*t))/ \
                        ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4 + eps**6)*(1 + 4*eps**2 + 10*eps**4 + 20*eps**6 + 10*eps**8 + 4*eps**10 + eps**12)))
                elif zernike_num[iz] == 19:
                    ab = (2*np.sqrt(3)*r3*(4 + 4*eps**8 - 5*r2 + eps**2*(4 - 5*r2) + eps**4*(4 - 5*r2) + eps**6*(4 - 5*r2))*np.sin(3*t))/ \
                        ((-1 + eps**2)*np.sqrt((1 + eps**2 + eps**4 + eps**6)*(1 + 4*eps**2 + 10*eps**4 + 20*eps**6 + 10*eps**8 + 4*eps**10 + eps**12)))
                elif zernike_num[iz] == 20:
                    ab = (2*np.sqrt(3)*r5*np.cos(5*t))/ np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8 + eps**10)
                elif zernike_num[iz] == 21:
                    ab = (2*np.sqrt(3)*r5*np.sin(5*t))/ np.sqrt(1 + eps**2 + eps**4 + eps**6 + eps**8 + eps**10)
                elif zernike_num[iz] == 22:
                    ab = (np.sqrt(7)*(1 + eps**6 - 12*r2 + 30*r4 - 20*r**6 + eps**4*(9 - 12*r2) + eps**2*(9 - 36*r2 + 30*r4)))/ (-1 + eps**2)**3
                    
                dmap[j,:] += zernike_val[iz] * ab
            
    if not proper.switch_set("NO_APPLY",**kwargs):
        if proper.switch_set("AMPLITUDE",**kwargs):
            a.wfarr *= proper.prop_shift_center(dmap)
        else:
            i = complex(0,1)
            a.wfarr *= np.exp(i*2*np.pi/a.lamda*proper.prop_shift_center(dmap))
            
    return dmap
