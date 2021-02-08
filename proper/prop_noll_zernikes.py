#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
#
#   Changes:
#   6 Sept 2019: JEK - check which version of scipy is used to load 
#                factorial from right module (thanks to Brandon Dube)


import numpy as np

import scipy
if scipy.__version__ > '1':
    from scipy.special import factorial
else:
    from scipy.misc import factorial

import proper

def prop_noll_zernikes(maxz, **kwargs):
    """Return a string array in which each element contains the Zernike polynomial
    equation corresponding to the index of that element.

    The polynomials are orthonormal for an unobscured circular aperture. They
    follow the ordering convention of Noll (J. Opt. Soc. America, 66, 207 (1976)).
    The first element (0) is always blank. The equations contain the variables "r"
    (normalized radius) and "t" (azimuth angle in radians). The polynomials have
    an RMS of 1.0 relative to a mean of 0.0.

    Parameters
    ----------
    maxz : int
        Maximum number of zernike polynomials to return. The returned string
        array will have max_z+1 elements, the first being blank.


    Returns
    -------
    z_list : numpy ndarray
        Returns a string array with each element containing z zernike polynomial
        (the first element is blank).

    max_r_power : float, optional
        The maximum radial exponent.

    max_theta_multiplier : float, optional
        Maximum multiplier of the angle.


    Other Parameters
    ----------------
    COMPACT : bool
       If set, the equations are returned using the naming convention for terms
       assumed by PROP_ZERNIKES.

    EXTRA_VALUES : bool
        If true, return maximum radial power and maximum theta multiplier in
        addition to equation strings

    Notes
    -----
    For example:
        zlist = prop_noll_zernikes(5)
        for i in range(1, 6):
            print(i, '   ', zlist[i])

    will display:
      1   1
      2   2 * (r)  * cos(t)
      3   2 * (r)  * sin(t)
      4   sqrt(3) * (2*r^2 - 1)
      5   sqrt(6) * (r^2)  * sin(2*t)
    Note that PROP_PRINT_ZERNIKES can also be used to print a table of Zernikes.
    """

    if proper.switch_set("COMPACT",**kwargs):
        rop = "_pow_"
    else:
        rop = "**"

    max_r_power = 0
    max_theta_multiplier = 0

    z_list = np.zeros(maxz+1, dtype = "S250")    # z_list[0] is always blank
    iz = 1
    n = 0

    while (iz <= maxz):
        for m  in range(np.mod(n,2), n+1, 2):
            for p in range(0, (m != 0) + 1):
                if n != 0:
                    if m != 0:
                        val = 2 * (n+1)
                    else:
                        val = n + 1

                    sqrt_val = int(np.sqrt(val))
                    if val == sqrt_val**2:
                        t = str(sqrt_val).strip() + " * ("
                    else:
                        t = "math.sqrt(" + str(val).strip() + ") * ("
                else:
                    z_list[iz] = "1"
                    iz += 1
                    continue

                for s in range(0, (n-m)//2 + 1):
                    term_top = int((-1)**s) * int(factorial(n-s))
                    term_bottom = int(factorial(s)) * int(factorial((n+m)/2-s)) * int(factorial((n-m)/2 - s))
                    term_val = int(term_top / term_bottom)
                    term = str(np.abs(term_val)).strip() + ".0"
                    term_r = int(n - 2*s)
                    rpower = str(term_r).strip()

                    if max_r_power < term_r:
                        max_r_power = term_r

                    if term_top != 0:
                        if s == 0:
                            if term_val < 0:
                                sign = "-"
                            else:
                                sign = ""
                        else:
                            if term_val < 0:
                                sign = " - "
                            else:
                                sign = " + "

                        if rpower == "0":
                            t += sign + term
                        elif term_r == 1:
                            if term_val != 1:
                                t += sign + term + "*r"
                            else:
                                t += sign + "r"
                        else:
                            if term_val != 1:
                                t += sign + term + "*r" + rop + rpower
                            else:
                                t += sign + "r" + rop + rpower

                if m > max_theta_multiplier:
                    max_theta_multiplier = m

                if m == 0:
                    cterm = ""
                else:
                    if (m != 1):
                        term_m = str(int(m)).strip() + "*t"
                    else:
                        term_m = "t"
                    if np.mod(iz,2) == 0:
                        if proper.switch_set("COMPACT",**kwargs):
                            cterm = " * cos" + str(int(m)).strip() + "t"
                        else:
                            cterm = " * cos(" + term_m + ")"
                    else:
                        if proper.switch_set("COMPACT",**kwargs):
                            cterm = " * sin" + str(int(m)).strip() + "t"
                        else:
                            cterm = " * sin(" + term_m + ")"

                if cterm != "":
                    z_list[iz] = t + ")" + cterm
                else:
                    t += ")"
                    z_list[iz] = t

                iz += 1
                if iz > maxz:
                    break

            if iz > maxz:
                break

        n += 1

    if proper.switch_set("EXTRA_VALUES",**kwargs):
        return (z_list, max_r_power, max_theta_multiplier)
    else:
        return z_list
