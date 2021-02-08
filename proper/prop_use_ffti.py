#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri


import os
import proper
import ctypes


def prop_use_ffti(**kwargs):
    """Instructs PROPER to use the Intel MKL FFT routines rather than the 
    built-in numpy FFT.  
    
    See the manual for how to use these. Intel FFT will be used by PROPER for 
    all future runs (including after exiting pyPROPER), unless changed by using 
    the DISABLE switch..  


    Parameters
    ----------
        None

    Other Parameters
    ----------------
    MKL_DIR : str
        Directory path to Intal MKL library
        
    DISABLE : bool
        Disable FFTI library use?

    Returns
    ----------
        None
    """

    ffti_dummy_file = os.path.join( os.path.expanduser('~'), '.proper_use_ffti' )
        
    if "MKL_DIR" in kwargs:
        mkl_dir = kwargs["MKL_DIR"]
    else:
        mkl_dir = None
        
    # Check if Intel MKL library is available on user's machine
    if os.name == 'posix':
        if  proper.system == 'Linux':
            ## for Linux system search for libmkl_rt.so
            if mkl_dir:
                mkl_lib = os.path.join(mkl_dir, 'libmkl_rt.so')
            else:
                mkl_lib = os.path.join('/opt/intel/mkl/lib/intel64', 'libmkl_rt.so')
                
            try:           
                ctypes.cdll.LoadLibrary(mkl_lib)
            except:
                print('Intel MKL Library not found. Using Numpy FFT.')
                ffti_flag = False
            else:
                ffti_flag = True
        elif proper.system == 'Darwin':
            ## for mac osx search for libmkl_rt.dylib
            if mkl_dir:
                mkl_lib = os.path.join(mkl_dir, 'libmkl_rt.dylib')
            else:
                mkl_lib = os.path.join('/opt/intel/mkl/lib/intel64', 'libmkl_rt.dylib')
                
            try:
                ctypes.cdll.LoadLibrary(mkl_lib)
            except:
                print('Intel MKL Library not found. Using Numpy FFT.')
                ffti_flag = False
            else:
                ffti_flag = True
    elif os.name == 'nt':
        ## for windows search for mk2_rt.dll
        if mkl_dir:
            mkl_lib = os.path.join(mkl_dir, 'mkl_rt.lib')
        else:
            mkl_lib = os.path.join('C:/Program Files(x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64', 'mkl_rt.lib')
        
        try:
            ctypes.cdll.LoadLibrary(mkl_lib)
        except:
            print('Intel MKL Library not found. Using Numpy FFT.')
            ffti_flag = False
        else:
            ffti_flag = True
    else:
        raise ValueError('Unsupported operating system %s. Stopping.' %(os.name))
        
    if proper.switch_set("DISABLE",**kwargs) or not ffti_flag:
        if os.path.isfile(ffti_dummy_file):
            os.remove(ffti_dummy_file)
        proper.use_ffti = False
        proper.use_fftw = False
    else:
        with open(ffti_dummy_file, 'w') as fd:
            fd.write('Using FFTI routines\n')
        proper.use_ffti = True
        proper.use_fftw = True
        
    return
