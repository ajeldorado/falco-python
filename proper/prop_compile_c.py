#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import os
import proper
import subprocess


def prop_compile_c():
    """Compile cubic convoluton interpolation and szoom C modules.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    if os.name == 'posix':
        # Compile and flags
        if proper.system == 'Linux':
            CC = 'gcc'
            FLAGS = ['-shared', '-fPIC']
        elif proper.system == 'Darwin':
            CC = 'gcc'
            FLAGS = ['-shared', '-framework Python']

        # Compile cubic convolution interpolation C code
        print('Compiling ' + proper.cubic_conv_c + ' ...')
        cmd = CC + ' ' + proper.cubic_conv_c + ' -o ' + proper.cubic_conv_lib
        for flag in FLAGS:
            cmd += ' ' + flag
        subprocess.call(cmd, shell = True)

        # Compile threaded cubic convolution interpolation C code
        print('Compiling ' + proper.cubic_conv_threaded_c + ' ...')
        cmd = CC + ' ' + proper.cubic_conv_threaded_c + ' -o ' + proper.cubic_conv_threaded_lib
        for flag in FLAGS:
            cmd += ' ' + flag
        subprocess.call(cmd, shell = True)

        # Compile szoom C code
        print('Compiling ' + proper.szoom_c + ' ...')
        cmd = CC + ' ' + proper.szoom_c + ' -o ' + proper.szoom_c_lib
        for flag in FLAGS:
            cmd += ' ' + flag
        subprocess.call(cmd, shell = True)
    elif os.name == 'nt':
        print('PROP_COMPILE_C: Cubic convolution interpolation C modules only available for Linux and Mac.')
    else:
        print('PROP_COMPILE_C: Unknown operating system type = ' + proper.system)

    return
