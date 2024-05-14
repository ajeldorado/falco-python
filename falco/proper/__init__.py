#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import os
import platform
from glob import glob
import os.path as _osp
import multiprocessing as mp

# PyPROPER library directory path
lib_dir = _osp.abspath(_osp.dirname(__file__))

# PyPROPER version number
__version__ = '3.2.7'


# If cubic convolution interpolation shared library exist,
# use it instead of scipy map_coordinates
system = platform.system()
use_cubic_conv = False
cubic_conv_c = os.path.join(lib_dir, 'cubic_conv_c.c')
cubic_conv_lib = ''

if os.name == 'posix':
    libcconvpath = glob(_osp.join(lib_dir, 'libcconv.*so'))
    if len(libcconvpath) > 0:
        cubic_conv_lib = libcconvpath[0]
        use_cubic_conv = True
    else:
        cubic_conv_lib = _osp.join(lib_dir, 'libcconv.so')

# If cubic convolution threaded interpolation shared library exist,
# use it instead of scipy map_coordinates
use_cubic_conv_threaded = False
cubic_conv_threaded_c = os.path.join(lib_dir, 'cubic_conv_threaded_c.c')
cubic_conv_threaded_lib = ''

if os.name == 'posix':
    libcconvpath = glob(_osp.join(lib_dir, 'libcconvthread.*so'))
    if len(libcconvpath) > 0:
        cubic_conv_threaded_lib = libcconvpath[0]
        use_cubic_conv_threaded = True
    else:
        cubic_conv_threaded_lib = _osp.join(lib_dir, 'libcconvthread.so')

# Compile prop_szoom_c.c if compilers exist
use_szoom_c = False
szoom_c = os.path.join(lib_dir, 'prop_szoom_c.c')
szoom_c_lib = ''

if os.name == 'posix':
    libszoompath = glob(_osp.join(lib_dir, 'libszoom.*so'))
    if len(libszoompath) > 0:
        szoom_c_lib = libszoompath[0]
        use_szoom_c = True
    else:
        szoom_c_lib = _osp.join(lib_dir, 'libszoom.so')


from .prop_cubic_conv import prop_cubic_conv
from .prop_compile_c import prop_compile_c

from .prop_ffti import mkl_fft2
from .prop_dftidefs import *
from .prop_use_ffti import prop_use_ffti

from .prop_fftw import prop_fftw
from .prop_fftw_wisdom import prop_fftw_wisdom
from .prop_use_fftw import prop_use_fftw

from .prop_8th_order_mask import prop_8th_order_mask
from .prop_add_phase import prop_add_phase
from .prop_add_wavefront import prop_add_wavefront
from .prop_begin import prop_begin
from .prop_circular_aperture import prop_circular_aperture
from .prop_circular_obscuration import prop_circular_obscuration

from .prop_define_entrance import prop_define_entrance
from .prop_divide import prop_divide
from .prop_dm import prop_dm
from .prop_ellipse import prop_ellipse
from .prop_elliptical_aperture import prop_elliptical_aperture
from .prop_elliptical_obscuration import prop_elliptical_obscuration
from .prop_end import prop_end
from .prop_end_savestate import prop_end_savestate
from .prop_errormap import prop_errormap
from .prop_execute_multi import prop_execute_multi
from .prop_fit_dm import prop_fit_dm
from .prop_fits_read import prop_fits_read
from .prop_fits_write import prop_fits_write
from .prop_fit_zernikes import prop_fit_zernikes
from .prop_shift_center import prop_shift_center
from .prop_get_amplitude import prop_get_amplitude
from .prop_get_beamradius import prop_get_beamradius
from .prop_get_distancetofocus import prop_get_distancetofocus
from .prop_execute_multi import prop_execute_multi
from .prop_get_fratio import prop_get_fratio
from .prop_get_gridsize import prop_get_gridsize
from .prop_get_nyquistsampling import prop_get_nyquistsampling
from .prop_get_phase import prop_get_phase
from .prop_get_refradius import prop_get_refradius
from .prop_get_sampling_arcsec import prop_get_sampling_arcsec
from .prop_get_sampling import prop_get_sampling
from .prop_get_sampling_radians import prop_get_sampling_radians
from .prop_get_wavefront import prop_get_wavefront
from .prop_get_wavelength import prop_get_wavelength
from .prop_get_z import prop_get_z
from .prop_hex_wavefront import prop_hex_wavefront
from .prop_hex_zernikes import prop_hex_zernikes
from .prop_init_savestate import prop_init_savestate
from .prop_irregular_polygon import prop_irregular_polygon
from .prop_is_statesaved import prop_is_statesaved
from .prop_lens import prop_lens
from .prop_load_fftw_wisdom import prop_load_fftw_wisdom
from .prop_magnify import prop_magnify
from .prop_multiply import prop_multiply
from .prop_noll_zernikes import prop_noll_zernikes
from .prop_pixellate import prop_pixellate
from .prop_polygon import prop_polygon
from .prop_print_zernikes import prop_print_zernikes
from .prop_propagate import prop_propagate
from .prop_psd_errormap import prop_psd_errormap
from .prop_ptp import prop_ptp
from .prop_qphase import prop_qphase
from .prop_radius import prop_radius
from .prop_readmap import prop_readmap
from .prop_rectangle import prop_rectangle
from .prop_rectangular_aperture import prop_rectangular_aperture
from .prop_rectangular_obscuration import prop_rectangular_obscuration
from .prop_resamplemap import prop_resamplemap
from .prop_rotate import prop_rotate
from .prop_rounded_rectangle import prop_rounded_rectangle
from .prop_run_multi import prop_run_multi
from .prop_run import prop_run
from .prop_savestate import prop_savestate
from .prop_select_propagator import prop_select_propagator
from .prop_set_antialiasing import prop_set_antialiasing
from .prop_shift_center import prop_shift_center
from .prop_sinc import prop_sinc
from .prop_state import prop_state
from .prop_stw import prop_stw
from .prop_szoom import prop_szoom
from .prop_table import prop_table
from .prop_wavefront import WaveFront
from .prop_writemap import prop_writemap
from .prop_wts import prop_wts
from .prop_zernikes import prop_zernikes
from .switch_set import switch_set

# Common configuration variables
n = 0
first_pass = 0
ndiam = 0
ndiam_frac = 0
verbose = False
layout_only = False
layout_array = []
print_it = True
rayleigh_factor = 1
old_opd = 0
phase_offset = False
print_total_intensity = False
do_table = False
save_state = 1
save_state_lam = []
statefile = ""
total_original_pupil = 0
action_num = 0
lens_fl_list = []
lens_eff_fratio_list = []
beam_diam_list = []
distance_list = []
surface_name_list = []
sampling_list = []


# Which FFT to use - numpy, fftw or Intel MKL fft
use_fftw = False
use_ffti = False
fft_nthreads = 0    # only used for FFTI or FFTW

# if using Intel FFT and not multiprocessing, experiments show best performance is obtained
# using only the real CPUs, not hyperthreads (I'm assuming 1/1 ratio of real/hyperthreads, as
# is the case on most Xeons); when multiprocessing, the best performance is with 1 threads per FFT

ffti_single_nthreads = mp.cpu_count() // 2
ffti_multi_nthreads = 1

# if using FFTW and not multiprocessing, experiments show best performance is obtained
# using only 8 real CPUs, not hyperthreads (I'm assuming 1/1 ratio of real/hyperthreads, as
# is the case on most Xeons); when multiprocessing, the best performance is with 1 thread 
# per FFT

fftw_single_nthreads = mp.cpu_count() // 2
fftw_single_nthreads = 1 if fftw_single_nthreads < 1 else 8 if fftw_single_nthreads > 8 else fftw_single_nthreads
fftw_multi_nthreads = 1
fftw_use_wisdom = False

home_dir = os.path.expanduser('~')
ffti_test = os.path.join( home_dir, '.proper_use_ffti' )
fftw_test = os.path.join( home_dir, '.proper_use_fftw' )

if os.path.isfile(ffti_test):
    use_ffti = True
    use_fftw = True
elif os.path.isfile(fftw_test):
    use_fftw = True

# define default value for subsampling of shape edges for antialiasing.
# can be overridden with prop_set_antialiasing

antialias_subsampling = int(11)

