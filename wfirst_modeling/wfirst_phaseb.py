#   Copyright 2019 California Institute of Technology
# ------------------------------------------------------------------

# Version 1.0, 3 January 2019, JEK
# Version 1.0a, 9 January 2019, JEK 
#    Changes: 1) now append polaxis to output_field_rootname 
#             2) can output field at FPM exit pupil (via output_field_rootname) without 
#                having to stop there
# Version 1.0b, 16 January 2019, JEK
#    Changes: Multiplied wfirst_phaseb_GROUND_TO_ORBIT_phase_error_V1.0.fits by 4.2x to get the
#               the RMS WFE at the CGI entrance to be 76.4 nm RMS.  The new maps is in
#               result to wfirst_phaseb_GROUND_TO_ORBIT_4.2X_phase_error_V1.0.fits
#   Version 1.0c, 13 February 2019, JEK 
#      Changes: Changed HLC design to HLC-20190210; added 'fpm_axis' parameter to specify
#                 axis of HLC FPM ('s' or 'p') due to FPM tilt; changed 'spc' coronagraph
#                 option to 'cor_type' to 'spc-ifs' and changed default SPC to SPC-20190130 and
#                 updated associated parameters; added 'spc-wide' option to 'cor_type' to use
#                 SPC-20191220.  Changed DM tilt to rotation about Y axis.
#   Version 1.0d, 6 March 2019, JEK 
#      Changes: Added new parameters: cgi & mask & lyot stop & FPM shifts in meters; final
#                 sampling in meters; simplified how some masks are shifted
#   Version 1.1, 7 May 2019, JEK
#      Changes: Created wfirst_phase_proper python package and altered code to access
#                 directory names and functions defined in that package. Removed option
#                 for fpm_axis, since the FPM tilt is small enough now that it doesn't matter
#   Version 1.2, 28 May 2019, JEK
#      Changed lib_dir to data_dir 
#   Version 1.4, 25 Sept 2019, JEK
#      Added optional use_pupil_mask parameter (SPC only)
#   Version 1.5, 25 Sept 2019, JEK
#      Aliased spc-ifs_short and spc-ifs_long to spc-spec_short and spc-spec_long
#   Version 1.6, 18 Oct 2019, JEK
#      Changes: Added option to shift field stop (HLC)
#   Version 1.6.1, 21 Nov 2019, JEK
#      Fixed bug in shifting SPC mask when shift given in meters
#   Version 1.7, 3 Dec 2019, JEK
#      Changes: Replaced defocus-vs-focal-length table with revised version; added logic to
#               force planar propagation when requested defocus is <=4 waves.

import proper
import numpy as np
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits
import math
from wfirst_phaseb_proper import trim, mft2, ffts, polmap
import wfirst_phaseb_proper

# Written by John Krist


###############################################################################33
def radius( n ):
    x = np.arange( n ) - int(n)//2
    r = np.sqrt( x*x + x[:,np.newaxis]**2 )
    return r

###############################################################################33
def angle( n ):
    x = np.arange( n ) - int(n)//2
    return np.arctan2( x[:,np.newaxis], x ) * (180 / np.pi)


###############################################################################33
def wfirst_phaseb( lambda_m, output_dim0, PASSVALUE={'dummy':0} ):

    # "output_dim" is used to specify the output dimension in pixels at the final image plane.
    # Computational grid sizes are hardcoded for each coronagraph.
    # Based on Zemax prescription "WFIRST_CGI_DI_LOWFS_Sep24_2018.zmx" by Hong Tang.


    data_dir = wfirst_phaseb_proper.data_dir
    if 'PASSVALUE' in locals():
        if 'data_dir' in PASSVALUE: data_dir = PASSVALUE['data_dir']

    map_dir = data_dir + wfirst_phaseb_proper.map_dir
    polfile = data_dir + wfirst_phaseb_proper.polfile

    cor_type = 'hlc'            # coronagraph type ('hlc', 'spc', 'none')
    source_x_offset_mas = 0     # source offset in mas (tilt applied at primary)
    source_y_offset_mas = 0                 
    source_x_offset = 0         # source offset in lambda0_m/D radians (tilt applied at primary)
    source_y_offset = 0                 
    polaxis = 0                 # polarization axis aberrations: 
                                #    -2 = -45d in, Y out 
                                #    -1 = -45d in, X out 
                                #     1 = +45d in, X out 
                                #     2 = +45d in, Y out 
                                #     5 = mean of modes -1 & +1 (X channel polarizer)
                                #     6 = mean of modes -2 & +2 (Y channel polarizer)
                                #    10 = mean of all modes (no polarization filtering)
    use_errors = 1              # use optical surface phase errors? 1 or 0 
    zindex = np.array([0,0])    # array of Zernike polynomial indices
    zval_m = np.array([0,0])    # array of Zernike coefficients (meters RMS WFE)
    use_aperture = 0            # use apertures on all optics? 1 or 0
    cgi_x_shift_pupdiam = 0     # X,Y shear of wavefront at FSM (bulk displacement of CGI); normalized relative to pupil diameter
    cgi_y_shift_pupdiam = 0          
    cgi_x_shift_m = 0           # X,Y shear of wavefront at FSM (bulk displacement of CGI) in meters
    cgi_y_shift_m = 0          
    fsm_x_offset_mas = 0        # offset in focal plane caused by tilt of FSM in mas
    fsm_y_offset_mas = 0         
    fsm_x_offset = 0            # offset in focal plane caused by tilt of FSM in lambda0/D
    fsm_y_offset = 0            
    end_at_fsm = 0              # end propagation after propagating to FSM (no FSM errors)
    focm_z_shift_m = 0          # offset (meters) of focus correction mirror (+ increases path length)
    use_hlc_dm_patterns = 0     # use Dwight's HLC default DM wavefront patterns? 1 or 0
    use_dm1 = 0                 # use DM1? 1 or 0
    use_dm2 = 0                 # use DM2? 1 or 0
    dm_sampling_m = 0.9906e-3   # actuator spacing in meters
    dm1_xc_act = 23.5           # for 48x48 DM, wavefront centered at actuator intersections: (0,0) = 1st actuator center
    dm1_yc_act = 23.5              
    dm1_xtilt_deg = 0           # tilt around X axis (deg)
    dm1_ytilt_deg = 5.7         # effective DM tilt in deg including 9.65 deg actual tilt and pupil ellipticity
    dm1_ztilt_deg = 0           # rotation of DM about optical axis (deg)
    dm2_xc_act = 23.5           # for 48x48 DM, wavefront centered at actuator intersections: (0,0) = 1st actuator center
    dm2_yc_act = 23.5               
    dm2_xtilt_deg = 0           # tilt around X axis (deg)
    dm2_ytilt_deg = 5.7         # effective DM tilt in deg including 9.65 deg actual tilt and pupil ellipticity
    dm2_ztilt_deg = 0           # rotation of DM about optical axis (deg)
    use_pupil_mask = 1          # SPC only: use SPC pupil mask (0 or 1)
    mask_x_shift_pupdiam = 0    # X,Y shear of shaped pupil mask; normalized relative to pupil diameter
    mask_y_shift_pupdiam = 0          
    mask_x_shift_m = 0          # X,Y shear of shaped pupil mask in meters
    mask_y_shift_m = 0          
    use_fpm = 1                 # use occulter? 1 or 0
    fpm_x_offset = 0            # FPM x,y offset in lambda0/D
    fpm_y_offset = 0
    fpm_x_offset_m = 0          # FPM x,y offset in meters
    fpm_y_offset_m = 0
    fpm_z_shift_m = 0           # occulter offset in meters along optical axis (+ = away from prior optics)
    pinhole_diam_m = 0          # FPM pinhole diameter in meters
    end_at_fpm_exit_pupil = 0   # return field at FPM exit pupil?
    output_field_rootname = ''  # rootname of FPM exit pupil field file (must set end_at_fpm_exit_pupil=1)
    use_lyot_stop = 1           # use Lyot stop? 1 or 0
    lyot_x_shift_pupdiam = 0    # X,Y shear of Lyot stop mask; normalized relative to pupil diameter
    lyot_y_shift_pupdiam = 0  
    lyot_x_shift_m = 0          # X,Y shear of Lyot stop mask in meters
    lyot_y_shift_m = 0  
    use_field_stop = 1          # use field stop (HLC)? 1 or 0
    field_stop_radius_lam0 = 0  # field stop radius in lambda0/D (HLC or SPC-wide mask only)
    field_stop_x_offset = 0     # field stop offset in lambda0/D
    field_stop_y_offset = 0
    field_stop_x_offset_m = 0   # field stop offset in meters
    field_stop_y_offset_m = 0
    use_pupil_lens = 0          # use pupil imaging lens? 0 or 1
    use_defocus_lens = 0        # use defocusing lens? Options are 1, 2, 3, 4, corresponding to +18.0, +9.0, -4.0, -8.0 waves P-V @ 550 nm 
    defocus = 0                 # instead of specific lens, defocus in waves P-V @ 550 nm (-8.7 to 42.0 waves)
    final_sampling_m = 0        # final sampling in meters (overrides final_sampling_lam0)
    final_sampling_lam0 = 0     # final sampling in lambda0/D
    output_dim = output_dim0    # dimension of output in pixels (overrides output_dim0)

    if 'PASSVALUE' in locals():
        if 'use_fpm' in PASSVALUE: use_fpm = PASSVALUE['use_fpm']
        if 'cor_type' in PASSVALUE: cor_type = PASSVALUE['cor_type']

    is_spc = False
    is_hlc = False

    if cor_type == 'hlc':
        is_hlc = True
        file_directory = data_dir + '/hlc_20190210/'         # must have trailing "/"
        prefix = file_directory + 'run461_'
        pupil_diam_pix = 309.0
        pupil_file = prefix + 'pupil_rotated.fits'
        lyot_stop_file = prefix + 'lyot.fits'
        lambda0_m = 0.575e-6
        lam_occ = [     5.4625e-07, 5.49444444444e-07, 5.52638888889e-07, 5.534375e-07, 5.55833333333e-07, 5.59027777778e-07, 5.60625e-07, 
                        5.62222222222e-07, 5.65416666667e-07, 5.678125e-07, 5.68611111111e-07, 5.71805555556e-07, 5.75e-07, 5.78194444444e-07, 
                        5.81388888889e-07, 5.821875e-07, 5.84583333333e-07, 5.87777777778e-07, 5.89375e-07, 5.90972222222e-07, 5.94166666667e-07, 
                        5.965625e-07, 5.97361111111e-07, 6.00555555556e-07, 6.0375e-07 ]
        lam_occs = [    '5.4625e-07', '5.49444444444e-07', '5.52638888889e-07', '5.534375e-07', '5.55833333333e-07', '5.59027777778e-07', 
                        '5.60625e-07', '5.62222222222e-07', '5.65416666667e-07', '5.678125e-07', '5.68611111111e-07', '5.71805555556e-07', 
                        '5.75e-07', '5.78194444444e-07', '5.81388888889e-07', '5.821875e-07', '5.84583333333e-07', '5.87777777778e-07', 
                        '5.89375e-07', '5.90972222222e-07', '5.94166666667e-07', '5.965625e-07', '5.97361111111e-07', '6.00555555556e-07', '6.0375e-07' ]
        lam_occs = [ prefix + 'occ_lam' + s + 'theta6.69polp_' for s in lam_occs ]
        # find nearest matching FPM wavelength
        wlam = (np.abs(lambda_m-np.array(lam_occ))).argmin()
        occulter_file_r = lam_occs[wlam] + 'real.fits'
        occulter_file_i = lam_occs[wlam] + 'imag.fits'
        n_default = 1024                # gridsize in non-critical areas
        if use_fpm == 1:
            n_to_fpm = 2048
        else:
            n_to_fpm = 1024
        n_from_lyotstop = 1024
        field_stop_radius_lam0 = 9.0
    elif cor_type == 'hlc_erkin':
        is_hlc = True
        file_directory = data_dir + '/hlc_20190206_v3/'         # must have trailing "/"
        prefix = file_directory + 'dsn17d_run2_pup310_fpm2048_'
        pupil_diam_pix = 310.0
        pupil_file = prefix + 'pupil.fits'
        lyot_stop_file = prefix + 'lyot.fits'
        lambda0_m = 0.575e-6
        lam_occ = [     5.4625e-07, 5.4944e-07, 5.5264e-07, 5.5583e-07, 5.5903e-07, 5.6222e-07, 5.6542e-07,
                        5.6861e-07, 5.7181e-07, 5.75e-07, 5.7819e-07, 5.8139e-07, 5.8458e-07, 5.8778e-07,
                        5.9097e-07, 5.9417e-07, 5.9736e-07, 6.0056e-07, 6.0375e-07 ]
        lam_occs =  [   '5.4625e-07', '5.4944e-07', '5.5264e-07', '5.5583e-07', '5.5903e-07', '5.6222e-07', '5.6542e-07',
                        '5.6861e-07', '5.7181e-07', '5.75e-07', '5.7819e-07', '5.8139e-07', '5.8458e-07', '5.8778e-07',
                        '5.9097e-07', '5.9417e-07', '5.9736e-07', '6.0056e-07', '6.0375e-07' ]
        lam_occs = [ prefix + 'occ_lam' + s + 'theta6.69pols_' for s in lam_occs ]
        # find nearest matching FPM wavelength
        wlam = (np.abs(lambda_m-np.array(lam_occ))).argmin()
        occulter_file_r = lam_occs[wlam] + 'real_rotated.fits'
        occulter_file_i = lam_occs[wlam] + 'imag_rotated.fits'
        n_default = 1024                # gridsize in non-critical areas
        if use_fpm == 1:
            n_to_fpm = 2048
        else:
            n_to_fpm = 1024
        n_from_lyotstop = 1024
        field_stop_radius_lam0 = 9.0
    elif cor_type == 'spc-ifs_short' or cor_type == 'spc-ifs_long' or cor_type == 'spc-spec_short' or cor_type == 'spc-spec_long':
        is_spc = True
        file_dir = data_dir + '/spc_20190130/' # must have trailing "/"
        pupil_diam_pix = 1000.0
        pupil_file = file_dir + 'pupil_SPC-20190130_rotated.fits'
        pupil_mask_file = file_dir + 'SPM_SPC-20190130.fits'
        fpm_file = file_dir + 'fpm_0.05lamdivD.fits'
        fpm_sampling = 0.05         # sampling in fpm_sampling_lambda_m/D of FPM mask 
        if cor_type == 'spc-ifs_short' or cor_type == 'spc-spec_short':
            fpm_sampling_lambda_m = 0.66e-6
            lambda0_m = 0.66e-6
        else:
            fpm_sampling_lambda_m = 0.73e-6
            lambda0_m = 0.73e-6     # FPM scaled for this central wavelength
        lyot_stop_file = file_dir + 'LS_SPC-20190130.fits'
        n_default = 2048            # gridsize in non-critical areas
        n_to_fpm = 2048             # gridsize to/from FPM
        n_mft = 1400                # gridsize to FPM (propagation to/from FPM handled by MFT)
        n_from_lyotstop = 4096
    elif cor_type == 'spc-wide':
        is_spc = True
        file_dir = data_dir + '/spc_20181220/' # must have trailing "/"
        pupil_diam_pix = 1000.0
        pupil_file = file_dir + 'pupil_SPC-20181220_1k_rotated.fits'
        pupil_mask_file = file_dir + 'SPM_SPC-20181220_1000_rounded9_gray.fits'
        fpm_file = file_dir + 'fpm_0.05lamdivD.fits'
        fpm_sampling = 0.05    # sampling in lambda0/D of FPM mask 
        fpm_sampling_lambda_m = 0.825e-6
        lambda0_m = 0.825e-6        # FPM scaled for this central wavelength
        lyot_stop_file = file_dir + 'LS_SPC-20181220_1k.fits'
        n_default = 2048            # gridsize in non-critical areas
        n_to_fpm = 2048             # gridsize to/from FPM
        n_mft = 1400              
        n_from_lyotstop = 4096
    elif cor_type == 'none': 
        file_directory = data_dir + '/hlc_20190210/'         # must have trailing "/"
        prefix = file_directory + 'run461_'
        pupil_diam_pix = 309.0
        pupil_file = prefix + 'pupil_rotated.fits'
        lambda0_m = 0.575e-6
        use_fpm = 0
        use_lyot_stop = 0
        use_field_stop = 0
        n_default = 1024
        n_to_fpm = 1024
        n_from_lyotstop = 1024
    else:
        raise Exception('ERROR: Unsupported cor_type: '+cor_type)

    if 'PASSVALUE' in locals():
        if 'lam0' in PASSVALUE: lamba0_m = PASSVALUE['lam0'] * 1.0e-6
        if 'lambda0_m' in PASSVALUE: lambda0_m = PASSVALUE['lambda0_m']
        mas_per_lamD = lambda0_m * 360.0 * 3600.0 / (2 * np.pi * 2.363) * 1000    # mas per lambda0/D
        if 'source_x_offset' in PASSVALUE: source_x_offset = PASSVALUE['source_x_offset']
        if 'source_y_offset' in PASSVALUE: source_y_offset = PASSVALUE['source_y_offset']
        if 'source_x_offset_mas' in PASSVALUE: source_x_offset = PASSVALUE['source_x_offset_mas'] / mas_per_lamD
        if 'source_y_offset_mas' in PASSVALUE: source_y_offset = PASSVALUE['source_y_offset_mas'] / mas_per_lamD
        if 'use_errors' in PASSVALUE: use_errors = PASSVALUE['use_errors']
        if 'polaxis' in PASSVALUE: polaxis = PASSVALUE['polaxis'] 
        if 'zindex' in PASSVALUE: zindex = np.array( PASSVALUE['zindex'] )
        if 'zval_m' in PASSVALUE: zval_m = np.array( PASSVALUE['zval_m'] )
        if 'end_at_fsm' in PASSVALUE: end_at_fsm = PASSVALUE['end_at_fsm']
        if 'cgi_x_shift_pupdiam' in PASSVALUE: cgi_x_shift_pupdiam = PASSVALUE['cgi_x_shift_pupdiam']
        if 'cgi_y_shift_pupdiam' in PASSVALUE: cgi_y_shift_pupdiam = PASSVALUE['cgi_y_shift_pupdiam']
        if 'cgi_x_shift_m' in PASSVALUE: cgi_x_shift_m = PASSVALUE['cgi_x_shift_m']
        if 'cgi_y_shift_m' in PASSVALUE: cgi_y_shift_m = PASSVALUE['cgi_y_shift_m']
        if 'fsm_x_offset' in PASSVALUE: fsm_x_offset = PASSVALUE['fsm_x_offset']
        if 'fsm_y_offset' in PASSVALUE: fsm_y_offset = PASSVALUE['fsm_y_offset']
        if 'fsm_x_offset_mas' in PASSVALUE: fsm_x_offset = PASSVALUE['fsm_x_offset_mas'] / mas_per_lamD
        if 'fsm_y_offset_mas' in PASSVALUE: fsm_y_offset = PASSVALUE['fsm_y_offset_mas'] / mas_per_lamD
        if 'focm_z_shift_m' in PASSVALUE: focm_z_shift_m = PASSVALUE['focm_z_shift_m']
        if 'use_hlc_dm_patterns' in PASSVALUE: use_hlc_dm_patterns = PASSVALUE['use_hlc_dm_patterns']
        if 'use_dm1' in PASSVALUE: use_dm1 = PASSVALUE['use_dm1'] 
        if 'dm1_m' in PASSVALUE: dm1_m = PASSVALUE['dm1_m']
        if 'dm1_xc_act' in PASSVALUE: dm1_xc_act = PASSVALUE['dm1_xc_act']
        if 'dm1_yc_act' in PASSVALUE: dm1_yc_act = PASSVALUE['dm1_yc_act']
        if 'dm1_xtilt_deg' in PASSVALUE: dm1_xtilt_deg = PASSVALUE['dm1_xtilt_deg']
        if 'dm1_ytilt_deg' in PASSVALUE: dm1_ytilt_deg = PASSVALUE['dm1_ytilt_deg']
        if 'dm1_ztilt_deg' in PASSVALUE: dm1_ztilt_deg = PASSVALUE['dm1_ztilt_deg']
        if 'use_dm2' in PASSVALUE: use_dm2 = PASSVALUE['use_dm2']
        if 'dm2_m' in PASSVALUE: dm2_m = PASSVALUE['dm2_m']
        if 'dm2_xc_act' in PASSVALUE: dm2_xc_act = PASSVALUE['dm2_xc_act']
        if 'dm2_yc_act' in PASSVALUE: dm2_yc_act = PASSVALUE['dm2_yc_act']
        if 'dm2_xtilt_deg' in PASSVALUE: dm2_xtilt_deg = PASSVALUE['dm2_xtilt_deg']
        if 'dm2_ytilt_deg' in PASSVALUE: dm2_ytilt_deg = PASSVALUE['dm2_ytilt_deg']
        if 'dm2_ztilt_deg' in PASSVALUE: dm2_ztilt_deg = PASSVALUE['dm2_ztilt_deg']
        if 'use_pupil_mask' in PASSVALUE: use_pupil_mask = PASSVALUE['use_pupil_mask']
        if 'mask_x_shift_pupdiam' in PASSVALUE: mask_x_shift_pupdiam = PASSVALUE['mask_x_shift_pupdiam']
        if 'mask_y_shift_pupdiam' in PASSVALUE: mask_y_shift_pupdiam = PASSVALUE['mask_y_shift_pupdiam']
        if 'mask_x_shift_m' in PASSVALUE: mask_x_shift_m = PASSVALUE['mask_x_shift_m']
        if 'mask_y_shift_m' in PASSVALUE: mask_y_shift_m = PASSVALUE['mask_y_shift_m']
        if 'fpm_x_offset' in PASSVALUE: fpm_x_offset = PASSVALUE['fpm_x_offset']
        if 'fpm_y_offset' in PASSVALUE: fpm_y_offset = PASSVALUE['fpm_y_offset']
        if 'fpm_x_offset_m' in PASSVALUE: fpm_x_offset_m = PASSVALUE['fpm_x_offset_m']
        if 'fpm_y_offset_m' in PASSVALUE: fpm_y_offset_m = PASSVALUE['fpm_y_offset_m']
        if 'fpm_z_shift_m' in PASSVALUE: fpm_z_shift_m = PASSVALUE['fpm_z_shift_m']
        if 'pinhole_diam_m' in PASSVALUE: pinhole_diam_m = PASSVALUE['pinhole_diam_m']
        if 'end_at_fpm_exit_pupil' in PASSVALUE: end_at_fpm_exit_pupil = PASSVALUE['end_at_fpm_exit_pupil']
        if 'output_field_rootname' in PASSVALUE: output_field_rootname = PASSVALUE['output_field_rootname']
        if 'use_lyot_stop' in PASSVALUE: use_lyot_stop = PASSVALUE['use_lyot_stop']
        if 'lyot_x_shift_pupdiam' in PASSVALUE: lyot_x_shift_pupdiam = PASSVALUE['lyot_x_shift_pupdiam']
        if 'lyot_y_shift_pupdiam' in PASSVALUE: lyot_y_shift_pupdiam = PASSVALUE['lyot_y_shift_pupdiam']
        if 'lyot_x_shift_m' in PASSVALUE: lyot_x_shift_m = PASSVALUE['lyot_x_shift_m']
        if 'lyot_y_shift_m' in PASSVALUE: lyot_y_shift_m = PASSVALUE['lyot_y_shift_m']
        if 'use_field_stop' in PASSVALUE: use_field_stop = PASSVALUE['use_field_stop']
        if 'field_stop_x_offset' in PASSVALUE: field_stop_x_offset = PASSVALUE['field_stop_x_offset']
        if 'field_stop_y_offset' in PASSVALUE: field_stop_y_offset = PASSVALUE['field_stop_y_offset']
        if 'field_stop_x_offset_m' in PASSVALUE: field_stop_x_offset_m = PASSVALUE['field_stop_x_offset_m']
        if 'field_stop_y_offset_m' in PASSVALUE: field_stop_y_offset_m = PASSVALUE['field_stop_y_offset_m']
        if 'use_pupil_lens' in PASSVALUE: use_pupil_lens = PASSVALUE['use_pupil_lens']
        if 'use_defocus_lens' in PASSVALUE: use_defocus_lens = PASSVALUE['use_defocus_lens']
        if 'defocus' in PASSVALUE: defocus = PASSVALUE['defocus']
        if 'output_dim' in PASSVALUE: output_dim = PASSVALUE['output_dim']
        if 'final_sampling_m' in PASSVALUE: final_sampling_m = PASSVALUE['final_sampling_m']
        if 'final_sampling_lam0' in PASSVALUE: final_sampling_lam0 = PASSVALUE['final_sampling_lam0']


    diam = 2.3633372
    fl_pri = 2.83459423440 * 1.0013
    d_pri_sec = 2.285150515460035
    d_focus_sec = d_pri_sec - fl_pri
    fl_sec = -0.653933011 * 1.0004095
    d_sec_focus = 3.580188916677103
    diam_sec = 0.58166
    d_sec_fold1 = 2.993753476654728
    d_fold1_focus = 0.586435440022375
    diam_fold1 = 0.09
    d_fold1_m3 = 1.680935841598811
    fl_m3 = 0.430216463069001
    d_focus_m3 = 1.094500401576436
    d_m3_pupil = 0.469156807701977
    d_m3_focus = 0.708841602661368
    diam_m3 = 0.2
    d_m3_m4 = 0.943514749358944
    fl_m4 = 0.116239114833590
    d_focus_m4 = 0.234673014520402
    d_m4_pupil = 0.474357941656967
    d_m4_focus = 0.230324117970585
    diam_m4 = 0.07
    d_m4_m5 = 0.429145636743193
    d_m5_focus = 0.198821518772608
    fl_m5 = 0.198821518772608
    d_m5_pupil = 0.716529242882632
    diam_m5 = 0.07
    d_m5_fold2 = 0.351125431220770
    diam_fold2 = 0.06
    d_fold2_fsm = 0.365403811661862
    d_fsm_oap1 = 0.354826767220001
    fl_oap1 = 0.503331895563883
    diam_oap1 = 0.06
    d_oap1_focm = 0.768005607094041
    d_focm_oap2 = 0.314483210543378
    fl_oap2 = 0.579156922073536
    diam_oap2 = 0.06
    d_oap2_dm1 = 0.775775726154228
    d_dm1_dm2 = 1.0
    d_dm2_oap3 = 0.394833855161549
    fl_oap3 = 1.217276467668519
    diam_oap3 = 0.06
    d_oap3_fold3 = 0.505329955078121
    diam_fold3 = 0.06
    d_fold3_oap4 = 1.158897671642761
    fl_oap4 = 0.446951159052363
    diam_oap4 = 0.06
    d_oap4_pupilmask = 0.423013568764728
    d_pupilmask_oap5 = 0.408810648253099
    fl_oap5 =  0.548189351937178
    diam_oap5 = 0.06
    d_oap5_fpm = 0.548189083164429
    d_fpm_oap6 = 0.548189083164429
    fl_oap6 = 0.548189083164429
    diam_oap6 = 0.06
    d_oap6_lyotstop = 0.687567667550736
    d_lyotstop_oap7 = 0.401748843470518
    fl_oap7 = 0.708251083480054
    diam_oap7 = 0.06
    d_oap7_fieldstop = 0.708251083480054  
    d_fieldstop_oap8 = 0.210985967281651
    fl_oap8 = 0.210985967281651
    diam_oap8 = 0.06
    d_oap8_pupil = 0.238185804200797
    d_oap8_filter = 0.368452268225530
    diam_filter = 0.01
    d_filter_lens = 0.170799548215162
    fl_lens = 0.246017378417573 + 0.050001306014153
    diam_lens = 0.01
    d_lens_fold4 = 0.246017378417573
    diam_fold4 = 0.02
    d_fold4_image = 0.050001578514650
    fl_pupillens = 0.149260576823040      


    n = n_default        # start off with less padding
 
    wavefront = proper.prop_begin( diam, lambda_m, n, float(pupil_diam_pix)/n )
    pupil = proper.prop_fits_read( pupil_file )
    proper.prop_multiply( wavefront, trim(pupil,n) )
    pupil = 0
    if polaxis != 0: polmap( wavefront, polfile, pupil_diam_pix, polaxis )
    proper.prop_define_entrance( wavefront )
    proper.prop_lens( wavefront, fl_pri )
    if source_x_offset != 0 or source_y_offset != 0:
        # compute tilted wavefront to offset source by xoffset,yoffset lambda0_m/D
        xtilt_lam = -source_x_offset * lambda0_m / lambda_m
        ytilt_lam = -source_y_offset * lambda0_m / lambda_m
        x = np.tile( (np.arange(n)-n//2)/(pupil_diam_pix/2.0), (n,1) )
        y = np.transpose(x)
        proper.prop_multiply( wavefront, np.exp(complex(0,1) * np.pi * (xtilt_lam * x + ytilt_lam * y)) )
        x = 0
        y = 0
    if zindex[0] != 0: proper.prop_zernikes( wavefront, zindex, zval_m )
    if use_errors != 0: 
        proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_PRIMARY_phase_error_V1.0.fits', WAVEFRONT=True )
        proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_GROUND_TO_ORBIT_4.2X_phase_error_V1.0.fits', WAVEFRONT=True )
    
    proper.prop_propagate( wavefront, d_pri_sec, 'secondary' )
    proper.prop_lens( wavefront, fl_sec )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_SECONDARY_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_sec/2.0 )  

    proper.prop_propagate( wavefront, d_sec_fold1, 'FOLD_1' )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_FOLD1_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_fold1/2.0 ) 

    proper.prop_propagate( wavefront, d_fold1_m3, 'M3' )
    proper.prop_lens( wavefront, fl_m3 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_M3_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_m3/2.0 ) 

    proper.prop_propagate( wavefront, d_m3_m4, 'M4' )
    proper.prop_lens( wavefront, fl_m4 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_M4_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_m4/2.0 ) 
 
    proper.prop_propagate( wavefront, d_m4_m5, 'M5' )
    proper.prop_lens( wavefront, fl_m5 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_M5_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_m5/2.0 ) 
 
    proper.prop_propagate( wavefront, d_m5_fold2, 'FOLD_2' )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_FOLD2_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_fold2/2.0 )

    proper.prop_propagate( wavefront, d_fold2_fsm, 'FSM' )
    if end_at_fsm == 1:
        (wavefront, sampling_m) = proper.prop_end( wavefront, NOABS=True )
        wavefront = trim(wavefront, n)
        return wavefront, sampling_m
    if cgi_x_shift_pupdiam != 0 or cgi_y_shift_pupdiam != 0 or cgi_x_shift_m != 0 or cgi_y_shift_m != 0:    # bulk coronagraph pupil shear
        # FFT the field, apply a tilt, FFT back
        if cgi_x_shift_pupdiam != 0 or cgi_y_shift_pupdiam != 0:
            # offsets are normalized to pupil diameter
            xt = -cgi_x_shift_pupdiam * pupil_diam_pix * float(pupil_diam_pix)/n 
            yt = -cgi_y_shift_pupdiam * pupil_diam_pix * float(pupil_diam_pix)/n
        else:
            # offsets are meters
            d_m = proper.prop_get_sampling(wavefront) 
            xt = -cgi_x_shift_m / d_m * float(pupil_diam_pix)/n 
            yt = -cgi_y_shift_m / d_m * float(pupil_diam_pix)/n 
        x = np.tile( (np.arange(n)-n//2) / (pupil_diam_pix/2.0), (n,1) )
        y = np.transpose(x)
        tilt = complex(0,1) * np.pi * (x*xt + y*yt)
        x = 0
        y = 0
        wavefront0 = proper.prop_get_wavefront(wavefront)
        wavefront0 = ffts( wavefront0, -1 )
        wavefront0 *= np.exp(tilt)
        wavefront0 = ffts( wavefront0, 1 )
        tilt = 0
        wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
        wavefront0 = 0
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_FSM_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_fsm/2.0 )
    if ( fsm_x_offset != 0.0 or fsm_y_offset != 0.0 ):
        # compute tilted wavefront to offset source by fsm_x_offset,fsm_y_offset lambda0_m/D
        xtilt_lam = fsm_x_offset * lambda0_m / lambda_m
        ytilt_lam = fsm_y_offset * lambda0_m / lambda_m
        x = np.tile( (np.arange(n)-n//2) / (pupil_diam_pix/2.0), (n,1) )
        y = np.transpose(x)
        proper.prop_multiply( wavefront, np.exp(complex(0,1) * np.pi * (xtilt_lam * x + ytilt_lam * y)) )
        x = 0
        y = 0

    proper.prop_propagate( wavefront, d_fsm_oap1, 'OAP1' )
    proper.prop_lens( wavefront, fl_oap1 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_OAP1_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap1/2.0 )  

    proper.prop_propagate( wavefront, d_oap1_focm+focm_z_shift_m, 'FOCM' )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_FOCM_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_focm/2.0 )

    proper.prop_propagate( wavefront, d_focm_oap2+focm_z_shift_m, 'OAP2' )
    proper.prop_lens( wavefront, fl_oap2 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_OAP2_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap2/2.0 )  

    proper.prop_propagate( wavefront, d_oap2_dm1, 'DM1' )
    if use_dm1 != 0: proper.prop_dm( wavefront, dm1_m, dm1_xc_act, dm1_yc_act, dm_sampling_m, XTILT=dm1_xtilt_deg, YTILT=dm1_ytilt_deg, ZTILT=dm1_ztilt_deg )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_DM1_phase_error_V1.0.fits', WAVEFRONT=True )
    if is_hlc == True and use_hlc_dm_patterns == 1:
        dm1wfe = proper.prop_fits_read( prefix+'dm1wfe.fits' )
        proper.prop_add_phase( wavefront, trim(dm1wfe, n) )
        dm1wfe = 0

    proper.prop_propagate( wavefront, d_dm1_dm2, 'DM2' )
    if use_dm2 == 1: proper.prop_dm( wavefront, dm2_m, dm2_xc_act, dm2_yc_act, dm_sampling_m, XTILT=dm2_xtilt_deg, YTILT=dm2_ytilt_deg, ZTILT=dm2_ztilt_deg )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_DM2_phase_error_V1.0.fits', WAVEFRONT=True )
    if is_hlc == True:
        if use_hlc_dm_patterns == 1:
            dm2wfe = proper.prop_fits_read( prefix+'dm2wfe.fits' )
            proper.prop_add_phase( wavefront, trim(dm2wfe, n) )
            dm2wfe = 0
        dm2mask = proper.prop_fits_read( prefix+'dm2mask.fits' )
        proper.prop_multiply( wavefront, trim(dm2mask, n) )
        dm2mask = 0

    proper.prop_propagate( wavefront, d_dm2_oap3, 'OAP3' )
    proper.prop_lens( wavefront, fl_oap3 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_OAP3_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap3/2.0 ) 

    proper.prop_propagate( wavefront, d_oap3_fold3, 'FOLD_3' )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_FOLD3_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_fold3/2.0 )

    proper.prop_propagate( wavefront, d_fold3_oap4, 'OAP4' )
    proper.prop_lens( wavefront, fl_oap4 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_OAP4_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap4/2.0 )

    proper.prop_propagate( wavefront, d_oap4_pupilmask, 'PUPIL_MASK' )    # flat/reflective shaped pupil 
    if is_spc == True and use_pupil_mask != 0:
        pupil_mask = proper.prop_fits_read( pupil_mask_file )
        pupil_mask = trim( pupil_mask, n )
        if mask_x_shift_pupdiam != 0 or mask_y_shift_pupdiam != 0 or mask_x_shift_m != 0 or mask_y_shift_m != 0:
            # shift SP mask by FFTing it, applying tilt, and FFTing back 
            if mask_x_shift_pupdiam != 0 or mask_y_shift_pupdiam != 0:
                # offsets are normalized to pupil diameter
                xt = -mask_x_shift_pupdiam * pupil_diam_pix * float(pupil_diam_pix)/n
                yt = -mask_y_shift_pupdiam * pupil_diam_pix * float(pupil_diam_pix)/n
            else:
                d_m = proper.prop_get_sampling(wavefront)
                xt = -mask_x_shift_m / d_m * float(pupil_diam_pix)/n
                yt = -mask_y_shift_m / d_m * float(pupil_diam_pix)/n
            x = np.tile( (np.arange(n)-n//2) / (pupil_diam_pix/2.0), (n,1) )
            y = np.transpose(x)
            tilt = complex(0,1) * np.pi * (x*xt + y*yt)
            x = 0
            y = 0
            pupil_mask = ffts( pupil_mask, -1 )
            pupil_mask *= np.exp(tilt)
            pupil_mask = ffts( pupil_mask, 1 )
            pupil_mask = pupil_mask.real
            tilt = 0
        proper.prop_multiply( wavefront, pupil_mask )
        pupil_mask = 0
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_PUPILMASK_phase_error_V1.0.fits', WAVEFRONT=True )
    # while at a pupil, use more padding to provide 2x better sampling at FPM
    diam = 2 * proper.prop_get_beamradius(wavefront)
    (wavefront, dx) = proper.prop_end( wavefront, NOABS=True )
    n = n_to_fpm
    wavefront0 = trim(wavefront,n)
    wavefront = proper.prop_begin( diam, lambda_m, n, float(pupil_diam_pix)/n )
    wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
    wavefront0 = 0

    proper.prop_propagate( wavefront, d_pupilmask_oap5, 'OAP5' )
    proper.prop_lens( wavefront, fl_oap5 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_OAP5_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap5/2.0 )  

    proper.prop_propagate( wavefront, d_oap5_fpm+fpm_z_shift_m, 'FPM', TO_PLANE=True )
    if use_fpm == 1:
        if fpm_x_offset != 0 or fpm_y_offset != 0 or fpm_x_offset_m != 0 or fpm_y_offset_m != 0:
            # To shift FPM, FFT field to pupil, apply tilt, FFT back to focus, 
            # apply FPM, FFT to pupil, take out tilt, FFT back to focus 
            if fpm_x_offset != 0 or fpm_y_offset != 0:
                # shifts are specified in lambda0/D
                x_offset_lamD = fpm_x_offset * lambda0_m / lambda_m
                y_offset_lamD = fpm_y_offset * lambda0_m / lambda_m
            else:
                d_m = proper.prop_get_sampling(wavefront)
                x_offset_lamD = fpm_x_offset_m / d_m * float(pupil_diam_pix)/n
                y_offset_lamD = fpm_y_offset_m / d_m * float(pupil_diam_pix)/n
            x = np.tile( (np.arange(n)-n//2) / (pupil_diam_pix/2.0), (n,1) )
            y = np.transpose(x)
            tilt = complex(0,1) * np.pi * (x*x_offset_lamD + y*y_offset_lamD)
            x = 0
            y = 0
            wavefront0 = proper.prop_get_wavefront(wavefront)
            wavefront0 = ffts( wavefront0, -1 )
            wavefront0 *= np.exp(tilt)
            wavefront0 = ffts( wavefront0, 1 )
            wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
            wavefront0 = 0
        if is_hlc == True:
            occ_r = proper.prop_fits_read( occulter_file_r )
            occ_i = proper.prop_fits_read( occulter_file_i )
            occ = np.array( occ_r + 1j * occ_i, dtype=np.complex128 )
            proper.prop_multiply( wavefront, trim(occ,n) )
            occ_r = 0
            occ_i = 0
            occ = 0
        elif is_spc == True:
            # super-sample FPM
            wavefront0 = proper.prop_get_wavefront(wavefront)
            wavefront0 = ffts( wavefront0, 1 )                # to virtual pupil
            wavefront0 = trim(wavefront0, n_mft)
            fpm = proper.prop_fits_read( fpm_file )
            nfpm = fpm.shape[1]
            fpm_sampling_lam = fpm_sampling * fpm_sampling_lambda_m / lambda_m
            wavefront0 = mft2(wavefront0, fpm_sampling_lam, pupil_diam_pix, nfpm, -1)   # MFT to highly-sampled focal plane
            wavefront0 *= fpm
            fpm = 0
            wavefront0 = mft2(wavefront0, fpm_sampling_lam, pupil_diam_pix, n, +1)  # MFT to virtual pupil 
            wavefront0 = ffts( wavefront0, -1 )    # back to normally-sampled focal plane
            wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
            wavefront0 = 0
        if fpm_x_offset != 0 or fpm_y_offset != 0 or fpm_x_offset_m != 0 or fpm_y_offset_m != 0:
            wavefront0 = proper.prop_get_wavefront(wavefront)
            wavefront0 = ffts( wavefront0, -1 )
            wavefront0 *= np.exp(-tilt)
            wavefront0 = ffts( wavefront0, 1 )
            wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
            wavefront0 = 0
            tilt = 0
    if pinhole_diam_m != 0:
        # "pinhole_diam_m" is pinhole diameter in meters
        dx_m = proper.prop_get_sampling(wavefront)
        dx_pinhole_diam_m = pinhole_diam_m / 101.0        # 101 samples across pinhole
        n_out = 105
        m_per_lamD = dx_m * n / float(pupil_diam_pix)        # current focal plane sampling in lambda_m/D
        dx_pinhole_lamD = dx_pinhole_diam_m / m_per_lamD    # pinhole sampling in lambda_m/D
        n_in = int(round(pupil_diam_pix * 1.2))
        wavefront0 = proper.prop_get_wavefront(wavefront)
        wavefront0 = ffts( wavefront0, +1 )            # to virtual pupil
        wavefront0 = trim(wavefront0, n_in)
        m = dx_pinhole_lamD * n_in * float(n_out) / pupil_diam_pix
        wavefront0 = mft2( wavefront0, dx_pinhole_lamD, pupil_diam_pix, n_out, -1)        # MFT to highly-sampled focal plane
        p = (radius(n_out)*dx_pinhole_diam_m) <= (pinhole_diam_m/2.0)
        p = p.astype(np.int)
        wavefront0 *= p
        p = 0
        wavefront0 = mft2( wavefront0, dx_pinhole_lamD, pupil_diam_pix, n, +1)            # MFT back to virtual pupil
        wavefront0 = ffts( wavefront0, -1 )            # back to normally-sampled focal plane
        wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
        wavefront0 = 0

    proper.prop_propagate( wavefront, d_fpm_oap6-fpm_z_shift_m, 'OAP6' )
    proper.prop_lens( wavefront, fl_oap6 )
    if use_errors != 0 and end_at_fpm_exit_pupil == 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_OAP6_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap6/2.0 )  

    proper.prop_propagate( wavefront, d_oap6_lyotstop, 'LYOT_STOP' )
    # while at a pupil, switch back to less padding
    diam = 2 * proper.prop_get_beamradius(wavefront)
    (wavefront, dx) = proper.prop_end( wavefront, NOABS=True )
    n = n_from_lyotstop
    wavefront = trim(wavefront, n)
    if output_field_rootname != '':
        lams = format( lambda_m*1e6, "6.4f" )
        pols = format( int(round(polaxis)) )
        hdu = pyfits.PrimaryHDU()
        hdu.data = np.real(wavefront)
        hdu.writeto( output_field_rootname+'_'+lams+'um_'+pols+'_real.fits', overwrite=True )
        hdu = pyfits.PrimaryHDU()
        hdu.data = np.imag(wavefront)
        hdu.writeto( output_field_rootname+'_'+lams+'um_'+pols+'_imag.fits', overwrite=True )
    if end_at_fpm_exit_pupil == 1:
        return wavefront, dx
    wavefront0 = wavefront.copy()
    wavefront = 0
    wavefront = proper.prop_begin( diam, lambda_m, n, float(pupil_diam_pix)/n )
    wavefront.wfarr[:,:] = proper.prop_shift_center(wavefront0)
    wavefront0 = 0

    if use_lyot_stop != 0:
        lyot = proper.prop_fits_read( lyot_stop_file )
        lyot = trim( lyot, n )
        if lyot_x_shift_pupdiam != 0 or lyot_y_shift_pupdiam != 0 or lyot_x_shift_m != 0 or lyot_y_shift_m != 0:
            # apply shift to lyot stop by FFTing the stop, applying a tilt, and FFTing back
            if lyot_x_shift_pupdiam != 0 or lyot_y_shift_pupdiam != 0:
                # offsets are normalized to pupil diameter
                xt = -lyot_x_shift_pupdiam * pupil_diam_pix * float(pupil_diam_pix)/n
                yt = -lyot_y_shift_pupdiam * pupil_diam_pix * float(pupil_diam_pix)/n
            else:
                d_m = proper.prop_get_sampling(wavefront)
                xt = -lyot_x_shift_m / d_m * float(pupil_diam_pix)/n
                yt = -lyot_y_shift_m / d_m * float(pupil_diam_pix)/n
            x = np.tile( (np.arange(n)-n//2) / (pupil_diam_pix/2.0), (n,1) )
            y = np.transpose(x)
            tilt = complex(0,1) * np.pi * (x*xt + y*yt)
            x = 0
            y = 0
            lyot = ffts(lyot,-1)
            lyot *= np.exp(tilt)
            lyot = ffts( lyot, 1 )
            lyot = lyot.real
            tilt = 0
        proper.prop_multiply( wavefront, lyot )
        lyot = 0
    if use_pupil_lens != 0 or pinhole_diam_m != 0: proper.prop_circular_aperture( wavefront, 1.1, NORM=True )

    proper.prop_propagate( wavefront, d_lyotstop_oap7, 'OAP7' )
    proper.prop_lens( wavefront, fl_oap7 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_OAP7_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap7/2.0 )  

    proper.prop_propagate( wavefront, d_oap7_fieldstop, 'FIELD_STOP' )
    if use_field_stop != 0 and (cor_type == 'hlc' or cor_type == 'hlc_erkin'):
        sampling_lamD = float(pupil_diam_pix) / n      # sampling at focus in lambda_m/D
        stop_radius = field_stop_radius_lam0 / sampling_lamD * (lambda0_m/lambda_m) * proper.prop_get_sampling(wavefront)
        if field_stop_x_offset != 0 or field_stop_y_offset != 0:
            # convert offsets in lambda0/D to meters
            x_offset_lamD = field_stop_x_offset * lambda0_m / lambda_m
            y_offset_lamD = field_stop_y_offset * lambda0_m / lambda_m
            pupil_ratio = float(pupil_diam_pix) / n
            field_stop_x_offset_m = x_offset_lamD / pupil_ratio * proper.prop_get_sampling(wavefront)
            field_stop_y_offset_m = y_offset_lamD / pupil_ratio * proper.prop_get_sampling(wavefront)
        proper.prop_circular_aperture( wavefront, stop_radius, -field_stop_x_offset_m, -field_stop_y_offset_m )

    proper.prop_propagate( wavefront, d_fieldstop_oap8, 'OAP8' )
    proper.prop_lens( wavefront, fl_oap8 )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_OAP8_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_oap8/2.0 )  

    proper.prop_propagate( wavefront, d_oap8_filter, 'filter' )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_FILTER_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_filter/2.0 ) 

    proper.prop_propagate( wavefront, d_filter_lens, 'LENS' )
    if use_pupil_lens == 0 and use_defocus_lens == 0 and defocus == 0:
        # use imaging lens to create normal focus
        proper.prop_lens( wavefront, fl_lens )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_LENS_phase_error_V1.0.fits', WAVEFRONT=True )
    elif use_pupil_lens != 0:
        # use pupil imaging lens
        proper.prop_lens( wavefront, fl_pupillens )
        if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_PUPILLENS_phase_error_V1.0.fits', WAVEFRONT=True )
    else:
        # table is waves P-V @ 575 nm
        z4_pv_waves = np.array( [-9.0545,-8.5543,-8.3550,-8.0300,-7.54500,-7.03350,-6.03300,-5.03300,-4.02000,
                                 -2.51980,0.00000000,3.028000,4.95000,6.353600,8.030000,10.10500,12.06000,
                                 14.06000,20.26000,28.34000,40.77500,56.65700] )
        fl_defocus_lens = np.array( [5.09118,1.89323,1.54206,1.21198,0.914799,0.743569,0.567599,0.470213,0.406973,
                                     0.350755,0.29601868,0.260092,0.24516,0.236606,0.228181,0.219748,0.213278,
                                     0.207816,0.195536,0.185600,0.176629,0.169984] )
        # subtract ad-hoc function to make z4 vs f_length more accurately spline interpolatible
        f = fl_defocus_lens / 0.005
        f0 = 59.203738
        z4t = z4_pv_waves - (0.005*(f0-f-40))/f**2/0.575e-6
        if use_defocus_lens != 0: 
            # use one of 4 defocusing lenses
            defocus = np.array([ 18.0, 9.0, -4.0, -8.0 ])    # waves P-V @ 575 nm
            f = interp1d( z4_pv_waves, z4t, kind='cubic' )
            z4x = f( defocus )
            f = interp1d( z4t, fl_defocus_lens, kind='cubic' )
            lens_fl = f( z4x )
            proper.prop_lens( wavefront, lens_fl[use_defocus_lens-1] ) 
            if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_DEFOCUSLENS'+str(use_defocus_lens)+'_phase_error_V1.0.fits', WAVEFRONT=True )
            defocus = defocus[use_defocus_lens-1]
        else:
            # specify amount of defocus (P-V waves @ 575 nm)
            f = interp1d( z4_pv_waves, z4t, kind='cubic' )
            z4x = f( defocus )
            f = interp1d( z4t, fl_defocus_lens, kind='cubic' )
            lens_fl = f( z4x )
            proper.prop_lens( wavefront, lens_fl ) 
            if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_DEFOCUSLENS1_phase_error_V1.0.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_lens/2.0 ) 

    proper.prop_propagate( wavefront, d_lens_fold4, 'FOLD_4' )
    if use_errors != 0: proper.prop_errormap( wavefront, map_dir+'wfirst_phaseb_FOLD4_phase_error_V1.1.fits', WAVEFRONT=True )
    if use_aperture != 0: proper.prop_circular_aperture( wavefront, diam_fold4/2.0 ) 

    if defocus != 0 or use_defocus_lens != 0:
        if np.abs(defocus) <= 4:
            proper.prop_propagate( wavefront, d_fold4_image, 'IMAGE', TO_PLANE=True )
        else:
            proper.prop_propagate( wavefront, d_fold4_image, 'IMAGE' )
    else:
        proper.prop_propagate( wavefront, d_fold4_image, 'IMAGE' )

    (wavefront, sampling_m) = proper.prop_end( wavefront, NOABS=True )

    if final_sampling_lam0 != 0 or final_sampling_m != 0:
        if final_sampling_m != 0:
            mag = sampling_m / final_sampling_m
            sampling_m = final_sampling_m 
        else:
            mag = (float(pupil_diam_pix)/n) / final_sampling_lam0 * (lambda_m/lambda0_m)
            sampling_m = sampling_m / mag
        wavefront = proper.prop_magnify( wavefront, mag, output_dim, AMP_CONSERVE=True )
    else:
        wavefront = trim(wavefront, output_dim)

    return wavefront, sampling_m

