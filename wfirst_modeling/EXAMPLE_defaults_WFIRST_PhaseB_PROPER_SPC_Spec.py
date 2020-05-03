import numpy as np
import sys
#sys.path.append('../')
#sys.path.append('~/Repos/falco-python/data/WFIRST/PhaseB/')
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline

import falco


mp = falco.config.ModelParameters()
mp.compact = falco.config.Object()
mp.full = falco.config.Object()

#--Record Keeping
mp.SeriesNum = 1;
mp.TrialNum = 1;

#--Special Computational Settings
mp.flagMultiproc = True
mp.useGPU = False
mp.flagPlot = False

#--General
mp.centering = 'pixel';

#--Whether to include planet in the images
mp.planetFlag = False

#--Method of computing core throughput:
# - 'HMI' for energy within half-max isophote divided by energy at telescope pupil
# - 'EE' for encircled energy within a radius (mp.thput_radius) divided by energy at telescope pupil
mp.thput_metric = 'HMI'; 
mp.thput_radius = 0.7; #--photometric aperture radius [lambda_c/D]. Used ONLY for 'EE' method.
mp.thput_eval_x = 7; # x location [lambda_c/D] in dark hole at which to evaluate throughput
mp.thput_eval_y = 0; # y location [lambda_c/D] in dark hole at which to evaluate throughput

#--Where to shift the source to compute the intensity normalization value.
mp.source_x_offset_norm = 7;  # x location [lambda_c/D] in dark hole at which to compute intensity normalization
mp.source_y_offset_norm = 0;  # y location [lambda_c/D] in dark hole at which to compute intensity normalization

# %# Bandwidth and Wavelength Specs

mp.lambda0 = 730e-9;   #--Central wavelength of the whole spectral bandpass [meters]
mp.fracBW = 0.15;       #--fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
mp.Nsbp = 5;            #--Number of sub-bandpasses to divide the whole bandpass into for estimation and control
mp.Nwpsbp = 3;          #--Number of wavelengths to used to approximate an image in each sub-bandpass

# %# Wavefront Estimation

#--Estimator Options:
# - 'perfect' for exact numerical answer from full model
# - 'pwp-bp' for pairwise probing with batch process estimation
# - 'pwp-kf' for pairwise probing with Kalman filter [NOT TESTED YET]
# - 'pwp-iekf' for pairwise probing with iterated extended Kalman filter  [NOT AVAILABLE YET]
mp.estimator = 'perfect';

#--New variables for pairwise probing estimation:
mp.est = falco.config.Object()
mp.est.probe = falco.config.Object()
mp.est.probe.Npairs = 3;     # Number of pair-wise probe PAIRS to use.
mp.est.probe.whichDM = 1;    # Which DM # to use for probing. 1 or 2. Default is 1
mp.est.probe.radius = 12;   # Max x/y extent of probed region [actuators].
mp.est.probe.offsetX = 0;   # offset of probe center in x [actuators]. Use to avoid central obscurations.
mp.est.probe.offsetY = 14;    # offset of probe center in y [actuators]. Use to avoid central obscurations.
mp.est.probe.axis = 'alternate'     # which axis to have the phase discontinuity along [x or y or xy/alt/alternate]
mp.est.probe.gainFudge = 1;     # empirical fudge factor to make average probe amplitude match desired value.

### Wavefront Control: General

mp.ctrl = falco.config.Object()
mp.ctrl.flagUseModel = True #--Whether to perform a model-based (vs empirical) grid search for the controller

#--Threshold for culling weak actuators from the Jacobian:
mp.logGmin = -6 # 10^(mp.logGmin) used on the intensity of DM1 and DM2 Jacobians to weed out the weakest actuators

#--Zernikes to suppress with controller
mp.jac = falco.config.Object()
mp.jac.zerns = np.array([1])  #--Which Zernike modes to include in Jacobian. Given as the max Noll index. Always include the value "1" for the on-axis piston mode.
mp.jac.Zcoef = 1e-9*np.ones(np.size(mp.jac.zerns)); #--meters RMS of Zernike aberrations. (piston value is reset to 1 later)
       
#--Zernikes to compute sensitivities for
mp.eval = falco.config.Object()
mp.eval.indsZnoll = np.array([2, 3]) #--Noll indices of Zernikes to compute values for
#--Annuli to compute 1nm RMS Zernike sensitivities over. Columns are [inner radius, outer radius]. One row per annulus.
mp.eval.Rsens = np.array([[3., 4.], [4., 5.], [5., 8.], [8., 9.]]);  # [2-D ndarray]

#--Grid- or Line-Search Settings
mp.ctrl.log10regVec = np.arange(-6,-2,1/2) #-6:1/2:-2; #--log10 of the regularization exponents (often called Beta values)
mp.ctrl.dmfacVec = np.array([1.])            #--Proportional gain term applied to the total DM delta command. Usually in range [0.5,1]. [1-D ndarray]
# # mp.ctrl.dm9regfacVec = 1;        #--Additional regularization factor applied to DM9
   
#--Spatial pixel weighting
mp.WspatialDef = [];# [3, 4.5, 3]; #--spatial control Jacobian weighting by annulus: [Inner radius, outer radius, intensity weight; (as many rows as desired)]

#--DM weighting
mp.dm1.weight = 1.;
mp.dm2.weight = 1.;

#--Voltage range restrictions
mp.dm1.maxAbsV = 1000;  #--Max absolute voltage (+/-) for each actuator [volts] #--NOT ENFORCED YET
mp.dm2.maxAbsV = 1000;  #--Max absolute voltage (+/-) for each actuator [volts] #--NOT ENFORCED YET
mp.maxAbsdV = 1000;     #--Max +/- delta voltage step for each actuator for DMs 1 and 2 [volts] #--NOT ENFORCED YET

# %# Wavefront Control: Controller Specific
# Controller options: 
#  - 'gridsearchEFC' for EFC as an empirical grid search over tuning parameters
#  - 'plannedEFC' for EFC with an automated regularization schedule
#  - 'SM-CVX' for constrained EFC using CVX. --> DEVELOPMENT ONLY
mp.controller = 'gridsearchEFC';

# # # # GRID SEARCH EFC DEFAULTS     
#--WFSC Iterations and Control Matrix Relinearization
mp.Nitr = 5; #--Number of estimation+control iterations to perform
mp.relinItrVec = np.arange(0, mp.Nitr) #1:mp.Nitr;  #--Which correction iterations at which to re-compute the control Jacobian [1-D ndarray]
mp.dm_ind = np.array([1, 2]) #--Which DMs to use [1-D ndarray]

# # PLANNED SEARCH EFC DEFAULTS     
#mp.dm_ind = np.array([1, 2]) # vector of DMs used in controller at ANY time (not necessarily all at once or all the time). 
#mp.ctrl.dmfacVec = np.array([1.])

#--CONTROL SCHEDULE. Columns of mp.ctrl.sched_mat are: 
    # Column 1: # of iterations, 
    # Column 2: log10(regularization), 
    # Column 3: which DMs to use (12, 128, 129, or 1289) for control
    # Column 4: flag (0 = false, 1 = true), whether to re-linearize
    #   at that iteration.
    # Column 5: flag (0 = false, 1 = true), whether to perform an
    #   EFC parameter grid search to find the set giving the best
    #   contrast .
    # The imaginary part of the log10(regularization) in column 2 is
    #  replaced for that iteration with the optimal log10(regularization)
    # A row starting with [0, 0, 0, 1...] is for relinearizing only at that time

# mp.ctrl.sched_mat = [...
#     [0,0,0,1,0];
#     repmat([1,1j,12,0,1],[10,1]);...
#     ];

# mp.ctrl.sched_mat = [...
#     repmat([1,1j,12,1,1],[4,1]);...
#     repmat([1,1j-1,12,1,1],[25,1]);...
#     repmat([1,1j,12,1,1],[1,1]);...
#     ];
# [mp.Nitr, mp.relinItrVec, mp.gridSearchItrVec, mp.ctrl.log10regSchedIn, mp.dm_ind_sched] = falco_ctrl_EFC_schedule_generator(mp.ctrl.sched_mat);


### Deformable Mirrors: Influence Functions
##--Influence Function Options:
## - falco.INFLUENCE_XINETICS uses the file 'influence_dm5v2.fits' for one type of Xinetics DM
## - INFLUENCE_BMC_2K uses the file 'influence_BMC_2kDM_400micron_res10.fits' for BMC 2k DM
## - INFLUENCE_BMC_KILO uses the file 'influence_BMC_kiloDM_300micron_res10_spline.fits' for BMC kiloDM

mp.dm1.inf_fn = falco.INFLUENCE_XINETICS
mp.dm2.inf_fn = falco.INFLUENCE_XINETICS

mp.dm1.dm_spacing = 0.9906e-3 #--User defined actuator pitch
mp.dm2.dm_spacing = 0.9906e-3 #--User defined actuator pitch

mp.dm1.inf_sign = '+';
mp.dm2.inf_sign = '+';

### Deformable Mirrors: Optical Layout Parameters

##--DM1 parameters
mp.dm1.Nact = 48;               # # of actuators across DM array
mp.dm1.VtoH = 1e-9*np.ones((48,48))  # gains of all actuators [nm/V of free stroke]
mp.dm1.xtilt = 0;               # for foreshortening. angle of rotation about x-axis [degrees]
mp.dm1.ytilt = 5.7               # for foreshortening. angle of rotation about y-axis [degrees]
mp.dm1.zrot = 0;                # clocking of DM surface [degrees]
mp.dm1.xc = (48/2 - 1/2);       # x-center location of DM surface [actuator widths]
mp.dm1.yc = (48/2 - 1/2);       # y-center location of DM surface [actuator widths]
mp.dm1.edgeBuffer = 1;          # max radius (in actuator spacings) outside of beam on DM surface to compute influence functions for. [actuator widths]

##--DM2 parameters
mp.dm2.Nact = 48;               # # of actuators across DM array
mp.dm2.VtoH = 1e-9*np.ones((48,48))  # gains of all actuators [nm/V of free stroke]
mp.dm2.xtilt = 0;               # for foreshortening. angle of rotation about x-axis [degrees]
mp.dm2.ytilt = 5.7               # for foreshortening. angle of rotation about y-axis [degrees]
mp.dm2.zrot = 0;              # clocking of DM surface [degrees]
mp.dm2.xc = (48/2 - 1/2);       # x-center location of DM surface [actuator widths]
mp.dm2.yc = (48/2 - 1/2);       # y-center location of DM surface [actuator widths]
mp.dm2.edgeBuffer = 1;          # max radius (in actuator spacings) outside of beam on DM surface to compute influence functions for. [actuator widths]

#--Aperture stops at DMs
mp.flagDM1stop = False #--Whether to apply an iris or not
mp.dm1.Dstop = 100e-3;  #--Diameter of iris [meters]
mp.flagDM2stop = True  #--Whether to apply an iris or not
mp.dm2.Dstop = 50e-3;   #--Diameter of iris [meters]

#--DM separations
mp.d_P2_dm1 = 0        # distance (along +z axis) from P2 pupil to DM1 [meters]
mp.d_dm1_dm2 = 1.000   # distance between DM1 and DM2 [meters]


### Optical Layout: All models

#--Key Optical Layout Choices
mp.flagSim = True      #--Simulation or not
mp.layout = 'wfirst_phaseb_proper';  #--Which optical layout to use
mp.coro = 'SPLC';
mp.flagApod = True    #--Whether to use an apodizer or not
mp.flagDMwfe = False  #--Whether to use BMC DM quilting maps

mp.Fend = falco.config.Object()

#--Final Focal Plane Properties
mp.Fend.res = 3; #(730/660)*2.; #--Sampling [ pixels per lambda0/D]
mp.Fend.FOV = 11.; #--half-width of the field of view in both dimensions [lambda0/D]

#--Correction and scoring region definition
mp.Fend.corr = falco.config.Object()
mp.Fend.corr.Rin = 2.6;   # inner radius of dark hole correction region [lambda0/D]
mp.Fend.corr.Rout  = 9;  # outer radius of dark hole correction region [lambda0/D]
mp.Fend.corr.ang  = 65;  # angular opening of dark hole correction region [degrees]

mp.Fend.score = falco.config.Object()
mp.Fend.score.Rin = 3;  # inner radius of dark hole scoring region [lambda0/D]
mp.Fend.score.Rout = 9;  # outer radius of dark hole scoring region [lambda0/D]
mp.Fend.score.ang = 65;  # angular opening of dark hole scoring region [degrees]

mp.Fend.sides = 'both'; #--Which side(s) for correction: 'both', 'left', 'right', 'top', 'bottom'
mp.Fend.clockAngDeg = 90; #--Amount to rotate the dark hole location

# %# Optical Layout: Compact Model (and Jacobian Model)

#--Focal Lengths
mp.fl = 1.; #--[meters] Focal length value used for all FTs in the compact model. Don't need different values since this is a Fourier model.

#--Pupil Plane Diameters
mp.P2.D = 46.3e-3 #46.2987e-3;
mp.P3.D = 46.3e-3
mp.P4.D = 46.3e-3

#--Pupil Plane Resolutions
mp.P1.compact.Nbeam = 386
# mp.P2.compact.Nbeam = 386;
# mp.P3.compact.Nbeam = 386;
mp.P4.compact.Nbeam = 60

#--Shaped Pupil Mask: Load and downsample.
mp.compact.flagGenApod = False
mp.full.flagGenApod = False
mp.SPname = 'SPC-20190130';
SP0 = fits.getdata('../data/WFIRST/PhaseB/SPM_SPC-20190130.fits', ext=0)
NbeamSP = 1000
if(mp.P1.compact.Nbeam == NbeamSP):
    mp.P3.compact.mask = SP0
else:
    nBeamIn = NbeamSP
    nBeamOut = mp.P1.compact.Nbeam
    dx = 0
    dy = 0
    mp.P3.compact.mask = falco.mask.rotate_shift_downsample_pupil_mask(SP0, nBeamIn, nBeamOut, dx, dy, 0.)

if(mp.P1.full.Nbeam == NbeamSP):
    mp.P3.full.mask = SP0
else:
    nBeamIn = NbeamSP
    nBeamOut = mp.P1.full.Nbeam
    dx = 0
    dy = 0
    mp.P3.full.mask = falco.mask.rotate_shift_downsample_pupil_mask(SP0, nBeamIn, nBeamOut, dx, dy, 0.)


#--Number of re-imaging relays between pupil planesin compact model. Needed
#to keep track of 180-degree rotations and (1/1j)^2 factors compared to the
#full model, which probably has extra collimated beams compared to the
#compact model.
mp.Nrelay1to2 = 1;
mp.Nrelay2to3 = 1;
mp.Nrelay3to4 = 1;
mp.NrelayFend = 1; #--How many times to rotate the final image by 180 degrees

#--FPM resolution
mp.F3.compact.res = 6;    # sampling of FPM for compact model [pixels per lambda0/D]

#--Load and downsample the FPM. To get good grayscale edges, convolve with the correct window before downsampling. 
FPM0 = fits.getdata('/Users/ajriggs/Repos/falco-python/data/WFIRST/PhaseB/FPM_res100_SPC-20190130.fits') #--Resolution of 100 pixels per lambda0/D
FPM0 = falco.util.pad_crop(FPM0, (1821, 1821))
# figure(1); imagesc(FPM0); axis xy equal tight; colormap jet; colorbar;
# figure(11); imagesc(FPM0-rot90(FPM0,2)); axis xy equal tight; colormap jet; colorbar;
dx0 = 1/100.
dx1 = 1/mp.F3.compact.res
N0 = FPM0.shape[0]
if mp.centering == 'pixel':
    N1 = falco.util.ceil_odd(N0*dx0/dx1)
elif mp.centering == 'pixel':
    N1 = falco.util.ceil_even(N0*dx0/dx1)

x0 = np.arange(-(N0-1)/2., (N0-1)/2.+1)*dx0 #(-(N0-1)/2:(N0-1)/2)*dx0
[X0, Y0] = np.meshgrid(x0, x0)
R0 = np.sqrt(X0**2 + Y0**2);
Window = 0*R0
Window[R0 <= dx1/2.] = 1
Window = Window/np.sum(Window)
# figure(10); imagesc(Window); axis xy equal tight; colormap jet; colorbar;
#--To get good grayscale edges, convolve with the correct window before downsampling.
FPM0 = np.fft.ifftshift(  np.fft.ifft2( np.fft.fft2(np.fft.fftshift(Window))*np.fft.fft2(np.fft.fftshift(FPM0)) )) 
FPM0 = np.roll(FPM0, (1,1), axis=(0,1)) #--Undo a centering shift
x1 = np.arange(-(N1-1)/2., (N1-1)/2.+1)*dx1 # (-(N1-1)/2:(N1-1)/2)*dx1;
# [X1, Y1] = np.meshgrid(x1, x1)
FPM0 = np.real(FPM0)
interp_spline = RectBivariateSpline(x0, x0, FPM0) # RectBivariateSpline is faster in 2-D than interp2d
FPM1 = interp_spline(x1, x1)
# FPM1 = interp2(X0, Y0, FPM0, X1, Y1, 'cubic', 0); #--Downsample by interpolation
if mp.centering == 'pixel':
        mp.F3.compact.ampMask = np.zeros((N1+1, N1+1))
        mp.F3.compact.ampMask[1::, 1::] = FPM1
elif mp.centering == 'interpixel':    
        mp.F3.compact.ampMask = FPM1

# figure(2); imagesc(FPM0); axis xy equal tight; colormap jet; colorbar;
# figure(3); imagesc(FPM1); axis xy equal tight; colormap jet; colorbar;
# figure(12); imagesc(FPM0-rot90(FPM0,2)); axis xy equal tight; colormap jet; colorbar;
# figure(13); imagesc(FPM1-rot90(FPM1,2)); axis xy equal tight; colormap jet; colorbar;

# %# Optical Layout: Full Model 

mp.full.data_dir = '/Users/ajriggs/Repos/proper-models/wfirst_cgi/data_phaseb/'; # mask design data path
mp.full.cor_type = 'spc-spec_long'; #   'hlc', 'spc', or 'none' (none = clear aperture, no coronagraph)

mp.full.flagGenFPM = False
mp.full.flagPROPER = True #--Whether the full model is a PROPER prescription

# #--Pupil Plane Resolutions
mp.P1.full.Nbeam = 1000
mp.P1.full.Narr = 1002

mp.full.output_dim = falco.util.ceil_even(1 + mp.Fend.res*(2*mp.Fend.FOV)); #  dimensions of output in pixels (overrides output_dim0)
mp.full.final_sampling_lam0 = 1/mp.Fend.res;	#   final sampling in lambda0/D

mp.full.pol_conds = [10] # [-2,-1,1,2]; #--Which polarization states to use when creating an image.
mp.full.polaxis = 10                #   polarization condition (only used with input_field_rootname)
mp.full.use_errors = True

mp.full.zindex = [4]
mp.full.zval_m = [0.19e-9]
mp.full.use_hlc_dm_patterns = False # whether to use design WFE maps for HLC
mp.full.lambda0_m = mp.lambda0
mp.full.input_field_rootname = ''	#   rootname of files containing aberrated pupil
mp.full.use_dm1 = 0;                #   use DM1? 1 or 0
mp.full.use_dm2 = 0;                #   use DM2? 1 or 0
mp.full.dm_sampling_m = 0.9906e-3     #   actuator spacing in meters; default is 1 mm
mp.full.dm1_xc_act = 23.5         #   for 48x48 DM, wavefront centered at actuator intersections: (0,0) = 1st actuator center
mp.full.dm1_yc_act = 23.5
mp.full.dm1_xtilt_deg = 0 		#   tilt around X axis
mp.full.dm1_ytilt_deg = 5.7		#   effective DM tilt in deg including 9.65 deg actual tilt and pupil ellipticity
mp.full.dm1_ztilt_deg = 0
mp.full.dm2_xc_act = 23.5	
mp.full.dm2_yc_act = 23.5
mp.full.dm2_xtilt_deg = 0  
mp.full.dm2_ytilt_deg = 5.7
mp.full.dm2_ztilt_deg = 0
mp.full.use_fpm  = 1
mp.full.fpm_axis = 'p';             #   HLC FPM axis: '', 's', 'p'

mp.full.dm1 = falco.config.Object()
mp.full.dm2 = falco.config.Object()
mp.full.dm1.flatmap = fits.getdata('/Users/ajriggs/Repos/proper-models/wfirst_cgi/models_phaseb/matlab/examples/errors_polaxis10_dm.fits');
mp.full.dm2.flatmap = np.zeros((mp.dm2.Nact, mp.dm2.Nact))



# %# Mask Definitions

mp.compact.flagGenFPM = False

#--Pupil definition
mp.whichPupil = 'WFIRST180718';
mp.P1.IDnorm = 0.303; #--ID of the central obscuration [diameter]. Used only for computing the RMS DM surface from the ID to the OD of the pupil. OD is assumed to be 1.
mp.P1.D = 2.3631; #--telescope diameter [meters]. Used only for converting milliarcseconds to lambda0/D or vice-versa.
mp.P1.Dfac = 1; #--Factor scaling inscribed OD to circumscribed OD for the telescope pupil.

#--Lyot stop shape
mp.LSshape = 'bowtie';
mp.P4.IDnorm = 0.38; #--Lyot stop ID [Dtelescope]
mp.P4.ODnorm = 0.92; #--Lyot stop OD [Dtelescope]
mp.P4.ang = 90;      #--Lyot stop opening angle [degrees]
mp.P4.wStrut = 0;    #--Lyot stop strut width [pupil diameters]

# #--FPM size
# mp.F3.Rin = 2.6;   # inner hard-edge radius of the focal plane mask [lambda0/D]. Needs to be <= mp.F3.Rin 
# mp.F3.Rout = 9;   # radius of outer opaque edge of FPM [lambda0/D]
# mp.F3.ang = 65;    # on each side, opening angle [degrees]

