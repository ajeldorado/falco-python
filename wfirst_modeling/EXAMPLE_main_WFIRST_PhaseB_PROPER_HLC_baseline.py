# Copyright 2018-2020 by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged. Any
# commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# -------------------------------------------------------------------------

# import sys
# sys.path.append('/Users/ajriggs/Repos/proper-models/wfirst_cgi/models_phaseb/python/wfirst_phaseb_proper/examples')
import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import copy

import falco
import proper
import wfirst_phaseb_proper

# Load defaults
import EXAMPLE_defaults_WFIRST_PhaseB_PROPER_HLC as DEFAULTS
mp = DEFAULTS.mp

mp.path = falco.config.Object()
mp.path.falco = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# mp.path.falco = '../'  #--Location of FALCO

# Step 1: Set paths for output if desired

# ##--Output Data Directories (Comment these lines out to use defaults within falco-matlab/data/ directory.)
# mp.path.config = './' #--Location of config files and minimal output files. Default is [mainPath filesep 'data' filesep 'brief' filesep]
# mp.path.ws = './' # (Mostly) complete workspace from end of trial. Default is [mainPath filesep 'data' filesep 'ws' filesep];


# Step 2: Overwrite default values as desired

# ##--Special Computational Settings
mp.flagPlot = True;
mp.flagParallel = False; #--whether to use multiprocessing to parallelize some large computations
#mp.Nthreads = 2         #--Number of threads to use when using multiprocessing. If undefined, it is set to the 

#--Record Keeping
mp.SeriesNum = 1;
mp.TrialNum = 1;

# #--DEBUGGING:
mp.fracBW = 0.01       #--fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
mp.Nsbp = 1            #--Number of sub-bandpasses to divide the whole bandpass into for estimation and control
mp.Nwpsbp = 1          #--Number of wavelengths to used to approximate an image in each sub-bandpass
# # mp.flagParfor = false; #--whether to use parfor for Jacobian calculation

# mp.controller = 'plannedEFC';
# mp.ctrl.sched_mat = [...
#     [0,0,0,1,0];
#     repmat([1,1j,12,0,1],[5,1]);...
#     [1,-5,12,0,0];...
#     repmat([1,1j,12,0,1],[9,1]);...
#     ];
# [mp.Nitr, mp.relinItrVec, mp.gridSearchItrVec, mp.ctrl.log10regSchedIn, mp.dm_ind_sched] = falco_ctrl_EFC_schedule_generator(mp.ctrl.sched_mat);

if mp.Nsbp == 1:
    lambdaFacs = np.array([1, ])
elif mp.Nwpsbp == 1:
    lambdaFacs = np.linspace(1-mp.fracBW/2, 1+mp.fracBW/2, mp.Nsbp)
else:
    DeltaBW = mp.fracBW/(mp.Nsbp)*(mp.Nsbp-1)/2
    lambdaFacs = np.linspace(1-DeltaBW, 1+DeltaBW, mp.Nsbp)


lam_occ = lambdaFacs*mp.lambda0
# lam_occ = [     5.4625e-07, 5.4944e-07, 5.5264e-07, 5.5583e-07, 5.5903e-07, 5.6222e-07, 5.6542e-07,
#                         5.6861e-07, 5.7181e-07, 5.75e-07, 5.7819e-07, 5.8139e-07, 5.8458e-07, 5.8778e-07,
#                         5.9097e-07, 5.9417e-07, 5.9736e-07, 6.0056e-07, 6.0375e-07 ]
# lam_occs =  [   '5.4625e-07', '5.4944e-07', '5.5264e-07', '5.5583e-07', '5.5903e-07', '5.6222e-07', '5.6542e-07',
#                 '5.6861e-07', '5.7181e-07', '5.75e-07', '5.7819e-07', '5.8139e-07', '5.8458e-07', '5.8778e-07',
#                 '5.9097e-07', '5.9417e-07', '5.9736e-07', '6.0056e-07', '6.0375e-07' ]
mp.F3.compact.Nxi = 40; #--Crop down to minimum size of the spot
mp.F3.compact.Neta = mp.F3.compact.Nxi;
mp.compact.fpmCube = np.zeros((mp.F3.compact.Nxi, mp.F3.compact.Nxi, mp.Nsbp), dtype=complex)
fpm_axis = 'p';

for si in range(mp.Nsbp):
    lambda_um = 1e6*mp.lambda0*lambdaFacs[si];
    fn_p_r = str('%shlc_20190210/run461_occ_lam%stheta6.69pol%s_real.fits' % (mp.full.data_dir, str(lam_occ[si]), fpm_axis))
    fn_p_i = str('%shlc_20190210/run461_occ_lam%stheta6.69pol%s_imag.fits' % (mp.full.data_dir, str(lam_occ[si]), fpm_axis))

    # fn_p_r = mp.full.data_dir + 'hlc_20190210/run461_occ_lam' + num2str(lam_occ[si],12) + 'theta6.69pol'  + fpm_axis + '_' 'real.fits'
    # fn_p_i = mp.full.data_dir + 'hlc_20190210/run461_occ_lam' + num2str(lam_occ[si],12) + 'theta6.69pol'  + fpm_axis + '_' 'imag.fits'   
    mp.compact.fpmCube[:, :, si] = falco.util.pad_crop(fits.getdata(fn_p_r) + 1j*fits.getdata(fn_p_i), mp.F3.compact.Nxi)


#%% Visually check the FPM cropping
for si in range(mp.Nsbp):
   plt.figure(); plt.imshow(np.abs(mp.compact.fpmCube[:,:,si])); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)

#%% Step 3b: Obtain the phase retrieval phase.

mp.full.input_field_rootname = '/Users/ajriggs/Repos/falco-matlab/data/maps/input_full';

optval = copy.copy(mp.full)
# optval.data_dir = mp.full.data_dir;
# optval.cor_type = mp.full.cor_type;
optval.source_x_offset = 0;
# optval.zindex = 4;
# optval.zval_m = 0.19e-9;
# optval.use_errors = mp.full.use_errors;
# optval.polaxis = mp.full.polaxis; 

optval.dm1_m = np.zeros((mp.dm1.Nact, mp.dm1.Nact));#0.5*fitsread('errors_polaxis10_dm.fits');
optval.dm2_m = np.zeros((mp.dm2.Nact, mp.dm2.Nact));#0.5*fitsread('errors_polaxis10_dm.fits');
optval.use_dm1 = 1;
optval.use_dm2 = 1;

optval.end_at_fpm_exit_pupil = 1
# optval.output_field_rootname = [fileparts(mp.full.input_field_rootname) filesep 'fld_at_xtPup'];
optval.use_fpm = False
optval.use_hlc_dm_patterns = False
nout = 1024 #512; 			# nout > pupil_daim_pix

nArray = falco.util.ceil_even(mp.P1.compact.Nbeam+1)
mp.P1.compact.E = np.ones( (nArray, nArray, mp.Nsbp), dtype=complex) #--Initialize
for si in range(mp.Nsbp):
    # lambda_um = 1e6*mp.lambda0*lambdaFacs[si]

    # fld = prop_run(['model_full_wfirst_phaseb'], lambda_um, nout, 'quiet', 'passvalue', optval );

    # % figure(601); imagesc(angle(fld)); axis xy equal tight; colorbar; colormap hsv;
    # % figure(602); imagesc(abs(fld)); axis xy equal tight; colorbar; colormap parula;
    # plt.figure(); plt.imshow(np.angle(fld)); plt.colorbar(); plt.hsv(); plt.pause(0.1)
    # plt.figure(); plt.imshow(np.abs(fld)); plt.colorbar(); plt.magma(); plt.pause(0.1)

    # lams = num2str(lambda_um, '%6.4f');
    
    lambda_um = 1e6*mp.lambda0*lambdaFacs[si]

    fldFull, sampling = proper.prop_run('wfirst_phaseb', lambda_um, nout,  QUIET=True, PASSVALUE=optval.__dict__)
    if(mp.flagPlot):
        plt.figure(1); plt.imshow(np.angle(fldFull)); plt.colorbar(); plt.gca().invert_yaxis(); plt.hsv(); plt.pause(1e-2)
        plt.figure(2); plt.imshow(np.abs(fldFull)); plt.colorbar(); plt.gca().invert_yaxis(); plt.magma(); plt.pause(0.5)
        # figure(605); imagesc(angle(fldFull)); axis xy equal tight; colorbar; colormap hsv; drawnow;
#         figure(606); imagesc(abs(fldFull)); axis xy equal tight; colorbar; colormap parula; drawnow;
        pass
    
    # pols = ['polaxis'  num2str(optval.polaxis,2)];
    # fitswrite(real(fld), [mp.full.input_field_rootname '_' lams 'um_' pols '_real.fits' ]);
    # fitswrite(imag(fld), [mp.full.input_field_rootname '_' lams 'um_' pols '_imag.fits' ]);

    ##--Downsampling for the compact model
    # dxF = 1
    # dxC = mp.P1.full.Nbeam/mp.P1.compact.Nbeam

    # Nf = length(fld);
    # Nc = ceil_even( (mp.P1.compact.Nbeam/mp.P1.full.Nbeam)*Nf );

    # xF = (-Nf/2:Nf/2-1)*dxF;
    # xC = (-Nc/2:Nc/2-1)*dxC;

    # [Xf,Yf] = meshgrid(xF);
    # [Xc,Yc] = meshgrid(xC);

    # fldC = interp2(Xf,Yf,fld,Xc,Yc,'cubic',0); #--Downsample by interpolation
    # fldC = pad_crop(fldC,ceil_even(mp.P1.compact.Nbeam+1));

    # figure(607); imagesc(angle(fldC)); axis xy equal tight; colorbar; colormap hsv; drawnow;
    # figure(608); imagesc(abs(fldC)); axis xy equal tight; colorbar; colormap parula; drawnow;

    
    
    # temp = 0*fldC;
    # temp(2:end,2:end) = rot90(fldC(2:end,2:end), 2);
    # mp.P1.compact.E(:,:,si) = temp;
    
    # figure(617+si-1); imagesc(angle(fldC)); axis xy equal tight; colorbar; colormap hsv;

    #--Downsampling for the compact model
    dxF = 1
    dxC = mp.P1.full.Nbeam/mp.P1.compact.Nbeam
    Nf = fldFull.shape[0] #--N full
    Nc = falco.util.ceil_even( (mp.P1.compact.Nbeam/mp.P1.full.Nbeam)*Nf ) #--N compact
    xF = np.arange(-Nf/2, Nf/2)*dxF
    xC = np.arange(-Nc/2, Nc/2)*dxC
    interp_spline_real = RectBivariateSpline(xF, xF, np.real(fldFull)) # RectBivariateSpline is faster in 2-D than interp2d
    interp_spline_imag = RectBivariateSpline(xF, xF, np.imag(fldFull)) # RectBivariateSpline is faster in 2-D than interp2d
    fldC = interp_spline_real(xC, xC) + 1j*interp_spline_imag(xC, xC)
    N = falco.util.ceil_even(mp.P1.compact.Nbeam+1)
    fldC = falco.util.pad_crop(fldC, (N, N))
    if mp.flagPlot:
        plt.figure(11); plt.imshow(np.angle(fldC)); plt.colorbar(); plt.gca().invert_yaxis(); plt.hsv(); plt.pause(1e-2)
        plt.figure(12); plt.imshow(np.abs(fldC)); plt.colorbar(); plt.gca().invert_yaxis(); plt.magma(); plt.pause(0.5)
        pass
        
    #--Assign to initial E-field in compact model.
#     Etemp = 0*fldC;
#     Etemp[2:end,2:end] = rot90(fldC(2:end,2:end),2);
#     mp.P1.compact.E[:,:,si] = Etemp
    mp.P1.compact.E[:, :, si] = falco.prop.relay(fldC, 1, centering=mp.centering)

#%% After getting input E-field, add back HLC DM shapes
# mp.dm1.V = fitsread('hlc_dm1.fits')./mp.dm1.VtoH;
# mp.dm2.V = fitsread('hlc_dm2.fits')./mp.dm2.VtoH;

mp.dm1.V = fits.getdata('/Users/ajriggs/Repos/proper-models/wfirst_cgi/models_phaseb/python/wfirst_phaseb_proper/examples/hlc_with_aberrations_dm1.fits')/mp.dm1.VtoH
mp.dm2.V = fits.getdata('/Users/ajriggs/Repos/proper-models/wfirst_cgi/models_phaseb/python/wfirst_phaseb_proper/examples/hlc_with_aberrations_dm2.fits')/mp.dm2.VtoH


#%% Step 4: Generate the label associated with this trial
# Step 4: Generate the label associated with this trial
mp.runLabel = 'Series' + ('%04d'%(mp.SeriesNum)) + '_Trial' + ('%04d_'%(mp.TrialNum)) + mp.coro + \
'_' + mp.whichPupil + '_' + str(np.size(mp.dm_ind)) + 'DM' + str(mp.dm1.Nact) + '_z' + \
str(mp.d_dm1_dm2) + '_IWA' + str(mp.Fend.corr.Rin) + '_OWA' + str(mp.Fend.corr.Rout) + '_' + \
str(mp.Nsbp) + 'lams' + str(round(1e9*mp.lambda0)) + 'nm_BW' + str(mp.fracBW*100) + '_' + mp.controller


## Step 5: Perform the Wavefront Sensing and Control

out = falco.setup.flesh_out_workspace(mp)
falco.wfsc.loop(mp, out)
