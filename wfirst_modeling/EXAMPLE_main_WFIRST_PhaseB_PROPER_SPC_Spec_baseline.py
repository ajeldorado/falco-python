# Copyright 2018-2020 by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged. Any
# commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# -------------------------------------------------------------------------


import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import copy

import falco
import proper
import wfirst_phaseb_proper

# Load defaults
import EXAMPLE_defaults_WFIRST_PhaseB_PROPER_SPC_Spec as DEFAULTS
mp = DEFAULTS.mp

mp.path = falco.config.Object()
mp.path.falco = '../'  #--Location of FALCO

# Step 1: Set paths for output if desired

# ##--Output Data Directories (Comment these lines out to use defaults within falco-matlab/data/ directory.)
# mp.path.config = './' #--Location of config files and minimal output files. Default is [mainPath filesep 'data' filesep 'brief' filesep]
# mp.path.ws = './' # (Mostly) complete workspace from end of trial. Default is [mainPath filesep 'data' filesep 'ws' filesep];


# Step 2: Overwrite default values as desired

# ##--Special Computational Settings
mp.flagPlot = True;
mp.flagMultiproc = False; #--whether to use multiprocessing to parallelize some large computations
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


# Step 3: Obtain the phase retrieval phase.

mp.full.input_field_rootname = '/Users/ajriggs/Repos/falco-matlab/data/maps/input_full';
optval = copy.copy(mp.full)
#optval = copy.copy(vars(mp.full))
optval.source_x_offset = 0
#optval.zindex = [4];
#optval.zval_m = [0.19e-9;]
optval.dm1_m = mp.full.dm1.flatmap #fits.getdata('errors_polaxis10_dm.fits');
optval.use_dm1 = True

optval.end_at_fpm_exit_pupil = True
#optval.output_field_rootname = [fileparts(mp.full.input_field_rootname) filesep 'fld_at_xtPup'];
optval.use_fpm = False
optval.use_hlc_dm_patterns = False
nout = 1024 #512; 			# nout > pupil_daim_pix
optval.output_dim = 1024 # Get the Input Pupil's E-field

if(mp.Nsbp==1):
    lambdaFacs = np.array([1.])
else:
    lambdaFacs = np.linspace(1-mp.fracBW/2., 1+mp.fracBW/2., mp.Nsbp)


#--Get the Input Pupil's E-field
mp.P1.compact.E = np.ones((mp.P1.compact.Nbeam+2, mp.P1.compact.Nbeam+2, mp.Nsbp), dtype=complex) #--Initialize
for si in range(mp.Nsbp):
    lambda_um = 1e6*mp.lambda0*lambdaFacs[si]

    fldFull, sampling = proper.prop_run('wfirst_phaseb', lambda_um, nout,  QUIET=True, PASSVALUE=optval.__dict__)
    if(mp.flagPlot):
        plt.figure(1); plt.imshow(np.angle(fldFull)); plt.colorbar(); plt.hsv(); plt.pause(1e-2)
        plt.figure(2); plt.imshow(np.abs(fldFull)); plt.colorbar(); plt.magma(); plt.pause(0.5)
        # figure(605); imagesc(angle(fldFull)); axis xy equal tight; colorbar; colormap hsv; drawnow;
#         figure(606); imagesc(abs(fldFull)); axis xy equal tight; colorbar; colormap parula; drawnow;
        pass

    lams = '%6.4f' % lambda_um 
    pols = 'polaxis%s' % optval.polaxis #['polaxis'  num2str(optval.polaxis,2)];
    fnReal = (mp.full.input_field_rootname + '_' + lams + 'um_' + pols + '_real.fits')
    fnImag = (mp.full.input_field_rootname + '_' + lams + 'um_' + pols + '_imag.fits')
#    fitswrite(real(fldFull), [mp.full.input_field_rootname '_' lams 'um_' pols '_real.fits' ]);
#    fitswrite(imag(fldFull), [mp.full.input_field_rootname '_' lams 'um_' pols '_imag.fits' ]);
    hduReal = fits.PrimaryHDU(np.real(fldFull))
    hduReal.writeto(fnReal,overwrite=True)
    hduImag = fits.PrimaryHDU(np.imag(fldFull))
    hduImag.writeto(fnImag,overwrite=True)
    

    #--Downsampling for the compact model
    dxF = 1
    dxC = mp.P1.full.Nbeam/mp.P1.compact.Nbeam
    Nf = fldFull.shape[0] #--N full
    Nc = falco.utils.ceil_even( (mp.P1.compact.Nbeam/mp.P1.full.Nbeam)*Nf ) #--N compact
    xF = np.arange(-Nf/2, Nf/2)*dxF
    xC = np.arange(-Nc/2, Nc/2)*dxC
#     [Xf,Yf] = np.meshgrid(xF);
#     [Xc,Yc] = np.meshgrid(xC);
    interp_spline_real = RectBivariateSpline(xF, xF, np.real(fldFull)) # RectBivariateSpline is faster in 2-D than interp2d
    interp_spline_imag = RectBivariateSpline(xF, xF, np.imag(fldFull)) # RectBivariateSpline is faster in 2-D than interp2d
    fldC = interp_spline_real(xC, xC) + 1j*interp_spline_imag(xC, xC)
#     fldC = interp2(Xf,Yf,fldFull,Xc,Yc,'cubic',0); #--Downsample by interpolation
    N = falco.utils.ceil_even(mp.P1.compact.Nbeam+1)
    fldC = falco.utils.pad_crop(fldC, (N, N))
    if mp.flagPlot:
        plt.figure(11); plt.imshow(np.angle(fldC)); plt.colorbar(); plt.hsv(); plt.pause(1e-2)
        plt.figure(12); plt.imshow(np.abs(fldC)); plt.colorbar();  plt.magma(); plt.pause(0.5)        
        # figure(607+si-1); imagesc(angle(fldC)); axis xy equal tight; colorbar; colormap hsv; drawnow;
#         figure(608); imagesc(abs(fldC)); axis xy equal tight; colorbar; colormap parula; drawnow;
        pass
        
    #--Assign to initial E-field in compact model.
#     Etemp = 0*fldC;
#     Etemp[2:end,2:end] = rot90(fldC(2:end,2:end),2);
#     mp.P1.compact.E[:,:,si] = Etemp
    mp.P1.compact.E[:,:,si] = falco.prop.relay(fldC, 1, centering=mp.centering)
    
    


# Step 4: Generate the label associated with this trial
mp.runLabel = 'Series' + ('%04d'%(mp.SeriesNum)) + '_Trial' + ('%04d_'%(mp.TrialNum)) + mp.coro + \
'_' + mp.whichPupil + '_' + str(np.size(mp.dm_ind)) + 'DM' + str(mp.dm1.Nact) + '_z' + \
str(mp.d_dm1_dm2) + '_IWA' + str(mp.Fend.corr.Rin) + '_OWA' + str(mp.Fend.corr.Rout) + '_' + \
str(mp.Nsbp) + 'lams' + str(round(1e9*mp.lambda0)) + 'nm_BW' + str(mp.fracBW*100) + '_' + mp.controller


## Step 5: Perform the Wavefront Sensing and Control

out = falco.setup.flesh_out_workspace(mp)
falco.wfsc.loop(mp, out)

# print('END OF MAIN: ', mp)
