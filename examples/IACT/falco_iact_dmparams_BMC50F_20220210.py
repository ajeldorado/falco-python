import numpy as np
from astropy.io import fits
from falco_update_dm_gain_map import falco_update_dm_gain_map


def falco_iact_dmparams_BMC50F_20220210(mp, tb=None):
    #--Pupil definition
    mp.whichPupil = 'Simple';
    mp.P1.IDnorm = 0; #--ID of the central obscuration [diameter]. Used only for computing the RMS DM surface from the ID to the OD of the pupil. OD is assumed to be 1.
    mp.P1.ODnorm = 1;# Outer diameter of the telescope [diameter]
    mp.P1.Nstrut = 0;# Number of struts 
    mp.P1.angStrut = [];#Array of angles of the radial struts (deg)
    mp.P1.wStrut = []; # Width of the struts (fraction of pupil diam.)
    mp.P1.stretch = 1;
    
    # set pupil size 
    if 'tb' is not None:
        Nbeam = tb.sciCam.pupilNbeam;
    else:
        Nbeam = 168.3766; 

    
    ## DM1 parameters 
    mp.dm1.inf_fn = 'influence_BMC_2kDM_400micron_res10.fits';
    mp.dm1.dm_spacing = 400e-6; #--User defined actuator pitch
    mp.dm1.inf_sign = '+';
    
    mp.dm1.centering = 'pixel';
    mp.dm1.Nact = 50;               # # of actuators across DM array
    mp.dm1.xtilt = 0;               # for foreshortening. angle of rotation about x-axis [degrees] 
    mp.dm1.ytilt = 0;               # for foreshortening. angle of rotation about y-axis [degrees]
    mp.dm1.zrot = -90.8252;          # clocking of DM surface [degrees]
    mp.dm1.xc = 24.4859;            # x-center location of DM surface [actuator widths]
    mp.dm1.yc = 25.0286;            # y-center location of DM surface [actuator widths]
    mp.dm1.edgeBuffer = 1;          # max radius (in actuator spacings) outside of beam on DM surface to compute influence functions for. [actuator widths]
    mp.dm1.Nactbeam = 46.2582;      # Number of actuators across the beam (diameter defined by Nbeam)
    mp.dm1.flagNbrRule = False;     # True to enforce neighbor rules
    mp.dm1.Vmin = 0;             # Minimum voltage for FALCO (not neccessarily the full DM command including the flat map)
    mp.dm1.Vmax = 0.3;              # Maximum voltage for FALCO 
    mp.dm1.maxAbsV = 100;
    mp.dm1.transp = False;  # True transposes the DM commands in the model (Doesn't work when applying DM constraints)
    
    if 'tb' is not None:
        mp.dm1.biasMap = tb.DM.flatmap; 
    #     mp.dm1.facesheetFlatmap = tb.DM1.facesheetFlatmap; 
    else:
        hdu = fits.open('/proj/iact/data/falco/calibration/dm_flats/cBMC_50F_flatmap_20211021a.fits')
        mp.dm1.biasMap = hdu[0].data
        #fitsread('/proj/iact/data/falco/calibration/dm_flats/cBMC_50F_flatmap_20211021a.fits');
    #     mp.dm1.facesheetFlatmap = fitsread('/proj/dst/data/AOX/DST/dm_flats/AOX_48.4_flatvolts-20180902-45vbias-30vcorners.fits')';# VSG calibs are transposed for 48.4

    
    # mp.dm1.biasMap = tb.DM.flatmap; 
    # # mp.dm1.biasMap = fitsread('/proj/iact/data/falco/calibration/dm_flats/cBMC_50F_flatmap_20210527_1.fits');
    
    mp.dm1.pinned = 2175;             # List of pinned actuators 
    mp.dm1.Vpinned = np.zeros(np.size(mp.dm1.pinned)); # Voltage of pinned actuators
    
    mp.dm1.tied = np.zeros([1,2]);       # List of tied actuators
    
    #--DM gains (m/V)
    # mp.dm1.fitType = 'linear';
    # mp.dm1.VtoH = 1.2*1.77629e-6*np.ones(mp.dm1.Nact);# from PR 
    # mp.dm1.VtoH(mp.dm1.tied) = mp.dm1.VtoH(mp.dm1.tied)/2;
    
    # DM gains from quadratic displacement: h = p1*V^2 + p2*V + p3
    mp.dm1.fitType = 'quadratic';
    mp.dm1.p1 = 5.587*1e-6*np.ones(mp.dm1.Nact); #from Zygo
    mp.dm1.p2 = 0.14088*1e-6*np.ones(mp.dm1.Nact);
    # mp.dm1.p1 = 3.1908*1e-6*np.ones(mp.dm1.Nact);
    # mp.dm1.p2 = 0.61079*1e-6*np.ones(mp.dm1.Nact);
    mp.dm1.p3 = 0;
    
    # mp.dm1.p1(mp.dm1.tied) = mp.dm1.p1(mp.dm1.tied)/2;
    # mp.dm1.p2(mp.dm1.tied) = mp.dm1.p2(mp.dm1.tied)/2;
    mp.dm1.VtoHfudge = 1.4;# Fudge factor for VtoH (scales the result of 2*p1*V+p2)
    mp.dm1.V = np.zeros(mp.dm1.Nact);
    mp.dm1 = falco_update_dm_gain_map(mp.dm1);  
    
    ## DM2 parameters (not used)
    mp.dm2.inf_fn = 'influence_BMC_2kDM_400micron_res10.fits';
    mp.dm2.dm_spacing = 400e-6; #--User defined actuator pitch
    mp.dm2.inf_sign = '+';
    
    mp.dm2.centering = 'pixel';
    mp.dm2.Nact = 1;               # # of actuators across DM array
    mp.dm2.xtilt = 0;               # for foreshortening. angle of rotation about x-axis [degrees]
    mp.dm2.ytilt = 0;               # for foreshortening. angle of rotation about y-axis [degrees]
    mp.dm2.zrot = 0;                # clocking of DM surface [degrees]
    mp.dm2.xc = 24.5;               # x-center location of DM surface [actuator widths]
    mp.dm2.yc = 24.5;               # y-center location of DM surface [actuator widths]
    mp.dm2.edgeBuffer = 1;          # max radius (in actuator spacings) outside of beam on DM surface to compute influence functions for. [actuator widths]
    mp.dm2.Nactbeam = 1;      # Number of actuators across the beam (diameter defined by Nbeam)
    mp.dm2.flagNbrRule = False;     # True to enforce neighbor rules
    mp.dm2.Vmin = 0;             # Minimum voltage for FALCO (not neccessarily the full DM command including the flat map)
    mp.dm2.Vmax = 100;              # Maximum voltage for FALCO 
    mp.dm2.maxAbsV = 100;
    mp.dm2.pinned = [];             # List of pinned actuators 
    mp.dm2.Vpinned = [];            # Voltage of pinned actuators 
    mp.dm2.tied = np.zeros([1,2]);       # List of tied actuators
    mp.dm2.transp = False;           # True transposes the DM commands in the model 
    mp.dm2.biasMap = np.zeros(mp.dm1.Nact); 
    
    #--DM gains (m/V)
    mp.dm2.fitType = 'linear';
    mp.dm2.VtoH = 1.86761e-6*np.ones(mp.dm2.Nact);

    return