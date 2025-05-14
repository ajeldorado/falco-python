"""Test differentiable DM surface fitting."""
from copy import deepcopy
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

import falco

DEBUG = True


def test_diff_dm_model():
    """Verify the orientation of the DM surface from gen_surf_from_act()."""
    mp = falco.config.ModelParameters()

    Nact = 48
    fCommand = np.zeros((Nact, Nact))
    #fCommand[30:40, 32] = 1
    #fCommand[40, 32:38] = 1
    #fCommand[35, 32:36] = 1
    fCommand[2,:] = 1.0
    fCommand[-3,:] = 1.0
    fCommand[:,2] = 1.0
    fCommand[:,-3] = 1.0
    mp.dm1.V = fCommand

    # DM1 parameters
    mp.dm1.centering = 'pixel'
    mp.dm1.Nact = Nact
    mp.dm1.VtoH = 1e-9*np.ones((mp.dm1.Nact, mp.dm1.Nact))
    #mp.dm1.xtilt = 10 # for foreshortening. angle of rotation about x-axis [degrees]
    mp.dm1.xtilt = 10
    #mp.dm1.ytilt = 15 # for foreshortening. angle of rotation about y-axis [degrees]
    mp.dm1.ytilt = 0
    #mp.dm1.zrot = -6  # clocking of DM surface [degrees]
    mp.dm1.zrot = 20
    mp.dm1.flagZYX = False
    mp.dm1.xc = (mp.dm1.Nact/2 - 1/2) + 1.1  # x-center location of DM surface [actuator widths]
    mp.dm1.yc = (mp.dm1.Nact/2 - 1/2) + 0.4 # y-center location of DM surface [actuator widths]
    mp.dm1.edgeBuffer = 1  # max radius (in actuator spacings) outside of beam on DM surface to compute influence functions for. [actuator widths]

    mp.dm1.fitType = 'linear'
    mp.dm1.pinned = np.array([])
    mp.dm1.Vpinned = np.zeros_like(mp.dm1.pinned)
    mp.dm1.tied = np.zeros((0, 2))
    mp.dm1.Vmin = 0
    mp.dm1.Vmax = 100
    mp.dm1.dVnbrLat = mp.dm1.Vmax
    mp.dm1.dVnbrDiag = mp.dm1.Vmax
    mp.dm1.biasMap = mp.dm1.Vmax/2 * np.ones((mp.dm1.Nact, mp.dm1.Nact))
    mp.dm1.facesheetFlatmap = mp.dm1.biasMap

    mp.dm1.inf_fn = falco.INFLUENCE_BMC_2K
    mp.dm1.dm_spacing = 400e-6  # User defined actuator pitch [meters]
    mp.dm1.inf_sign = '+'

    dx1 = None
    pitch1 = None
    mp.dm1.inf0 = None
    mp.dm1.dx_inf0 = None
    with fits.open(mp.dm1.inf_fn) as hdul:
        PrimaryData = hdul[0].header
        dx1 = PrimaryData['P2PDX_M']  # pixel width of influence function IN THE FILE [meters]
        pitch1 = PrimaryData['C2CDX_M']  # actuator spacing x (m)
        mp.dm1.ppact = pitch1/dx1
        mp.dm1.inf0 = np.squeeze(hdul[0].data)
    mp.dm1.dx_inf0 = mp.dm1.dm_spacing*(dx1/pitch1)

    if mp.dm1.inf_sign[0] in ['-', 'n', 'm']:
        mp.dm1.inf0 = -1*mp.dm1.inf0
    elif mp.dm1.inf_sign[0] in ['+', 'p']:
        pass
    else:
        raise ValueError('Sign of influence function not recognized')


    ppact = 5.43
    mp.dm1.dx = mp.dm1.dm_spacing/ppact
    Narray = int(np.ceil(ppact*Nact*1.5/2)*2 + 2)  # Must be odd for this test


    mp.dm1.orientation = 'rot0'
    # Generate surfaces for all orientations
    
    surfFalcoDm = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.dx, Narray)
    backprojFalcoDm = falco.dm.fit_surf_to_act(mp.dm1, surfFalcoDm)
    
    mp.dm1.useDifferentiableModel = True
    surfDiffDm = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.dx, Narray)
    backprojDiffDm = mp.dm1.differentiableModel.render_backprop(surfDiffDm, mp.dm1.VtoH, wfe=False)
    
    if DEBUG:
        plt.figure()
        plt.imshow(surfFalcoDm)
        plt.colorbar()
        plt.title('DM Surface FALCO')

        plt.figure()
        plt.imshow(surfDiffDm)
        plt.colorbar()
        plt.title('DM Surface Diff Model')
        
        plt.figure()
        plt.imshow(surfFalcoDm-surfDiffDm)
        plt.colorbar()
        plt.title('DM Surface Difference')
        
        
        plt.figure()
        plt.imshow(mp.dm1.V)
        plt.colorbar()
        plt.title('DM Voltages Truth')
        
        plt.figure()
        plt.imshow(backprojFalcoDm)
        plt.colorbar()
        plt.title('DM Backproj Voltages FALCO')

        plt.figure()
        plt.imshow(backprojDiffDm)
        plt.colorbar()
        plt.title('DM Backproj Voltages Model')
        
        plt.figure()
        plt.imshow(backprojFalcoDm-backprojDiffDm)
        plt.colorbar()
        plt.title('Backproj Voltage Difference')

        plt.show()

    abs_tol = 0.005*np.max(surfFalcoDm)

    maxAbsDiff = np.max(np.abs(surfFalcoDm - surfDiffDm))
    #maxAbsDiff = np.rmsnp.abs(surfFalcoDm - surfDiffDm))
    assert maxAbsDiff < abs_tol


    return None




if __name__ == '__main__':
    test_diff_dm_model()
