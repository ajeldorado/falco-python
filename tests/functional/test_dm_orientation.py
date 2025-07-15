"""Test DM surface fitting."""
from copy import deepcopy
from astropy.io import fits
import numpy as np

import falco

ORIENTATIONS = ('rot0', 'rot90', 'rot180', 'rot270', 'flipxrot0', 'flipxrot90',
                'flipxrot180', 'flipxrot270')


def test_surface_orientation():
    """Verify the orientation of the DM surface from gen_surf_from_act()."""
    mp = falco.config.ModelParameters()

    Nact = 48
    fCommand = np.zeros((Nact, Nact))
    fCommand[30:40, 32] = 1
    fCommand[40, 32:38] = 1
    fCommand[35, 32:36] = 1
    mp.dm1.V = fCommand

    # DM1 parameters
    mp.dm1.centering = 'pixel'
    mp.dm1.Nact = Nact
    mp.dm1.VtoH = 0.9*np.ones((mp.dm1.Nact, mp.dm1.Nact))
    mp.dm1.xtilt = 0  # for foreshortening. angle of rotation about x-axis [degrees]
    mp.dm1.ytilt = 0  # for foreshortening. angle of rotation about y-axis [degrees]
    mp.dm1.zrot = 0  # clocking of DM surface [degrees]
    mp.dm1.xc = (mp.dm1.Nact/2 - 1/2)  # x-center location of DM surface [actuator widths]
    mp.dm1.yc = (mp.dm1.Nact/2 - 1/2)  # y-center location of DM surface [actuator widths]
    mp.dm1.edgeBuffer = 1  # max radius (in actuator spacings) outside of beam on DM surface to compute influence functions for. [actuator widths]

    mp.dm1.fitType = 'linear'
    mp.dm1.dead = []
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

    ppact = 3
    dx = mp.dm1.dm_spacing/ppact
    Narray = int(np.ceil(ppact*Nact*1.5/2)*2 + 1)  # Must be odd for this test

    # Generate surfaces for all orientations
    mp.dm1.orientation = 'rot0'
    surfRot0 = falco.dm.gen_surf_from_act(mp.dm1, dx, Narray)

    mp.dm1.orientation = 'rot90'
    surfRot90 = falco.dm.gen_surf_from_act(mp.dm1, dx, Narray)

    mp.dm1.orientation = 'rot180'
    surfRot180 = falco.dm.gen_surf_from_act(mp.dm1, dx, Narray)

    mp.dm1.orientation = 'rot270'
    surfRot270 = falco.dm.gen_surf_from_act(mp.dm1, dx, Narray)

    mp.dm1.orientation = 'flipxrot0'
    surfFlipxRot0 = falco.dm.gen_surf_from_act(mp.dm1, dx, Narray)

    mp.dm1.orientation = 'flipxrot90'
    surfFlipxRot90 = falco.dm.gen_surf_from_act(mp.dm1, dx, Narray)

    mp.dm1.orientation = 'flipxrot180'
    surfFlipxRot180 = falco.dm.gen_surf_from_act(mp.dm1, dx, Narray)

    mp.dm1.orientation = 'flipxrot270'
    surfFlipxRot270 = falco.dm.gen_surf_from_act(mp.dm1, dx, Narray)

    abs_tol = 100*np.finfo(float).eps

    maxAbsDiff = np.max(np.abs(surfRot0 - np.rot90(surfRot90, -1)))
    assert maxAbsDiff < abs_tol

    maxAbsDiff = np.max(np.abs(surfRot0 - np.rot90(surfRot180, -2)))
    assert maxAbsDiff < abs_tol

    maxAbsDiff = np.max(np.abs(surfRot0 - np.rot90(surfRot270, -3)))
    assert maxAbsDiff < abs_tol

    maxAbsDiff = np.max(np.abs(surfRot0 - np.fliplr(surfFlipxRot0)))
    assert maxAbsDiff < abs_tol

    maxAbsDiff = np.max(np.abs(surfRot0 - np.fliplr(np.rot90(surfFlipxRot90, -1))))
    assert maxAbsDiff < abs_tol

    maxAbsDiff = np.max(np.abs(surfRot0 - np.fliplr(np.rot90(surfFlipxRot180, -2))))
    assert maxAbsDiff < abs_tol

    maxAbsDiff = np.max(np.abs(surfRot0 - np.fliplr(np.rot90(surfFlipxRot270, -3))))
    assert maxAbsDiff < abs_tol

    return None


def test_surface_orientation_from_cube():
    """Verify the orientation of the DM surface made from superposition."""
    mp = falco.config.ModelParameters()
    mp.P2.full = falco.config.Object()
    mp.P2.compact = falco.config.Object()

    Nact = 48
    fCommand = np.zeros((Nact, Nact))
    fCommand[30:40, 32] = 1
    fCommand[40, 32:38] = 1
    fCommand[35, 32:36] = 1
    mp.dm1.V = fCommand

    # DM1 parameters
    mp.dm1.centering = 'pixel'
    mp.dm1.Nact = Nact
    mp.dm1.VtoH = 0.9*np.ones((mp.dm1.Nact, mp.dm1.Nact))
    mp.dm1.xtilt = 0  # for foreshortening. angle of rotation about x-axis [degrees]
    mp.dm1.ytilt = 0  # for foreshortening. angle of rotation about y-axis [degrees]
    mp.dm1.zrot = 0  # clocking of DM surface [degrees]
    mp.dm1.xc = (mp.dm1.Nact/2 - 1/2)  # x-center location of DM surface [actuator widths]
    mp.dm1.yc = (mp.dm1.Nact/2 - 1/2)  # y-center location of DM surface [actuator widths]
    mp.dm1.edgeBuffer = 1  # max radius (in actuator spacings) outside of beam on DM surface to compute influence functions for. [actuator widths]

    mp.dm1.fitType = 'linear'
    mp.dm1.dead = []
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

    ppact = 5
    dx = mp.dm1.dm_spacing/ppact
    Narray = int(np.ceil(ppact*Nact*1.5/2)*2 + 1)  # Must be odd for this test

    # Make DM poke cube
    mp.centering = 'pixel'
    mp.dm_ind = np.array([1])
    mp.P2.full.dx = dx
    mp.P2.compact.dx = dx
    mp.sbp_centers = np.array([575e-9])
    mp.d_P2_dm1 = 0.0
    mp.d_dm1_dm2 = 0.2

    # Read the influence function header data from the FITS file
    # info = fitsinfo(mp.dm1.inf_fn);
    header = fits.getheader(mp.dm1.inf_fn)
    dx1 = header["P2PDX_M"]  # pixel width in the file
    pitch1 = header["C2CDX_M"]  # acuator pitch in the file

    mp.dm1.inf0 = fits.getdata(mp.dm1.inf_fn)
    mp.dm1.dx_inf0 = mp.dm1.dm_spacing * (dx1/pitch1)

    if mp.dm1.inf_sign[0] in ('-', 'n', 'm'):
        mp.dm1.inf0 = -1*mp.dm1.inf0

    # Create influence function datacubes for each DM
    mp.dm1.centering = mp.centering

    mp.dm1.compact = falco.config.Object()
    # mdc = deepcopy(mp.dm1.compact)
    mp.dm1.compact = mp.dm1
    mpCopy = deepcopy(mp)
    # for fn = fieldnames(mdc)
    #     mp.dm1.compact.(fn{1}) = mdc.(fn{1});

    # Tests
    abs_tol = 100*np.finfo(float).eps

    for orientation in ORIENTATIONS:

        # print('Orientation = %s' % orientation)
        mp = deepcopy(mpCopy)
        mp.dm1.orientation = orientation
        mp.dm1.compact.orientation = orientation

        surfA = falco.dm.gen_surf_from_act(mp.dm1, dx, Narray)

        falco.dm.gen_poke_cube(mp.dm1, mp, mp.P2.full.dx, NOCUBE=True)
        falco.dm.gen_poke_cube(mp.dm1.compact, mp, mp.P2.compact.dx)
        surfB = falco.util.pad_crop(
            falco.dm.gen_surf_from_poke_cube(mp.dm1, 'compact'),
            Narray)

        maxAbsDiff = np.max(np.abs(surfA - surfB))
        # print('maxAbsDiff = %.4g' % maxAbsDiff)
        assert maxAbsDiff < abs_tol


if __name__ == '__main__':
    test_surface_orientation()
    test_surface_orientation_from_cube()
