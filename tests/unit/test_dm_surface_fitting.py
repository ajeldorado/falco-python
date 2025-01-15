"""Unit test suite for DM surface generation and fitting."""
import unittest

from astropy.io import fits
import numpy as np

import falco

import matplotlib.pyplot as plt
DEBUG = False

class TestSurface(unittest.TestCase):
    """Test functionality and accuracy of DM surface generation and fitting."""

    @classmethod
    def setUpClass(self):
        """Initialize variables used in all the functional tests."""

        mp = falco.config.ModelParameters()

        Nact = 48
        fCommand = np.zeros((Nact, Nact))
        # F
        fCommand[30:40, 32] = 1
        fCommand[40, 32:38] = 1
        fCommand[35, 32:36] = 1
        # # Outer outline inset a couple rows and columns
        # fCommand[2,:] = 1.0
        # fCommand[-3,:] = 1.0
        # fCommand[:,2] = 1.0
        # fCommand[:,-3] = 1.0
        mp.dm1.V = fCommand

        # DM1 parameters
        mp.dm1.centering = 'pixel'
        mp.dm1.Nact = Nact
        mp.dm1.VtoH = 4e-9*np.ones((mp.dm1.Nact, mp.dm1.Nact))
        mp.dm1.xtilt = 10 # for foreshortening. angle of rotation about x-axis [degrees]
        mp.dm1.ytilt = 0 # for foreshortening. angle of rotation about y-axis [degrees]
        mp.dm1.zrot = 20  # clocking of DM surface [degrees]
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

        # mp.dm1.surfFitMethod = 'lsq'  # 'proper' or 'lsq'
        with fits.open(mp.dm1.inf_fn) as hdul:
            PrimaryData = hdul[0].header
            dx1 = PrimaryData['P2PDX_M']  # pixel width of influence function IN THE FILE [meters]
            pitch1 = PrimaryData['C2CDX_M']  # actuator spacing x (m)
            mp.dm1.ppact = pitch1/dx1  # pixel per actuator
            mp.dm1.inf0 = np.squeeze(hdul[0].data)

        dx1 = None
        pitch1 = None
        mp.dm1.inf0 = None
        mp.dm1.dx_inf0 = None
        with fits.open(mp.dm1.inf_fn) as hdul:
            PrimaryData = hdul[0].header
            dx1 = PrimaryData['P2PDX_M']  # pixel width of influence function IN THE FILE [meters]
            pitch1 = PrimaryData['C2CDX_M']  # actuator spacing x (m)

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
    
        self.surfFalcoDm = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.dx, Narray)

        mp.dm1.surfFitMethod = 'proper'  # 'proper' or 'lsq'
        self.backProjPROPER = falco.dm.fit_surf_to_act(mp.dm1, self.surfFalcoDm)

        mp.dm1.surfFitMethod = 'lsq'  # 'proper' or 'lsq'
        self.backProjLSQ = falco.dm.fit_surf_to_act(mp.dm1, self.surfFalcoDm)

        mp.dm1.useDifferentiableModel = True
        self.surfDiffDm = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.dx, Narray)
        self.backprojDiffDm = mp.dm1.differentiableModel.render_backprop(self.surfDiffDm, wfe=False) / mp.dm1.VtoH

        self.V0 = mp.dm1.V

    def testSameSurf(self):
        """Test that the surfaces generated two different ways are the same."""
        self.assertTrue(np.allclose(self.surfFalcoDm, self.surfDiffDm, rtol=1e-2))

        if DEBUG:
            plt.figure(1)
            plt.imshow(self.surfFalcoDm)
            plt.title('self.surfFalcoDm')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.figure(32)
            plt.imshow(self.surfDiffDm)
            plt.title('self.backprojDiffDm')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.figure(33)
            plt.imshow(self.surfFalcoDm - self.surfDiffDm)
            plt.title('self.surfFalcoDm - self.backprojDiffDm')
            plt.colorbar()
            plt.gca().invert_yaxis()


    def testFittingWithPROPER(self):
        """Test iterated surface fitting with FALCO+PROPER."""
        if DEBUG:

            plt.figure(11)
            plt.imshow(self.V0)
            plt.title('self.V0')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.figure(12)
            plt.imshow(self.backProjPROPER)
            plt.title('self.backProjPROPER')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.figure(13)
            plt.imshow(self.V0 - self.backProjPROPER)
            plt.title('self.V0 - self.backProjPROPER')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.show()
        self.assertTrue(np.allclose(self.V0, self.backProjPROPER, atol=3e-2))

    def testFittingWithLeastSquares(self):
        """Test the surface fitting with FALCO + a least-squares direct fit."""
        if DEBUG:

            plt.figure(21)
            plt.imshow(self.V0)
            plt.title('self.V0')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.figure(22)
            plt.imshow(self.backProjLSQ)
            plt.title('self.backProjLSQ')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.figure(23)
            plt.imshow(self.V0 - self.backProjLSQ)
            plt.title('self.V0 - self.backProjLSQ')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.show()
        self.assertTrue(np.allclose(self.V0, self.backProjPROPER, atol=3e-2))

    def testFittingDifferentiableModel(self):
        """Test surface fitting with the differentiable model."""
        if DEBUG:

            plt.figure(31)
            plt.imshow(self.V0)
            plt.title('self.V0')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.figure(32)
            plt.imshow(self.backprojDiffDm)
            plt.title('self.backprojDiffDm')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.figure(33)
            plt.imshow(self.V0 - self.backprojDiffDm)
            plt.title('self.V0 - self.backprojDiffDm')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.show()
        self.assertTrue(np.allclose(self.V0, self.backprojDiffDm, atol=1e-1))


if __name__ == '__main__':
    unittest.main()
