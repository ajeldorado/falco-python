"""Test DM surface fitting."""
import unittest
import numpy as np

import proper
import falco
import testing_config_LC as DEFAULTS


class TestFitSurfToAct(unittest.TestCase):
    """Test falco.dm.fit_surf_to_act."""

    def test_nada(self):
        pass

    # def test_fit_DM_surface_with_DM(self):
    #     mp = DEFAULTS.mp
            
    #     mp.path = falco.config.Object()
    #     mp.path.falco = './'  # Location of FALCO
    #     mp.path.proper = './'  # Location of the MATLAB PROPER library
        
    #     mp.path.config = './'  # Location of config files and brief output
    #     mp.path.ws = './'  # (Mostly) complete workspace from end of trial.
        
    #     # Overwrite default values as desired
    #     mp.dm1.xtilt = 45
    #     mp.dm1.ytilt = 20
    #     mp.dm1.zrot = 30
        
    #     mp.dm1.xc = mp.dm1.Nact/2 - 1/2 + 1
    #     mp.dm1.yc = mp.dm1.Nact/2 - 1/2 - 1
        
    #     # Step 4: Initialize the rest of the workspace
    #     out = falco.setup.flesh_out_workspace(mp)
        
    #     # Generate a DM surface and try to re-create the actuator commands
    #     normFac = 1
    #     mp.dm1.V = normFac * np.random.rand(mp.dm1.Nact, mp.dm1.Nact)
    #     DM1Surf = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.compact.dx,
    #                                           mp.dm1.compact.Ndm)
        
    #     # Fit the surface
    #     Vout = falco.dm.fit_surf_to_act(mp.dm1, DM1Surf)/mp.dm1.VtoH
    #     Verror = mp.dm1.V - Vout
    #     rmsVError = np.sqrt(np.mean(Verror.flatten()**2))/normFac
    #     print('RMS fitting error to voltage map is %.2f%%.\n' %
    #           (rmsVError*100))
            
    #     self.assertTrue(rmsVError < 2.5/100)
        
    # def test_fit_PSD_error_map_with_DM(self):
    
    #     mp = DEFAULTS.mp
            
    #     mp.path = falco.config.Object()
    #     mp.path.falco = './'  # Location of FALCO
    #     mp.path.proper = './'  # Location of the MATLAB PROPER library
        
    #     mp.path.config = './'  # Location of config files and brief output
    #     mp.path.ws = './'  # (Mostly) complete workspace from end of trial.
        
    #     # Overwrite default values as desired
    #     mp.dm1.xtilt = 45
    #     mp.dm1.ytilt = 20
    #     mp.dm1.zrot = 30
        
    #     mp.dm1.xc = (mp.dm1.Nact/2 - 1/2) + 1
    #     mp.dm1.yc = (mp.dm1.Nact/2 - 1/2) - 1
        
    #     # Initialize the rest of the workspace
    #     out = falco.setup.flesh_out_workspace(mp)
        
    #     # Determine the region of the array corresponding to the DM surface
    #     mp.dm1.V = np.ones((mp.dm1.Nact, mp.dm1.Nact))
    #     testSurf = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.compact.dx,
    #                                           mp.dm1.compact.NdmPad)
    #     testArea = np.zeros(testSurf.shape)
    #     testArea[testSurf >= 0.5*np.max(testSurf)] = 1
        
    #     # PROPER initialization
    #     pupil_ratio = 1  # beam diameter fraction
    #     wl_dummy = 1e-6  # dummy value
    #     wavefront = proper.prop_begin(mp.dm1.compact.NdmPad*mp.dm1.dx,
    #                                   wl_dummy, mp.dm1.compact.NdmPad,
    #                                   pupil_ratio)
    #     # PSD Error Map Generation using PROPER
    #     amp = 9.6e-19
    #     b = 4.0
    #     c = 3.0
    #     errorMap = proper.prop_psd_errormap(wavefront, amp, b, c, TPF=True)
    #     errorMap = errorMap*testArea
        
    #     # Fit the surface
    #     Vout = falco.dm.fit_surf_to_act(mp.dm1, errorMap)/mp.dm1.VtoH
    #     mp.dm1.V = Vout
    #     DM1Surf = falco.dm.gen_surf_from_act(mp.dm1, mp.dm1.compact.dx,
    #                                          mp.dm1.compact.NdmPad)
    #     surfError = errorMap - DM1Surf
    #     rmsError = np.sqrt(np.mean((surfError[testArea == 1].flatten()**2)))
    #     print('RMS fitting error to voltage map is %.2e meters.\n' % rmsError)
        
    #     self.assertTrue(rmsError < 2.0e-9)


if __name__ == '__main__':
    unittest.main()
