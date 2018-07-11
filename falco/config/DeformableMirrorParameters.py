import numpy as np
import os
import scipy.io
from falco.utils import _spec_arg

_influence_function_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "influence_dm5v2.mat")

class DeformableMirrorParameters:
    class _base_dm:
        def __init__(self,**kwargs):
            self.dm_spacing = _spec_arg("dm_spacing", kwargs, 0.001)
            self.dx_inf0 = _spec_arg("dx_inf0", kwargs, 0.0001)
            self.xc = _spec_arg("xc", kwargs, 23.5)
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.ytilt = _spec_arg("ytilt", kwargs, 0)
            self.xtilt = _spec_arg("xtilt", kwargs, 0)
            self.Nact = _spec_arg("Nact", kwargs, 48)
            self.edgeBuffer = _spec_arg("edgeBuffer", kwargs, 1)
            self.maxAbsV = _spec_arg("maxAbsV", kwargs, 125)
            self.yc = _spec_arg("yc", kwargs, 23.5)
            self.zrot = _spec_arg("zrot", kwargs, 0)
            self.VtoH = _spec_arg("VtoH", kwargs, 1e-9*np.ones((1,self.Nact)))
            self.inf0 = _spec_arg("inf0", kwargs, scipy.io.loadmat(_influence_function_file)["inf0"])

    def __init__(self,**kwargs):
        self.dm_weights = _spec_arg("dm_weights", kwargs, [1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.dm_ind = _spec_arg("dm_ind", kwargs, [1, 2])
        self.maxAbsdV = _spec_arg("maxAbsdV", kwargs, 30)
        self.dm1 = _spec_arg("dm1", kwargs, self._base_dm())
        self.dm2 = _spec_arg("dm2", kwargs, self._base_dm())
