from numpy import inf
import numpy as np

def _spec_arg(k,kwargs,v):
    if k in kwargs:
        return kwargs[k]
    elif "mat_struct" in kwargs:
        return eval("kwargs[\"mat_struct\"]." + k)
    else:
        return v

class ModelParameters:
    class _base_dm1:
        def __init__(self,**kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.Dstop = _spec_arg("Dstop", kwargs, 0.048)
    class _base_dm2:
        def __init__(self,**kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.Dstop = _spec_arg("Dstop", kwargs, 0.048)
    class _base_P2:
        def __init__(self,**kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.D = _spec_arg("D", kwargs, 0.0463)
    class _base_P3:
        def __init__(self,**kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.D = _spec_arg("D", kwargs, 0.0463)
    class _base_P1:
        class _base_compact:
            def __init__(self,**kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.Nbeam = _spec_arg("Nbeam", kwargs, 324)
        class _base_full:
            def __init__(self,**kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.Nbeam = _spec_arg("Nbeam", kwargs, 324)
        def __init__(self,**kwargs):
            self.compact = _spec_arg("compact", kwargs, self._base_compact())
            self.full = _spec_arg("full", kwargs, self._base_full())
            self.D = _spec_arg("D", kwargs, 2.3631)
    class _base_P4:
        class _base_compact:
            def __init__(self,**kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.Nbeam = _spec_arg("Nbeam", kwargs, 324)
        class _base_full:
            def __init__(self,**kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.Nbeam = _spec_arg("Nbeam", kwargs, 324)
        def __init__(self,**kwargs):
            self.compact = _spec_arg("compact", kwargs, self._base_compact())
            self.IDnorm = _spec_arg("IDnorm", kwargs, 0.5)
            self.ODnorm = _spec_arg("ODnorm", kwargs, 0.8)
            self.full = _spec_arg("full", kwargs, self._base_full())
            self.D = _spec_arg("D", kwargs, 0.0463)
    class _base_F3:
        class _base_compact:
            def __init__(self,**kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.res = _spec_arg("res", kwargs, 30)
        class _base_full:
            def __init__(self,**kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.res = _spec_arg("res", kwargs, 50)
        def __init__(self,**kwargs):
            self.compact = _spec_arg("compact", kwargs, self._base_compact())
            self.Rin = _spec_arg("Rin", kwargs, 2.8)
            self.Rout = _spec_arg("Rout", kwargs, inf)
            self.full = _spec_arg("full", kwargs, self._base_full())
            self.ang = _spec_arg("ang", kwargs, 180)
    class _base_F4:
        class _base_compact:
            def __init__(self,**kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.res = _spec_arg("res", kwargs, 3)
        class _base_full:
            def __init__(self,**kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.res = _spec_arg("res", kwargs, 6)
        class _base_score:
            def __init__(self,**kwargs):
                self.Rin = _spec_arg("Rin", kwargs, 2.8)
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.Rout = _spec_arg("Rout", kwargs, 10)
                self.ang = _spec_arg("ang", kwargs, 180)
        class _base_corr:
            def __init__(self,**kwargs):
                self.Rin = _spec_arg("Rin", kwargs, 2.8)
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.Rout = _spec_arg("Rout", kwargs, 10)
                self.ang = _spec_arg("ang", kwargs, 180)
        def __init__(self,**kwargs):
            self.compact = _spec_arg("compact", kwargs, self._base_compact())
            self.FOV = _spec_arg("FOV", kwargs, 11)
            self.full = _spec_arg("full", kwargs, self._base_full())
            self.score = _spec_arg("score", kwargs, self._base_score())
            self.corr = _spec_arg("corr", kwargs, self._base_corr())
            self.sides = _spec_arg("sides", kwargs, "both")
    def __init__(self,**kwargs):
        self.Nitr = _spec_arg("Nitr", kwargs, 10)
        self.SPname = _spec_arg("SPname", kwargs, 0)
        self.TToffset = _spec_arg("TToffset", kwargs, 1)
        self.TrialNum = _spec_arg("TrialNum", kwargs, 1)
        self.planetFlag = _spec_arg("planetFlag", kwargs, 0)
        self.Nsbp = _spec_arg("Nsbp", kwargs, 1)
        self.SeriesNum = _spec_arg("SeriesNum", kwargs, 1)
        self.thput_radius = _spec_arg("thput_radius", kwargs, 0.7)
        self.lambda0 = _spec_arg("lambda0", kwargs, 5.75e-07)
        self.flagApod = _spec_arg("flagApod", kwargs, 0)
        self.centering = _spec_arg("centering", kwargs, "pixel")
        self.Nwpsbp = _spec_arg("Nwpsbp", kwargs, 1)
        self.useGPU = _spec_arg("useGPU", kwargs, 0)
        self.NlamForTT = _spec_arg("NlamForTT", kwargs, 1)
        self.flagNewPSD = _spec_arg("flagNewPSD", kwargs, 0)
        self.pup_strut_width = _spec_arg("pup_strut_width", kwargs, 0.0322)
        self.dm1 = _spec_arg("dm1", kwargs, self._base_dm1())
        self.dm2 = _spec_arg("dm2", kwargs, self._base_dm2())
        self.whichPupil = _spec_arg("whichPupil", kwargs, "WFIRST20180103")
        self.flagDM2stop = _spec_arg("flagDM2stop", kwargs, 0)
        self.d_dm1_dm2 = _spec_arg("d_dm1_dm2", kwargs, 1)
        self.P2 = _spec_arg("P2", kwargs, self._base_P2())
        self.P3 = _spec_arg("P3", kwargs, self._base_P3())
        self.P1 = _spec_arg("P1", kwargs, self._base_P1())
        self.P4 = _spec_arg("P4", kwargs, self._base_P4())
        self.relinItrVec = _spec_arg("relinItrVec", kwargs, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.flagParfor = _spec_arg("flagParfor", kwargs, 0)
        self.coro = _spec_arg("coro", kwargs, "LC")
        self.Ntt = _spec_arg("Ntt", kwargs, 1)
        self.controller = _spec_arg("controller", kwargs, "EFC")
        self.logGmin = _spec_arg("logGmin", kwargs, -6)
        self.fracBW = _spec_arg("fracBW", kwargs, 0.01)
        self.LS_strut_width = _spec_arg("LS_strut_width", kwargs, 0.038)
        self.fl = _spec_arg("fl", kwargs, 1)
        self.F3 = _spec_arg("F3", kwargs, self._base_F3())
        self.F4 = _spec_arg("F4", kwargs, self._base_F4())
        self.d_P2_dm1 = _spec_arg("d_P2_dm1", kwargs, 0)
        self.FPMampFac = _spec_arg("FPMampFac", kwargs, 0)
        self.flagDM1stop = _spec_arg("flagDM1stop", kwargs, 0)
        self.thput_eval_y = _spec_arg("thput_eval_y", kwargs, 0)
        self.thput_eval_x = _spec_arg("thput_eval_x", kwargs, 6)
        self.WspatialDef = _spec_arg("WspatialDef", kwargs, [])

def get_default_LC_config():
    return ModelParameters() #All there is for now
