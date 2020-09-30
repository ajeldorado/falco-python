# import copy
from numpy import inf
# import numpy as np
# import falco.mask
# import falco.util
from falco.util import _spec_arg
# from falco.config import DeformableMirrorParameters
# import collections
# from falco import model

class Object(object):
    pass

class ModelParameters:
#     class _base_compact:
#         def __init__(self, **kwargs):
#             self.dummy = _spec_arg("dummy", kwargs, 1)
# 
#     class _base_full:
#         def __init__(self, **kwargs):
#             self.dummy = _spec_arg("dummy", kwargs, 1)

    class _base_dm1:
        def __init__(self, **kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.Dstop = _spec_arg("Dstop", kwargs, 0.048)
            self.dx= _spec_arg("dx", kwargs, 0)
            self.NdmPad= _spec_arg("NdmPad", kwargs, 0)

    class _base_dm2:
        def __init__(self, **kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.Dstop = _spec_arg("Dstop", kwargs, 0)
            self.dx= _spec_arg("dx", kwargs, 0.048)
            self.NdmPad= _spec_arg("NdmPad", kwargs, 0)
    
    class _base_dm8:
        def __init__(self, **kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            
    class _base_dm9:
        def __init__(self, **kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)

    class _base_P2:
        def __init__(self, **kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.D = _spec_arg("D", kwargs, 0.0463)

#     class _base_P3:
#         def __init__(self, **kwargs):
#             self.dummy = _spec_arg("dummy", kwargs, 1)
#             self.D = _spec_arg("D", kwargs, 0.0463)
                
    class _base_P1:
        class _base_compact:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.Nbeam = _spec_arg("Nbeam", kwargs, 324)

        class _base_full:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.Nbeam = _spec_arg("Nbeam", kwargs, 324)

        def __init__(self, **kwargs):
            self.compact = _spec_arg("compact", kwargs, self._base_compact())
            self.full = _spec_arg("full", kwargs, self._base_full())
#             self.D = _spec_arg("D", kwargs, 2.3631)

    class _base_P3:
        class _base_compact:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
#                 self.Nbeam = _spec_arg("Nbeam", kwargs, 324)

        class _base_full:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
        
        def __init__(self, **kwargs):
            self.compact = _spec_arg("compact", kwargs, self._base_compact())
            self.full = _spec_arg("full", kwargs, self._base_full())

    class _base_P4:
        class _base_compact:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
#                 self.Nbeam = _spec_arg("Nbeam", kwargs, 324)

        class _base_full:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
#                 self.Nbeam = _spec_arg("Nbeam", kwargs, 324)

        def __init__(self, **kwargs):
            self.compact = _spec_arg("compact", kwargs, self._base_compact())
            self.IDnorm = _spec_arg("IDnorm", kwargs, 0.5)
            self.ODnorm = _spec_arg("ODnorm", kwargs, 0.8)
            self.full = _spec_arg("full", kwargs, self._base_full())
            self.D = _spec_arg("D", kwargs, 0.0463)

    class _base_F3:
        class _base_compact:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.res = _spec_arg("res", kwargs, 30.0)

        class _base_full:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.res = _spec_arg("res", kwargs, 50.0)

        def __init__(self, **kwargs):
            self.compact = _spec_arg("compact", kwargs, self._base_compact())
            self.full = _spec_arg("full", kwargs, self._base_full())
            self.Rin = _spec_arg("Rin", kwargs, 2.8)
            self.Rout = _spec_arg("Rout", kwargs, inf)
            self.ang = _spec_arg("ang", kwargs, 180)

    class _base_Fend:
        class _base_compact:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
#                 self.res = _spec_arg("res", kwargs, 3)

        class _base_full:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
#                 self.res = _spec_arg("res", kwargs, 6)

        class _base_score:
            def __init__(self, **kwargs):
#                 self.Rin = _spec_arg("Rin", kwargs, 2.8)
                self.dummy = _spec_arg("dummy", kwargs, 1)
#                 self.Rout = _spec_arg("Rout", kwargs, 10)
#                 self.ang = _spec_arg("ang", kwargs, 180)

        class _base_corr:
            def __init__(self, **kwargs):
#                 self.Rin = _spec_arg("Rin", kwargs, 2.8)
                self.dummy = _spec_arg("dummy", kwargs, 1)
#                 self.Rout = _spec_arg("Rout", kwargs, 10)
#                 self.ang = _spec_arg("ang", kwargs, 180)

        def __init__(self, **kwargs):
            self.compact = _spec_arg("compact", kwargs, self._base_compact())
            self.FOV = _spec_arg("FOV", kwargs, 11)
            self.full = _spec_arg("full", kwargs, self._base_full())
            self.score = _spec_arg("score", kwargs, self._base_score())
            self.corr = _spec_arg("corr", kwargs, self._base_corr())
            self.sides = _spec_arg("sides", kwargs, "both")


    def __repr__(self):
        for k, v in self.__dict__:
            print('%s: '%(k), v)
        pass

    def printInfo(self, level=1):
        print('--------------------------\n')
        print('Number of keys: %d\n\n'%(len(self.__dict__.keys())))
        #for k in self.__dict__:
        #    if type(self.__dict__[k]) in [str, int, bool, float] or type(self.__dict__[k]) not np.array:
        #        print('%s: %s\n'%(k, str(type(self.__dict__[k])), str(self.__dict__[k])))
        #    else:
        #        print('%s: \ttype:%s value:\n'%(k, str(type(self.__dict__[k]))))
        
    def __str__(self):
        retstr = ''
        retstr += '--------------------------\n'
        retstr += ('Number of keys: %d\n\n'%(len(self.__dict__.keys())))
        #retsrt += '\n'
        for k in self.__dict__:
            if type(self.__dict__[k]) in [str, int, bool, float]:
                retstr += ('%s: \ttype:%s value: %s\n'%(k, str(type(self.__dict__[k])), str(self.__dict__[k])))
                #retstr += ('mp.%s\n'%(k))
            else:
                retstr += ('%s: \ttype:%s value:\n'%(k, str(type(self.__dict__[k]))))
                #retstr += ('mp.%s\n'%(k))
        retstr += '--------------------------\n'
        #retsrt += '\n'
        return retstr

    def __init__(self, **kwargs):
        

        import os 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)

#         self.compact = _spec_arg("compact", kwargs, self._base_compact())
#         self.full = _spec_arg("full", kwargs, self._base_full())

#         self.Nitr = _spec_arg("Nitr", kwargs, 10)
#         self.SPname = _spec_arg("SPname", kwargs, 0)
#         self.TToffset = _spec_arg("TToffset", kwargs, 1)
#         self.TrialNum = _spec_arg("TrialNum", kwargs, 1)
#         self.planetFlag = _spec_arg("planetFlag", kwargs, 0)
#         self.Nsbp = _spec_arg("Nsbp", kwargs, 1)
#         self.SeriesNum = _spec_arg("SeriesNum", kwargs, 1)
#         self.thput_radius = _spec_arg("thput_radius", kwargs, 0.7)
#         self.lambda0 = _spec_arg("lambda0", kwargs, 5.75e-07)
#         self.flagApod = _spec_arg("flagApod", kwargs, 0)
#         self.centering = _spec_arg("centering", kwargs, "pixel")
#         self.Nwpsbp = _spec_arg("Nwpsbp", kwargs, 1)
#         self.useGPU = _spec_arg("useGPU", kwargs, 0)
#         self.NlamForTT = _spec_arg("NlamForTT", kwargs, 1)
#         self.flagNewPSD = _spec_arg("flagNewPSD", kwargs, 0)
#         self.pup_strut_width = _spec_arg("pup_strut_width", kwargs, 0.0322)
        self.dm1 = _spec_arg("dm1", kwargs, self._base_dm1())
        self.dm2 = _spec_arg("dm2", kwargs, self._base_dm2())
        self.dm8 = _spec_arg("dm1", kwargs, self._base_dm8())
        self.dm9 = _spec_arg("dm1", kwargs, self._base_dm9())
#         self.whichPupil = _spec_arg("whichPupil", kwargs, "WFIRST20180103")
#         self.flagDM2stop = _spec_arg("flagDM2stop", kwargs, 0)
#         self.d_dm1_dm2 = _spec_arg("d_dm1_dm2", kwargs, 1)
        self.P2 = _spec_arg("P2", kwargs, self._base_P2())
        self.P3 = _spec_arg("P3", kwargs, self._base_P3())
        self.P1 = _spec_arg("P1", kwargs, self._base_P1())
        self.P4 = _spec_arg("P4", kwargs, self._base_P4())
#         self.relinItrVec = _spec_arg("relinItrVec", kwargs, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#         self.flagParfor = _spec_arg("flagParfor", kwargs, 0)
#         self.coro = _spec_arg("coro", kwargs, "LC")
#         self.Ntt = _spec_arg("Ntt", kwargs, 1)
#         self.controller = _spec_arg("controller", kwargs, "EFC")
#         self.logGmin = _spec_arg("logGmin", kwargs, -6)
#         self.fracBW = _spec_arg("fracBW", kwargs, 0.01)
#         self.LS_strut_width = _spec_arg("LS_strut_width", kwargs, 0.038)
#         self.fl = _spec_arg("fl", kwargs, 1)
        self.F3 = _spec_arg("F3", kwargs, self._base_F3())
        self.Fend = _spec_arg("Fend", kwargs, self._base_Fend())
#         self.d_P2_dm1 = _spec_arg("d_P2_dm1", kwargs, 0)
#         self.FPMampFac = _spec_arg("FPMampFac", kwargs, 0)
#         self.flagDM1stop = _spec_arg("flagDM1stop", kwargs, 0)
#         self.thput_eval_y = _spec_arg("thput_eval_y", kwargs, 0)
#         self.thput_eval_x = _spec_arg("thput_eval_x", kwargs, 6)
#         self.WspatialDef = _spec_arg("WspatialDef", kwargs, [])

