# import copy
import math
from pathlib import Path

import numpy
from numpy import inf

import falco
from falco.config.yaml_loader import object_constructor, load_from_str
from falco.util import _spec_arg
from falco.config import Probe, ProbeSchedule, Object


class ModelParameters(Object):

    class _BaseDm1(Object):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.Dstop = _spec_arg("Dstop", kwargs, 0.048)
            self.dx = _spec_arg("dx", kwargs, 0)
            self.NdmPad = _spec_arg("NdmPad", kwargs, 0)
            self.useDifferentiableModel = \
                _spec_arg("useDifferentiableModel", kwargs, False)
            self.surfFitMethod = _spec_arg('surfFitMethod', kwargs, 'lsq')
            self.ppact = _spec_arg('ppact', kwargs, 1)

    class _BaseDm2(Object):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.Dstop = _spec_arg("Dstop", kwargs, 0.048)
            self.dx = _spec_arg("dx", kwargs, 0)
            self.NdmPad = _spec_arg("NdmPad", kwargs, 0)
            self.useDifferentiableModel = _spec_arg("useDifferentiableModel", kwargs, False)
            self.surfFitMethod = _spec_arg('surfFitMethod', kwargs, 'lsq')
            self.ppact = _spec_arg('ppact', kwargs, 1)

    class _BaseDm8(Object):
        pass

    class _BaseDm9(Object):
        pass

    class _BaseP1(Object):
        class _BaseCompact(Object):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.Nbeam = _spec_arg("Nbeam", kwargs, 324)

        class _BaseFull(Object):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.Nbeam = _spec_arg("Nbeam", kwargs, 324)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.compact = _spec_arg("compact", kwargs, self._BaseCompact())
            self.full = _spec_arg("full", kwargs, self._BaseFull())
#             self.D = _spec_arg("D", kwargs, 2.3631)

    class _BaseP2(Object):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.D = _spec_arg("D", kwargs, 0.0463)

    class _BaseP3(Object):
        class _BaseCompact(Object):
            pass

        class _BaseFull(Object):
            pass

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.compact = _spec_arg("compact", kwargs, self._BaseCompact())
            self.full = _spec_arg("full", kwargs, self._BaseFull())

    class _BaseP4(Object):
        class _BaseCompact(Object):
            pass

        class _BaseFull(Object):
            pass

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.compact = _spec_arg("compact", kwargs, self._BaseCompact())
            self.IDnorm = _spec_arg("IDnorm", kwargs, 0.5)
            self.ODnorm = _spec_arg("ODnorm", kwargs, 0.8)
            self.full = _spec_arg("full", kwargs, self._BaseFull())
            self.D = _spec_arg("D", kwargs, 0.0463)

    class _BaseF3(Object):
        class _BaseCompact(Object):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.res = _spec_arg("res", kwargs, 30.0)

        class _BaseFull(Object):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.res = _spec_arg("res", kwargs, 50.0)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.compact = _spec_arg("compact", kwargs, self._BaseCompact())
            self.full = _spec_arg("full", kwargs, self._BaseFull())
            self.Rin = _spec_arg("Rin", kwargs, 2.8)
            self.Rout = _spec_arg("Rout", kwargs, inf)
            self.ang = _spec_arg("ang", kwargs, 180)

    class _BaseFend(Object):
        class _BaseCompact(Object):
            pass

        class _BaseFull(Object):
            pass

        class _BaseScore(Object):
            pass

        class _BaseCorr(Object):
            pass

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.compact = _spec_arg("compact", kwargs, self._BaseCompact())
            self.FOV = _spec_arg("FOV", kwargs, 11)
            self.full = _spec_arg("full", kwargs, self._BaseFull())
            self.score = _spec_arg("score", kwargs, self._BaseScore())
            self.corr = _spec_arg("corr", kwargs, self._BaseCorr())
            self.sides = _spec_arg("sides", kwargs, "both")


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

#         self.compact = _spec_arg("compact", kwargs, self._BaseCompact())
#         self.full = _spec_arg("full", kwargs, self._BaseFull())

#         self.Nitr = _spec_arg("Nitr", kwargs, 10)
#         self.SPname = _spec_arg("SPname", kwargs, 0)
#         self.TToffset = _spec_arg("TToffset", kwargs, 1)
#         self.TrialNum = _spec_arg("TrialNum", kwargs, 1)
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
        self.dm1 = _spec_arg("dm1", kwargs, self._BaseDm1())
        self.dm2 = _spec_arg("dm2", kwargs, self._BaseDm2())
        self.dm8 = _spec_arg("dm8", kwargs, self._BaseDm8())
        self.dm9 = _spec_arg("dm9", kwargs, self._BaseDm9())
#         self.whichPupil = _spec_arg("whichPupil", kwargs, "WFIRST20180103")
#         self.flagDM2stop = _spec_arg("flagDM2stop", kwargs, 0)
#         self.d_dm1_dm2 = _spec_arg("d_dm1_dm2", kwargs, 1)
        self.P2 = _spec_arg("P2", kwargs, self._BaseP2())
        self.P3 = _spec_arg("P3", kwargs, self._BaseP3())
        self.P1 = _spec_arg("P1", kwargs, self._BaseP1())
        self.P4 = _spec_arg("P4", kwargs, self._BaseP4())
#         self.relinItrVec = _spec_arg("relinItrVec", kwargs, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#         self.flagParfor = _spec_arg("flagParfor", kwargs, 0)
#         self.coro = _spec_arg("coro", kwargs, "LC")
#         self.Ntt = _spec_arg("Ntt", kwargs, 1)
#         self.controller = _spec_arg("controller", kwargs, "EFC")
#         self.logGmin = _spec_arg("logGmin", kwargs, -6)
#         self.fracBW = _spec_arg("fracBW", kwargs, 0.01)
#         self.LS_strut_width = _spec_arg("LS_strut_width", kwargs, 0.038)
#         self.fl = _spec_arg("fl", kwargs, 1)
        self.F3 = _spec_arg("F3", kwargs, self._BaseF3())
        self.Fend = _spec_arg("Fend", kwargs, self._BaseFend())
#         self.d_P2_dm1 = _spec_arg("d_P2_dm1", kwargs, 0)
#         self.FPMampFac = _spec_arg("FPMampFac", kwargs, 0)
#         self.flagDM1stop = _spec_arg("flagDM1stop", kwargs, 0)
#         self.thput_eval_y = _spec_arg("thput_eval_y", kwargs, 0)
#         self.thput_eval_x = _spec_arg("thput_eval_x", kwargs, 6)
#         self.WspatialDef = _spec_arg("WspatialDef", kwargs, [])

        super().__init__(**kwargs)

    @staticmethod
    def from_yaml(text, context=None):
        """
        Construct a ModelParameters object from a yaml string.

        All basic dictionaries are instead deserialized as `Object` instances.
        Includes special constructors for `Probe`, and `ProbeSchedule`, using yaml tags. E.g.:

        ```
        myProbe: !Probe
            a: 5
            b: a string
        ```

        The above will create a `Probe` instance with the `a` and `b` fields set accordingly, rather than
        creating the usual `Object` instance.

        You can also create a default `Probe` with `!Probe {}`.

        You can also write python expressions and evaluate them with `!eval`. This is useful for a few things:
        - Numpy expressions. Numpy is available as `np`. Ex: `my_array: !eval np.array([1, 2])`
        - Referring to falco objects. Ex: `inf_fn: !eval falco.INFLUENCE_BMC_2K`
        - Self-referential expressions. These expressions are evaluated lazily, and can refer to other fields in
          the model parameters object under the name `mp`. Ex: `Zcoef: !eval 1e-9*np.ones(np.size(mp.jac.zerns))`

          Circular dependencies between expressions will be detected and an error will be raised.

        :param text: a yaml string to deserialize
        :param context addition local variables to expose to eval code
        :return: a `ModelParameters` instance
        """

        result = ModelParameters()

        if context is None:
            context = {}
        context['mp'] = result

        data = load_from_str(
            text,
            {'np': numpy, 'falco': falco, 'math': math},
            context,
            {
                "!Probe": object_constructor(Probe),
                "!ProbeSchedule": object_constructor(ProbeSchedule)
            }
        )

        result.__init__(**data)
        return result

    @staticmethod
    def from_yaml_file(path_string, context=None):
        """
        Reads the yaml file at the given path and passes it to `ModelParameters.from_yaml`.

        See `ModelParameters.from_yaml` for info.
        """
        return ModelParameters.from_yaml(Path(path_string).read_text(), context)
