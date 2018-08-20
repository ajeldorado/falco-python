import copy
from numpy import inf
import numpy as np
import falco.masks
import falco.utils
from falco.utils import _spec_arg
import collections
from falco import models


class ModelParameters:
    class _base_dm1:
        def __init__(self, **kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.Dstop = _spec_arg("Dstop", kwargs, 0.048)

    class _base_dm2:
        def __init__(self, **kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.Dstop = _spec_arg("Dstop", kwargs, 0.048)

    class _base_P2:
        def __init__(self, **kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.D = _spec_arg("D", kwargs, 0.0463)

    class _base_P3:
        def __init__(self, **kwargs):
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.D = _spec_arg("D", kwargs, 0.0463)

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
            self.D = _spec_arg("D", kwargs, 2.3631)

    class _base_P4:
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
            self.Rin = _spec_arg("Rin", kwargs, 2.8)
            self.Rout = _spec_arg("Rout", kwargs, inf)
            self.full = _spec_arg("full", kwargs, self._base_full())
            self.ang = _spec_arg("ang", kwargs, 180)

    class _base_F4:
        class _base_compact:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.res = _spec_arg("res", kwargs, 3)

        class _base_full:
            def __init__(self, **kwargs):
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.res = _spec_arg("res", kwargs, 6)

        class _base_score:
            def __init__(self, **kwargs):
                self.Rin = _spec_arg("Rin", kwargs, 2.8)
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.Rout = _spec_arg("Rout", kwargs, 10)
                self.ang = _spec_arg("ang", kwargs, 180)

        class _base_corr:
            def __init__(self, **kwargs):
                self.Rin = _spec_arg("Rin", kwargs, 2.8)
                self.dummy = _spec_arg("dummy", kwargs, 1)
                self.Rout = _spec_arg("Rout", kwargs, 10)
                self.ang = _spec_arg("ang", kwargs, 180)

        def __init__(self, **kwargs):
            self.compact = _spec_arg("compact", kwargs, self._base_compact())
            self.FOV = _spec_arg("FOV", kwargs, 11)
            self.full = _spec_arg("full", kwargs, self._base_full())
            self.score = _spec_arg("score", kwargs, self._base_score())
            self.corr = _spec_arg("corr", kwargs, self._base_corr())
            self.sides = _spec_arg("sides", kwargs, "both")

    def __init__(self, **kwargs):
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

    def get_PSF_norm_factor(self, DM):
        """
        Compute the normalization value for the compact and full models.  The normalization value is
        the peak intensity for an on-axis object with the entire coronagraph in place, except with
        the focal plane mask removed.

        Parameters
        ----------
        DM : DeformableMirrorParameters (placeholder class for now)
            Parameter structure for deformable mirrors

        """
        modvar = {
            'flagCalcJac': 0,
            'ttIndex': 1,  # 1 is the zero-offset tip/tilt setting
            'whichSource': 'star'
        }

        self.F4.compact.I00 = np.ones((1, self.Nsbp), dtype=np.float64)
        self.F4.full.I00 = np.ones((1, self.Nsbp), dtype=np.float64)

        for si in range(self.Nsbp):
            Im_temp_full = np.zeros((self.F4.full.Neta, self.F4.full.Nxi, self.Nwpsbp),
                                    dtype=np.float64)

            for wi in range(self.Nwpsbp):
                modvar['sbpIndex'] = si
                modvar['wpsbpIndex'] = wi

                EtempFull = models.model_full(self, DM, modvar)
                Im_temp_full[:, :, wi] = np.abs(EtempFull) ** 2

            EtempCompact = models.model_compact(self, DM, modvar)
            Im_temp_compact = np.abs(EtempCompact) ** 2

            self.F4.full.I00[si] = np.mean(Im_temp_full, axis=2).max()
            self.F4.compact.I00[si] = np.mean(Im_temp_compact)

        modvar['flagGetNormVal'] = False  # Not sure if/why this is needed

    def init_ws(self):
        # MATLAB prints were commented out but left for clarity
        #disp(['DM 1-2 Fresnel number = ',num2str((mp.P2.D/2)^2/(mp.d_dm1_dm2*mp.lambda0))]);
        self.si_ref = int(np.ceil(self.Nsbp/2.0))
        self.wi_ref = int(np.ceil(self.Nwpsbp/2.0))
        self.fracBWsbp = self.fracBW/self.Nsbp

        #fprintf(' Using %d discrete wavelength(s) in each of %d sub-bandpasses over a %.1f%% total
        # bandpass \n\n', mp.Nwpsbp, mp.Nsbp,100*mp.fracBW) ;

        # Tip/Tilt and Spatial Weighting of the Control Jacobian  #NEWFORTIPTILT
        self.ti_ref = int(np.ceil(self.Ntt/2.0))
        # Conversion factor: milliarcseconds (mas) to lambda0/D
        mas2lam0D = 1.0/(self.lambda0/self.P1.D*180/np.pi*3600*1000)

        # Define the (x,y) values for each tip/tilt offset in units of lambda0/D
        assert(self.Ntt in (1, 4, 5))

        if self.Ntt == 5:
            offsets = np.radians(np.arange(0, 360, 90))

        elif self.Ntt == 4:
            offsets = np.radians(np.arange(0, 360, 120))

        else:
            offsets = []

        # Tip-tilt values
        self.ttx = np.concatenate([np.zeros(1), mas2lam0D*self.TToffset * np.cos(offsets)])
        self.tty = np.concatenate([np.zeros(1), mas2lam0D*self.TToffset * np.sin(offsets)])

        if np.isinf(self.NlamForTT):
            # Full usage and equal weighting for all T/T's and sub-bandpasses.
            self.Wttlam = np.ones(self.Nsbp, self.Ntt)

        else:
            assert(self.NlamForTT in range(4))

            # Initialize weighting matrix of each tip/tilt-wavelength mode for the controller
            self.Wttlam = np.zeros((self.Nsbp, self.Ntt))

            if self.NlamForTT == 3:
                # Set tip/tilt offsets at the middle and both end sub-bandpasses.
                self.Wttlam[:, 0] = self.Ntt*np.ones((self.Nsbp, 1))
                self.Wttlam[0, :] = np.ones((1, self.Ntt))
                self.Wttlam[self.si_ref - 1, :] = np.ones((1, self.Ntt))
                self.Wttlam[-1, :] = np.ones((1, self.Ntt))

            elif self.NlamForTT == 2:
                # Set tip/tilt offsets at only both end sub-bandpasses.
                self.Wttlam[:, 0] = self.Ntt*np.ones((self.Nsbp, 1))
                self.Wttlam[0, :] = np.ones((1, self.Ntt))
                self.Wttlam[-1, :] = np.ones((1, self.Ntt))

            elif self.NlamForTT == 1:
                # Set tip/tilt offsets at only the middle sub-bandpass.
                self.Wttlam[:, 0] = self.Ntt*np.ones((self.Nsbp, 1))
                self.Wttlam[self.si_ref-1, :] = np.ones((1, self.Ntt))

            elif self.NlamForTT == 0:
                # Set tip/tilt offsets at no sub-bandpasses.
                self.Wttlam[:, 0] = self.Ntt*np.ones((self.Nsbp, 1))

        self.Wsum = np.sum(self.Wttlam)  # Sum of all the control Jacobian weights

        # Indices of the non-zero control Jacobian modes in the weighting matrix
        self.Wttlam_ele = np.where(self.Wttlam > 0)[0]
        self.WttlamVec = self.Wttlam[self.Wttlam_ele]  # Vector of control Jacobian mode weights
        self.Nttlam = len(self.Wttlam_ele)  # Number of modes in the control Jacobian

        # Get the wavelength indices for the nonzero values in the weight matrix.
        si, ti = np.meshgrid(range(self.Nsbp), range(self.Ntt))
        self.Wttlam_si = si[self.Wttlam_ele]
        self.Wttlam_ti = ti[self.Wttlam_ele]

        # A temporary hack since MATLAB's config files don't create the following subclasses:
        self.P2.full = copy.deepcopy(self.P1.full)
        self.P2.compact = copy.deepcopy(self.P1.compact)
        self.dm1.full = copy.deepcopy(self.P1.full)
        self.dm1.compact = copy.deepcopy(self.P1.compact)
        self.dm2.full = copy.deepcopy(self.P1.full)
        self.dm2.compact = copy.deepcopy(self.P1.compact)

        # Input pupil plane resolution, masks, and coordinates
        # Resolution at input pupil and DM1 and DM2
        self.P2.full.dx = self.P2.D/self.P2.full.Nbeam
        self.P2.compact.dx = self.P2.D/self.P2.compact.Nbeam

        assert(self.whichPupil == "WFIRST20180103")  # Only this one is supported so far
        if self.whichPupil == "WFIRST20180103":
            # Generate high-res input pupil for the 'full' model
            self.P1.full.mask = falco.masks.falco_gen_pupil_WFIRST_20180103(
                self.P1.full.Nbeam, self.centering)
            # Generate low-res input pupil for the 'compact' model
            self.P1.compact.mask = falco.masks.falco_gen_pupil_WFIRST_20180103(
                self.P1.compact.Nbeam, self.centering)

        # Total number of pixels across array containing the pupil in the full model. Add 2 pixels
        # to Nbeam when the beam is pixel-centered.
        self.P1.full.Narr = len(self.P1.full.mask)

        # Number of pixels across the array containing the input pupil in the compact model
        self.P1.compact.Narr = len(self.P1.compact.mask)
        self.sumPupil = np.sum(np.abs(self.P1.full.mask) ** 2)

        # NORMALIZED (in pupil diameter) coordinate grids in the input pupil for making the
        # tip/tilted input wavefront within the compact and full models
        if self.centering == "interpixel":
            self.P2.full.xsDL = np.arange(
                -(self.P1.full.Narr - 1) / 2,
                (self.P1.full.Narr + 1) / 2) * self.P2.full.dx / self.P2.D

            self.P2.compact.xsDL = np.arange(
                -(self.P1.compact.Narr - 1) / 2,
                (self.P1.compact.Narr + 1) / 2) * self.P2.compact.dx / self.P2.D

        else:
            self.P2.full.xsDL = np.arange(
                -self.P1.full.Narr / 2, self.P1.full.Narr / 2) * self.P2.full.dx / self.P2.D

            self.P2.compact.xsDL = np.arange(
                -self.P1.compact.Narr / 2,
                self.P1.compact.Narr / 2) * self.P2.compact.dx / self.P2.D

        self.P2.full.XsDL, self.P2.full.YsDL = np.meshgrid(self.P2.full.xsDL, self.P2.full.xsDL)
        self.P2.compact.XsDL, self.P2.compact.YsDL = np.meshgrid(
            self.P2.compact.xsDL, self.P2.compact.xsDL)

        # DM aperture stops
        self.dm1.full.mask = falco.masks.falco_gen_DM_stop(
            self.P2.full.dx, self.dm1.Dstop, self.centering)
        self.dm1.compact.mask = falco.masks.falco_gen_DM_stop(
            self.P2.compact.dx, self.dm1.Dstop, self.centering)

        self.dm2.full.mask = falco.masks.falco_gen_DM_stop(
            self.P2.full.dx, self.dm2.Dstop, self.centering)
        self.dm2.compact.mask = falco.masks.falco_gen_DM_stop(
            self.P2.compact.dx, self.dm2.Dstop, self.centering)

        assert(not self.flagApod)  # Only this is supported so far

        # Lyot plane resolution, coordinates, and cropped-down mask for compact model
        # Resolution at Lyot Plane
        self.P4.full.dx = self.P4.D / self.P4.full.Nbeam
        self.P4.compact.dx = self.P4.D / self.P4.compact.Nbeam

        assert(self.whichPupil == "WFIRST20180103")  # Only this one is supported so far

        if self.whichPupil == "WFIRST20180103":
            # Make or read in Lyot stop (LS) for the 'full' model
            self.P4.full.mask = falco.masks.falco_gen_pupil_WFIRSTcycle6_LS(
                self.P4.full.Nbeam, self.P4.D, self.P4.IDnorm, self.P4.ODnorm,
                self.LS_strut_width, self.centering, True)
            self.P4.compact.mask = falco.masks.falco_gen_pupil_WFIRSTcycle6_LS(
                self.P4.compact.Nbeam, self.P4.D, self.P4.IDnorm, self.P4.ODnorm,
                self.LS_strut_width, self.centering, True)

        assert(self.coro not in ("Vortex", "vortex", "AVC",
                                 "VC", "LUVOIRA5predef"))  # Not inplemented yet

        if self.coro not in ("Vortex", "vortex", "AVC", "VC", "LUVOIRA5predef"):
            # Crop down the high-resolution Lyot stop to get rid of extra zero padding
            LSsum = np.sum(self.P4.full.mask)
            LSdiff = 0
            counter = 2

            while abs(LSdiff) <= 1e-7:
                self.P4.full.Narr = len(self.P4.full.mask) - counter
                # Subtract an extra 2 to negate the extra step that overshoots.
                LSdiff = (LSsum -
                          falco.utils.padOrCropEven(self.P4.full.mask, self.P4.full.Narr - 2).sum())
                counter = counter + 2

            # The cropped-down Lyot stop for the full model.
            self.P4.full.croppedMask = falco.utils.padOrCropEven(
                self.P4.full.mask, self.P4.full.Narr)

            # Crop down the low-resolution Lyot stop to get rid of extra zero padding. Speeds up the compact model.
            LSsum = np.sum(self.P4.compact.mask)
            LSdiff = 0
            counter = 2

            while abs(LSdiff) <= 1e-7:
                # Number of points across the cropped-down Lyot stop
                self.P4.compact.Narr = len(self.P4.compact.mask) - counter
                # Subtract an extra 2 to negate the extra step that overshoots.
                LSdiff = (LSsum -
                          falco.utils.padOrCropEven(self.P4.compact.mask,
                                                    self.P4.compact.Narr - 2).sum())
                counter += 2

            # The cropped-down Lyot stop for the compact model
            self.P4.compact.croppedMask = falco.utils.padOrCropEven(
                self.P4.compact.mask, self.P4.compact.Narr)

            # (METERS) Lyot plane coordinates (over the cropped down to Lyot stop mask) for MFTs in
            # the compact model from the FPM to the LS.
            if self.centering == "interpixel":
                self.P4.compact.xs = np.arange(-(self.P4.compact.Narr - 1) / 2,
                                               (self.P4.compact.Narr + 1) / 2) * self.P4.compact.dx
            else:
                self.P4.compact.xs = np.arange(-self.P4.compact.Narr / 2,
                                               self.P4.compact.Narr / 2) * self.P4.compact.dx

            self.ysLScompactCrop = self.P4.compact.xs

        assert(self.coro in ("LC", "DMLC", "APLC"))  # Only these are implemented so far

        if self.coro in ("LC", "DMLC", "APLC"):
            # Occulting spot FPM
            # Make or read in focal plane mask (FPM) amplitude for the full model
            self.F3.full.mask = collections.namedtuple("_F3_full_mask", "amp")
            self.F3.full.mask.amp = falco.masks.falco_gen_annular_FPM(
                self.F3.full.res, self.F3.Rin, self.F3.Rout, self.FPMampFac, self.centering, False)
            self.F3.full.Neta, self.F3.full.Nxi = self.F3.full.mask.amp.shape

            # Number of points across the FPM in the compact model
            if np.isinf(self.F3.Rout):
                if self.centering == "pixel":
                    self.F3.compact.Nxi = falco.utils.ceil_even(
                        2 * (self.F3.Rin * self.F3.compact.res + 0.5))
                else:
                    self.F3.compact.Nxi = falco.utils.ceil_even(
                        2 * self.F3.Rin * self.F3.compact.res)

            else:
                if self.centering == "pixel":
                    self.F3.compact.Nxi = falco.utils.ceil_even(
                        2 * (self.F3.Rout*self.F3.compact.res + 0.5))
                else:
                    self.F3.compact.Nxi = falco.utils.ceil_even(
                        2 * self.F3.Rout * self.F3.compact.res)

            self.F3.compact.Neta = self.F3.compact.Nxi

            # Coordinates for the FPMs in the full and compact models
            if self.centering == "interpixel" or self.F3.full.Nxi % 2 == 1:
                self.F3.full.xisDL = np.arange(-(self.F3.full.Nxi - 1) / 2,
                                               (self.F3.full.Nxi - 1) / 2 + 1) / self.F3.full.res
                self.F3.full.etasDL = np.arange(-(self.F3.full.Neta - 1) / 2,
                                                (self.F3.full.Neta - 1) / 2 + 1) / self.F3.full.res

                self.F3.compact.xisDL = np.arange(
                    -(self.F3.compact.Nxi - 1) / 2,
                    (self.F3.compact.Nxi - 1) / 2 + 1) / self.F3.compact.res
                self.F3.compact.etasDL = np.arange(
                    -(self.F3.compact.Neta - 1) / 2,
                    (self.F3.compact.Neta - 1) / 2 + 1) / self.F3.compact.res

            else:
                self.F3.full.xisDL = np.arange(-self.F3.full.Nxi / 2,
                                               self.F3.full.Nxi / 2) / self.F3.full.res
                self.F3.full.etasDL = np.arange(-self.F3.full.Neta / 2,
                                                self.F3.full.Neta / 2) / self.F3.full.res

                self.F3.compact.xisDL = np.arange(
                    -self.F3.compact.Nxi / 2,
                    self.F3.compact.Nxi / 2) / self.F3.compact.res
                self.F3.compact.etasDL = np.arange(
                    -self.F3.compact.Neta / 2,
                    self.F3.compact.Neta / 2) / self.F3.compact.res

            # Make or read in focal plane mask (FPM) amplitude for the compact model
            self.F3.compact.mask = collections.namedtuple("_F3_compact_mask", "amp")
            self.F3.compact.mask.amp = falco.masks.falco_gen_annular_FPM(
                self.F3.compact.res, self.F3.Rin, self.F3.Rout, self.FPMampFac,
                self.centering, rot180=False)

        # FPM coordinates
        assert(self.coro not in ("Vortex", "vortex", "AVC",
                                 "VC", "LUVOIRA5predef"))  # Not inplemented yet
        if self.coro not in ("Vortex", "vortex", "AVC", "VC", "LUVOIRA5predef"):
            self.F3.full.dxi = (self.fl * self.lambda0 / self.P2.D) / self.F3.full.res
            self.F3.full.deta = self.F3.full.dxi
            self.F3.compact.dxi = (self.fl * self.lambda0 / self.P2.D) / self.F3.compact.res
            self.F3.compact.deta = self.F3.compact.dxi

            # Compute coordinates in plane of FPM in the compact model (in meters)
            if self.centering == "interpixel" or self.F3.compact.Nxi % 2 == 1:
                self.F3.compact.xis = np.arange(
                    -(self.F3.compact.Nxi - 1) / 2,
                    (self.F3.compact.Nxi - 1) / 2 + 1) * self.F3.compact.dxi

                self.F3.compact.etas = np.arange(
                    -(self.F3.compact.Neta-1) / 2,
                    (self.F3.compact.Neta-1) / 2 + 1) * self.F3.compact.deta
            else:
                self.F3.compact.xis = np.arange(-self.F3.compact.Nxi / 2,
                                                self.F3.compact.Nxi / 2) * self.F3.compact.dxi
                self.F3.compact.etas = np.arange(-self.F3.compact.Neta / 2,
                                                 self.F3.compact.Neta / 2) * self.F3.compact.deta

        # Sampling/Resolution and Scoring/Correction Masks for 2nd Focal Plane
        self.F4.compact.dxi = self.fl * self.lambda0 / self.P4.D / self.F4.compact.res
        self.F4.compact.deta = self.F4.compact.dxi

        self.F4.full.dxi = self.fl * self.lambda0 / self.P4.D / self.F4.full.res
        self.F4.full.deta = self.F4.full.dxi

        # Software Mask for Correction
        maskCorr = {}
        maskCorr["pixresFP"] = self.F4.compact.res
        maskCorr["rhoInner"] = self.F4.corr.Rin  # lambda0/D
        maskCorr["rhoOuter"] = self.F4.corr.Rout  # lambda0/D
        maskCorr["angDeg"] = self.F4.corr.ang  # degrees
        maskCorr["centering"] = self.centering
        maskCorr["FOV"] = self.F4.FOV
        maskCorr["whichSide"] = self.F4.sides  # which (sides) of the dark hole have open

        self.F4.compact.corr = collections.namedtuple(
            "_F4_compact_corr", "mask xisDL etasDL settings inds")

        self.F4.compact.corr.mask, self.F4.compact.xisDL, self.F4.compact.etasDL = \
            falco.masks.falco_gen_SW_mask(**maskCorr)

        # Store values for future reference
        self.F4.compact.corr.settings = collections.namedtuple(
            "_F4_compact_corr_settings", maskCorr.keys())(**maskCorr)

        maskCorr["pixresFP"] = self.F4.full.res

        self.F4.full.corr = collections.namedtuple(
            "_F4_full_corr", "mask xisDL etasDL settings inds")

        self.F4.full.corr.mask, self.F4.full.xisDL, self.F4.full.etasDL = \
            falco.masks.falco_gen_SW_mask(**maskCorr)

        # Store values for future reference
        self.F4.full.corr.settings = collections.namedtuple(
            "_F4_full_corr_settings", maskCorr.keys())(**maskCorr)

        # Software Mask for Scoring Contrast
        maskScore = {}
        maskScore["pixresFP"] = self.F4.compact.res
        maskScore["rhoInner"] = self.F4.score.Rin  # lambda0/D
        maskScore["rhoOuter"] = self.F4.score.Rout  # lambda0/D
        maskScore["angDeg"] = self.F4.score.ang  # degrees
        maskScore["centering"] = self.centering
        maskScore["FOV"] = self.F4.FOV
        maskScore["whichSide"] = self.F4.sides  # which (sides) of the dark hole have open

        # Store values for future reference
        self.F4.compact.score = collections.namedtuple("_F4_compact_score", "mask settings inds")
        self.F4.compact.score.mask, _, _ = falco.masks.falco_gen_SW_mask(**maskScore)
        self.F4.compact.score.settings = collections.namedtuple(
            "_F4_compact_score_settings", maskScore.keys())(**maskScore)

        maskScore["pixresFP"] = self.F4.full.res
        self.F4.full.score = collections.namedtuple("_F4_full_score", "mask settings inds")
        self.F4.full.score.mask, _, _ = falco.masks.falco_gen_SW_mask(**maskScore)

        # Store values for future reference
        self.F4.full.score.settings = collections.namedtuple(
            "_F4_full_score_settings", maskScore.keys())(**maskScore)

        # Indices of dark hole pixels
        self.F4.compact.corr.inds = np.where(self.F4.compact.corr.mask != 0)
        self.F4.compact.score.inds = np.where(self.F4.compact.score.mask != 0)

        self.F4.full.score.inds = np.where(self.F4.full.score.mask != 0)
        self.F4.full.corr.inds = np.where(self.F4.full.corr.mask != 0)

        self.F4.compact.Neta, self.F4.compact.Nxi = self.F4.compact.score.mask.shape
        self.F4.full.Neta, self.F4.full.Nxi = self.F4.full.score.mask.shape

        # Spatial weighting of pixel intensity (compac model only since for control)
        XISLAMD, ETASLAMD = np.meshgrid(self.F4.compact.xisDL, self.F4.compact.etasDL)
        RHOS = np.hypot(XISLAMD, ETASLAMD)

        self.Wspatial = self.F4.compact.corr.mask

        assert(len(self.WspatialDef) == 0)  # Not implemented yet
        self.Wspatial_ele = self.Wspatial[self.F4.compact.corr.inds]

        # Bandpass parameters
        if self.Nsbp > 1:
            # Set so that the edge of the extremum sub-bandpasses match the edge of the entire
            # bandpass.
            # Vector of central wavelengths for each sub-bandpass
            self.sbp_center_vec = np.linspace(
                self.lambda0 * (1 - (self.fracBW - self.fracBWsbp) / 2),
                self.lambda0 * (1 + (self.fracBW - self.fracBWsbp) / 2),
                self.Nsbp)

        else:
            self.sbp_center_vec = self.lambda0

        # Fill the whole band if just one sub-bandpass
        if self.Nwpsbp > 1 and self.Nsbp == 1:
            self.lamFac_vec = np.linspace(
                # vector of lambda factors about the center wavelength of each sub-bandpass
                1 - self.fracBWsbp / 2.0, 1 + self.fracBWsbp / 2.0, self.Nwpsbp)

        elif self.Nwpsbp > 1:
            # Evenly spaced so that in full bandpass no wavelength is weighted more heavily.
            self.lamFac_vec = np.linspace(
                1 - (self.fracBWsbp / 2.0) * (1 - 1.0 / self.Nwpsbp),
                1 + (self.fracBWsbp / 2.0) * (1 - 1.0 / self.Nwpsbp),
                self.Nwpsbp)

        else:
            self.lamFac_vec = 1

        self.lam_array = np.outer(self.lamFac_vec, self.sbp_center_vec)

        # Initial Electric Fields for Star and Exoplanet
        # Starlight. Can add some propagation here to create an aberrate wavefront
        # starting from a primary mirror.
        assert(self.whichPupil != "LUVOIRA5predef")  # Not implemented yet
        if self.whichPupil != "LUVOIRA5predef":
            # Input E-field at entrance pupil
            self.P1.full.E = np.ones((self.P1.full.Narr, self.P1.full.Narr, self.Nwpsbp, self.Nsbp))

            # Initialize the input E-field for the planet at the entrance pupil. Will apply the
            # phase ramp later
            self.Eplanet = self.P1.full.E
            self.P1.compact.E = np.ones((self.P1.compact.Narr, self.P1.compact.Narr, self.Nsbp))

        # Off-axis, incoherent point source (exoplanet)
        self.c_planet = 1  # contrast of exoplanet
        self.x_planet = 6  # x position of exoplanet in lambda0/D
        self.y_planet = 0  # 7 position of exoplanet in lambda0/D

        # Field Stop
        self.F4.compact.mask = np.ones((self.F4.compact.Neta, self.F4.compact.Nxi))

        # Contrast to Normalized Intensity Map Calculation

        #fprintf('Beginning Trial %d of Series %d.\n',mp.TrialNum,mp.SeriesNum);

        # Get the starlight normalization factor for the compact and full models (to convert images
        # to normalized intensity)
        # TODO: uncommentd when get_PSF_norm_factor is implemented and DM is initialized elsewhere
        #mp = self.get_PSF_norm_factor(DM);

        XIS, ETAS = np.meshgrid(self.F4.full.xisDL - self.x_planet,
                                self.F4.full.etasDL - self.y_planet)
        self.FP4 = collections.namedtuple("_FP4", "compact")
        self.FP4.compact = collections.namedtuple("_FP4_compact", "RHOS")
        self.FP4.compact.RHOS = np.hypot(XIS, ETAS)
        self.maskHMcore = np.zeros_like(self.FP4.compact.RHOS)
        self.maskCore = np.zeros_like(self.FP4.compact.RHOS)
        self.maskCore[self.FP4.compact.RHOS <= self.thput_radius] = 1

        self.thput_vec = np.zeros((self.Nitr, 1))
