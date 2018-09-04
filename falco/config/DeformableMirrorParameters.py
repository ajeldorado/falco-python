import numpy as np
import os
import scipy.io
import scipy.interpolate
import falco.utils
from falco.utils import _spec_arg
import collections

_influence_function_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "influence_dm5v2.mat")

def falco_gen_dm_poke_cube(dm, mp, dx_dm, flagGenCube=True):
    #Compute sampling of the pupil. Assume that it is square.
    dm.dx_dm = dx_dm
    dm.dx = dx_dm

    # Compute coordinates of original influence function
    Ninf0 = len(dm.inf0) #Number of points across the influence function at its native resolution
    #x_inf0 = (-(Ninf0-1)/2:(Ninf0-1)/2)*dm.dx_inf0 # True for even- or odd-sized influence function maps as long as they are centered on the array.
    x_inf0 = np.arange(-(Ninf0-1)//2, (Ninf0-1)//2)*dm.dx_inf0

    Ndm0 = falco.utils.ceil_even( Ninf0 + (dm.Nact - 1)*(dm.dm_spacing/dm.dx_inf0) ) #Number of points across the DM surface at native influence function resolution
    dm.NdmMin = falco.utils.ceil_even( Ndm0*(dm.dx_inf0/dm.dx))+2 #Number of points across the (un-rotated) DM surface at new, desired resolution.
    dm.Ndm = falco.utils.ceil_even( max(np.abs([np.sqrt(2)*np.cos(np.radians(45-dm.zrot)), np.sqrt(2)*np.sin(np.radians(45-dm.zrot))]))*Ndm0*(dm.dx_inf0/dm.dx))+2 #Number of points across the array to fully contain the DM surface at new, desired resolution and z-rotation angle.

    Xinf0,Yinf0 = np.meshgrid(x_inf0, x_inf0)

    #Compute list of initial actuator center coordinates (in actutor widths).
    dm.Xact, dm.Yact = np.meshgrid(np.arange(dm.Nact)-dm.xc, np.arange(dm.Nact)-dm.yc) # in actuator widths
    x_vec = dm.Xact.reshape(-1)
    y_vec = dm.Yact.reshape(-1)

    dm.NactTotal = len(x_vec) #Total number of actuators in the 2-D array


    sa = np.sin(np.radians(dm.xtilt))
    ca = np.cos(np.radians(dm.xtilt))
    sb = np.sin(np.radians(dm.ytilt))
    cb = np.cos(np.radians(dm.ytilt))
    sg = np.sin(np.radians(-dm.zrot))
    cg = np.cos(np.radians(-dm.zrot))

    Mrot = [[cb * cg, sa * sb * cg - ca * sg, ca * sb * cg + sa * sg, 0.0], \
            [cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - sa * cg, 0.0], \
            [    -sb,                sa * cb,                ca * cb, 0.0], \
            [    0.0,                    0.0,                    0.0, 1.0]]
  
    dm.xy_cent_act = []
    for iact in range(dm.NactTotal):
        xyzVals = [x_vec[iact], y_vec[iact], 0, 1]
        xyzValsRot = np.dot(Mrot, xyzVals)
        dm.xy_cent_act.append(xyzValsRot[:2])

    dm.xy_cent_act = np.array(dm.xy_cent_act).T[::-1]

  
    N0 = max(dm.inf0.shape)
    Npad = falco.utils.ceil_odd( np.sqrt(2)*N0 )
    inf0pad = np.zeros((Npad,Npad))

    fromx = int(np.ceil(Npad/2)-np.floor(N0/2))
    tox = int(np.ceil(Npad/2)+np.floor(N0/2)) + 1
    inf0pad[fromx:tox, fromx:tox] = dm.inf0

    ydim,xdim = inf0pad.shape

    xd2  = np.floor(xdim / 2) + 1
    yd2  = np.floor(ydim / 2) + 1
    cx   = np.arange(xdim) - xd2
    cy   = np.arange(ydim) - yd2
    Xs0, Ys0 = np.meshgrid(cx, cy)

    xsNew = 0*Xs0
    ysNew = 0*Ys0

    for ii in range(xdim):
        for jj in range(ydim):
            xyzVals = [Xs0[ii,jj], Ys0[ii,jj], 0, 1]
            xyzValsRot = np.dot(Mrot, xyzVals)
            xsNew[ii,jj] = xyzValsRot[0]
            ysNew[ii,jj] = xyzValsRot[1]


    # Calculate the interpolated DM grid (set extrapolated values to 0.0)
    dm.infMaster = scipy.interpolate.griddata((xsNew.reshape(-1),ysNew.reshape(-1)), inf0pad.reshape(-1), (Xs0.reshape(-1),Ys0.reshape(-1)), "cubic")
    dm.infMaster[np.isnan(dm.infMaster)] = 0
    dm.infMaster = dm.infMaster.reshape(inf0pad.shape)
 
    #Crop down the influence function until it has no zero padding left
    infSum = np.sum(dm.infMaster)
    infDiff = 0
    counter = 2
    while abs(infDiff) <= 1e-7:
        counter += 2
        Ninf0pad = len(dm.infMaster)-counter #Number of points across the rotated, cropped-down influence function at the original resolution
        infDiff = infSum - np.sum( dm.infMaster[1+counter//2:1-counter//2,1+counter//2:1-counter//2] ) #Subtract an extra 2 to negate the extra step that overshoots.

    counter -= 2
    Ninf0pad = len(dm.infMaster)-counter
    infMaster2 = dm.infMaster[1+counter//2:1-counter//2,1+counter//2:1-counter//2]

    dm.infMaster = infMaster2
    Npad = Ninf0pad

    x_inf0 = np.arange(-(Npad-1)//2, (Npad+1)//2)*dm.dx_inf0 # True for even- or odd-sized influence function maps as long as they are centered on the array.
    Xinf0,Yinf0 = np.meshgrid(x_inf0,x_inf0)

    #Compute the size of the postage stamps.
    dm.Nbox = falco.utils.ceil_even(Ninf0pad*dm.dx_inf0//dx_dm) # Number of points across the influence function in the pupil file's spacing. Want as even
    #Also compute their padded sizes for the angular spectrum (AS) propagation between P2 and DM1 or between DM1 and DM2
    Nmin = falco.utils.ceil_even( max(mp.sbp_center_vec)*max(np.abs([mp.d_P2_dm1, mp.d_dm1_dm2, mp.d_P2_dm1+mp.d_dm1_dm2]))//dx_dm**2 ) #Minimum number of points across for accurate angular spectrum propagation
    dm.NboxAS = max([dm.Nbox, Nmin]) #Uses a larger array if the max sampling criterion for angular spectrum propagation is violated


    ## Pad the pupil to at least the size of the DM(s) surface(s) to allow all actuators to be located outside the pupil. (Same for both DMs)

    #Find actuator farthest from center:
    dm.r_cent_act = np.sqrt(dm.xy_cent_act[0]**2 + dm.xy_cent_act[1]**2)
    dm.rmax = max(np.abs(dm.r_cent_act))
    NpixPerAct = dm.dm_spacing/dx_dm
    dm.NdmPad = falco.utils.ceil_even( ( dm.NboxAS + 2*(1+ (np.max(np.abs(dm.xy_cent_act))+0.5)*NpixPerAct)) ) # DM surface array padded by the width of the padded influence function to prevent indexing outside the array. The 1/2 term is because the farthest actuator center is still half an actuator away from the nominal array edge. 


    #Compute coordinates (in meters) of the full DM array
    if dm.centering ==  "pixel":
        dm.x_pupPad = np.arange(-dm.NdmPad/2, dm.NdmPad/2)*dx_dm # meters, coords for the full DM arrays. Origin is centered on a pixel
    else:
        dm.x_pupPad = np.arange(-(dm.NdmPad-1)/2, (dm.NdmPad+1)/2)*dx_dm # meters, coords for the full DM arrays. Origin is centered between pixels for an even-sized array

    dm.y_pupPad = dm.x_pupPad



    #DM: (use NboxPad-sized postage stamps)

    if flagGenCube:
        #-Find the locations of the postage stamps arrays in the larger pupilPad array
        dm.xy_cent_act_inPix = dm.xy_cent_act*(dm.dm_spacing/dx_dm) # Convert units to pupil-file pixels
        dm.xy_cent_act_inPix = dm.xy_cent_act_inPix + 0.5 #For the half-pixel offset if pixel centered. 

        dm.xy_cent_act_box = np.round(dm.xy_cent_act_inPix) # Center locations of the postage stamps (in between pixels), in actuator widths
        dm.xy_cent_act_box_inM = dm.xy_cent_act_box*dx_dm # now in meters 
        dm.xy_box_lowerLeft = dm.xy_cent_act_box + (dm.NdmPad-dm.Nbox)/2 + 1 # indices of pixel in lower left of the postage stamp within the whole pupilPad array

        #Starting coordinates (in actuator widths) for updated influence function. This is interpixel centered, so do not translate!
        dm.x_box0 = np.arange(-(dm.Nbox-1)/2, (dm.Nbox+1)/2)*dx_dm
        dm.Xbox0,dm.Ybox0 = np.meshgrid(dm.x_box0, dm.x_box0) #meters, interpixel-centered coordinates for the master influence function

        #Limit the actuators used to those within 1 actuator width of the pupil
        r_cent_act_box_inM = np.sqrt(dm.xy_cent_act_box_inM[0]**2 + dm.xy_cent_act_box_inM[1]**2)
        #Compute and store all the influence functions:
        dm.inf_datacube = np.zeros((dm.Nbox,dm.Nbox,dm.NactTotal)) #initialize array of influence function "postage stamps"

        for iact in range(dm.NactTotal): #dm.Nact^2
            Xbox = dm.Xbox0 - (dm.xy_cent_act_inPix[0,iact]-dm.xy_cent_act_box[0,iact])*dx_dm
            Ybox = dm.Ybox0 - (dm.xy_cent_act_inPix[1,iact]-dm.xy_cent_act_box[1,iact])*dx_dm

            #For consistency, the MATLAB data was generated also with 'cubic' interpolation method in falco_gen_dm_poke_cube (dm.inf_datacube(:,:,iact) = interp2(Xinf0,Yinf0,dm.infMaster,dm.Xbox,dm.Ybox,'cubic',0);)
            #The result is still very slightly off because of non-identical interpolation algorithm
            dm.inf_datacube[:,:,iact] = scipy.interpolate.griddata((Xinf0.reshape(-1),Yinf0.reshape(-1)), dm.infMaster.reshape(-1), (Xbox.reshape(-1),Ybox.reshape(-1)), "cubic", fill_value=0).reshape(Xbox.shape)

    dm.act_ele = np.arange(dm.NactTotal)


class DeformableMirrorParameters:
    class _base_dm:
        def __init__(self, **kwargs):
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
            self.VtoH = _spec_arg("VtoH", kwargs, 1e-9*np.ones((1, self.Nact)))
            self.inf0 = _spec_arg("inf0", kwargs, scipy.io.loadmat(
                _influence_function_file)["inf0"])

    def __init__(self, **kwargs):
        self.dm_weights = _spec_arg("dm_weights", kwargs, [1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.dm_ind = _spec_arg("dm_ind", kwargs, [1, 2])
        self.maxAbsdV = _spec_arg("maxAbsdV", kwargs, 30)
        self.dm1 = _spec_arg("dm1", kwargs, self._base_dm())
        self.dm2 = _spec_arg("dm2", kwargs, self._base_dm())


    def init_ws(self, mp):
        self.flagZYX = False

        self.dm1.centering = mp.centering;
        self.dm2.centering = mp.centering;

        assert(self.dm_ind == [1,2])#Anything else is not implemented yet
        self.dm1.compact = self.dm1
        falco_gen_dm_poke_cube(self.dm1, mp, mp.P2.full.dx, False)
        falco_gen_dm_poke_cube(self.dm1.compact, mp, mp.P2.compact.dx);

        self.dm2.compact = self.dm2
        self.dm2.dx = mp.P2.full.dx
        self.dm2.compact.dx = mp.P2.compact.dx
        
        falco_gen_dm_poke_cube(self.dm2, mp, mp.P2.full.dx, False)
        falco_gen_dm_poke_cube(self.dm2.compact, mp, mp.P2.compact.dx)


        #Initial DM voltages
        self.dm1.V = np.zeros(self.dm1.Nact)
        self.dm2.V = np.zeros(self.dm2.Nact)

        #Peak-to-Valley DM voltages
        self.dm1.Vpv = np.zeros((mp.Nitr,1))
        self.dm2.Vpv = np.zeros((mp.Nitr,1))

        #First delta DM settings are zero (for covariance calculation in Kalman filters or robust controllers)
        self.dm1.dV = np.zeros((self.dm1.Nact,self.dm1.Nact)) # delta voltage on DM1
        self.dm2.dV = np.zeros((self.dm2.Nact,self.dm2.Nact)) # delta voltage on DM2

        #Store the DM commands 
        self.dm1.Vall = np.zeros((self.dm1.Nact,self.dm1.Nact,mp.Nitr+1))
        self.dm2.Vall = np.zeros((self.dm2.Nact,self.dm2.Nact,mp.Nitr+1))

        #Initialize the number of actuators used.
        self.dm1.Nele = len(self.dm1.act_ele)
        self.dm2.Nele = len(self.dm2.act_ele)
        
        
        ## Array Sizes for Angular Spectrum Propagation with FFTs

        #Compact Model: Set nominal DM plane array sizes as a power of 2 for angular spectrum propagation with FFTs
        NdmPad = 2**np.ceil(1 + np.log2(max(self.dm1.compact.NdmPad, self.dm2.compact.NdmPad)))

        while (NdmPad < min(mp.sbp_center_vec)*abs(mp.d_dm1_dm2)/mp.P2.full.dx**2) or (NdmPad < min(mp.sbp_center_vec)*abs(mp.d_P2_dm1)/mp.P2.compact.dx**2): #Double the zero-padding until the angular spectrum sampling requirement is not violated
            NdmPad = 2*NdmPad

        self.compact = collections.namedtuple("_compact", "NdmPad")
        self.compact.NdmPad = NdmPad

        #Full Model: Set nominal DM plane array sizes as a power of 2 for angular spectrum propagation with FFTs
        NdmPad = 2**np.ceil(1 + np.log2(max(self.dm1.NdmPad, self.dm2.NdmPad)))
        while (NdmPad < min(mp.lam_array)*abs(mp.d_dm1_dm2)/mp.P2.full.dx**2) or (NdmPad < min(mp.lam_array)*abs(mp.d_P2_dm1)/mp.P2.full.dx**2): #Double the zero-padding until the angular spectrum sampling requirement is not violated
            NdmPad = 2*NdmPad

        self.full = collections.namedtuple("_full", "NdmPad")
        self.full.NdmPad = NdmPad

