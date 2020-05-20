import cupy as cp
import sys
sys.path.insert(0,"../")
import falco
import numpy as np

import matplotlib.pyplot as plt

Nbeam = 250

icp.ts = {} # initialize
#icp.ts["flagPROPER"] = True
icp.ts["Nbeam"] = Nbeam #- Number of samples across the beam 
icp.ts["Npad"] = 256
icp.ts["ID"] = 0.10 #- Outer diameter (fraction of Nbeam)
icp.ts["OD"] = 0.80#- Inner diameter (fraction of Nbeam)
#icp.ts["Nstrut"] = 3 #- Number of struts
#icp.ts["angStrut"] = cp.array([30., 150., 270.])#- Array of struct angles (deg)
#icp.ts["wStrut"] = 0.01 #- Strut widths (fraction of Nbeam)
#icp.ts["stretch"] = 1.#- Create an elliptical aperture by changing Nbeam along the horizontal direction by a factor of stretch (PROPER version isn't implemented as of March 2019).
#icp.ts["centering"] = 'pixel'

pupil = falco.masks.falco_gen_pupil_Simple(icp.ts)

plt.imshow(pupil); plt.colorbar(); plt.pause(0.1)

if("centering" in icp.ts.keys()):
    if icp.ts["centering"]=='pixel':
        plt.imshow(pupil[1::,1::]-cp.fliplr(pupil[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering
    elif icp.ts["centering"]=='interpixel':
        plt.imshow(pupil-cp.fliplr(pupil)); plt.colorbar(); plt.pause(0.1) #--Check centering

