import cupy as cp
import sys
sys.path.insert(0,"../")
import falco
import numpy as np

import matplotlib.pyplot as plt


icp.ts = {}
icp.ts["FPMampFac"] = 0.
icp.ts["pixresFPM"] = 3
icp.ts["rhoInner"] = 6.5
icp.ts["centering"] = 'pixel'

# %% With Outer Ring

icp.ts["rhoOuter"] = 20.0
fpm = falco.masks.falco_gen_annular_FPM(icp.ts)

plt.imshow(fpm); plt.colorbar(); plt.pause(0.1)
if("centering" in icp.ts.keys()): # Check symmetry
    if icp.ts["centering"]=='pixel':
        plt.imshow(fpm[1::,1::]-cp.fliplr(fpm[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering
    elif icp.ts["centering"]=='interpixel':
        plt.imshow(fpm-cp.fliplr(fpm)); plt.colorbar(); plt.pause(0.1) #--Check centering

# %% Without Outer Ring
        
icp.ts["rhoOuter"] = cp.Infinity
fpm = falco.masks.falco_gen_annular_FPM(icp.ts)

plt.imshow(fpm); plt.colorbar(); plt.pause(0.1)
if("centering" in icp.ts.keys()): # Check symmetry
    if icp.ts["centering"]=='pixel':
        plt.imshow(fpm[1::,1::]-cp.fliplr(fpm[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering
    elif icp.ts["centering"]=='interpixel':
        plt.imshow(fpm-cp.fliplr(fpm)); plt.colorbar(); plt.pause(0.1) #--Check centering
        