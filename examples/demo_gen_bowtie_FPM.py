import cupy as cp
import sys
sys.path.insert(0,"../")
import falco
import numpy as np

import matplotlib.pyplot as plt


icp.ts = {} # initialize
icp.ts["angDegrees"] = 65 # Opening angle on each side of the bowtie
icp.ts["pixresFPM"] = 6
icp.ts["rhoInner"] = 2.6
icp.ts["rhoOuter"] = 9.0
icp.ts["centering"] = 'pixel'

fpm = falco.masks.falco_gen_bowtie_FPM(icp.ts)

plt.imshow(fpm); plt.colorbar(); plt.pause(0.1)

if("centering" in icp.ts.keys()):
    if icp.ts["centering"]=='pixel':
        plt.imshow(fpm[1::,1::]-cp.fliplr(fpm[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering
    elif icp.ts["centering"]=='interpixel':
        plt.imshow(fpm-cp.fliplr(fpm)); plt.colorbar(); plt.pause(0.1) #--Check centering

