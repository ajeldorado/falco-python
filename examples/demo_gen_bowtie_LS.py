import cupy as cp
import sys
sys.path.insert(0,"../")
import falco
import numpy as np

import matplotlib.pyplot as plt

icp.ts = {}
icp.ts["Nbeam"] = 100
icp.ts["ID"] = 0.38
icp.ts["OD"] = 0.92
icp.ts["ang"] = 115

#--Optional Icp.ts
#icp.ts['centering'] = 'pixel'
#icp.ts['xShear'] = 0. #--x-axis shear of mask [pupil diameters]
#icp.ts['yShear'] = 0.5 #--y-axis shear of mask [pupil diameters]
#icp.ts['clocking'] = 30  #--Clocking of the mask [degrees]
#icp.ts['magfac'] = 1.5 #--magnification factor of the pupil diameter

LS = falco.masks.falco_gen_bowtie_LS(icp.ts)

plt.imshow(LS); plt.colorbar(); plt.pause(0.1)

if("centering" in icp.ts.keys()):
    if icp.ts["centering"]=='pixel':
        plt.imshow(LS[1::,1::]-cp.fliplr(LS[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering
    elif icp.ts["centering"]=='interpixel':
        plt.imshow(LS-cp.fliplr(LS)); plt.colorbar(); plt.pause(0.1) #--Check centering
