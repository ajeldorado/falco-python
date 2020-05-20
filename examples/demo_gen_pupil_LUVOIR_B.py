import cupy as cp
import sys
sys.path.insert(0,"../")
import falco
#import proper
#import numpy as np

import matplotlib.pyplot as plt

Nbeam = 300

pupil = falco.masks.falco_gen_pupil_LUVOIR_B(Nbeam)

plt.imshow(falco.utils.padOrCropEven(pupil,308)); plt.colorbar(); plt.pause(0.1)

#plt.imshow(pupil[1::,1::]-cp.fliplr(pupil[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering

icp.ts = {}
icp.ts["Nbeam"] = Nbeam
pupil2 = falco.masks.falco_gen_pupil_LUVOIR_B_PROPER(icp.ts)

plt.imshow(pupil2); plt.colorbar(); plt.pause(0.1)

plt.imshow(pupil2-falco.utils.padOrCropEven(pupil,308)); plt.colorbar(); plt.pause(0.1)


