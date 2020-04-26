import sys
sys.path.insert(0,"../")
import falco
#import proper
#import numpy as np

import matplotlib.pyplot as plt

Nbeam = 300

pupil = falco.masks.falco_gen_pupil_LUVOIR_B(Nbeam)

plt.imshow(falco.utils.pad_crop(pupil,308)); plt.colorbar(); plt.pause(0.1)

#plt.imshow(pupil[1::,1::]-np.fliplr(pupil[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering

inputs = {}
inputs["Nbeam"] = Nbeam
pupil2 = falco.masks.falco_gen_pupil_LUVOIR_B_PROPER(inputs)

plt.imshow(pupil2); plt.colorbar(); plt.pause(0.1)

plt.imshow(pupil2-falco.utils.pad_crop(pupil,308)); plt.colorbar(); plt.pause(0.1)


