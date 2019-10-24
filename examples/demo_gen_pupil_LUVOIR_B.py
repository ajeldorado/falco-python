import sys
sys.path.insert(0,"../")
import falco
import proper
import numpy as np

import matplotlib.pyplot as plt

Nbeam = 300

pupil = falco.masks.falco_gen_pupil_LUVOIR_B(Nbeam)

plt.imshow(pupil); plt.colorbar(); plt.pause(0.1)

plt.imshow(pupil[1::,1::]-np.fliplr(pupil[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering

