import sys
sys.path.insert(0,"../")
import falco
import proper
import numpy as np

import matplotlib.pyplot as plt

inputs = {}
inputs["Nbeam"] = 1000
inputs["magfacD"] = 1.
inputs["wStrut"] = 0.01

pupil = falco.masks.falco_gen_pupil_LUVOIR_A_final(inputs)

plt.imshow(pupil); plt.colorbar(); plt.pause(0.1)

plt.imshow(pupil[1::,1::]-np.fliplr(pupil[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering

#figure(2); imagesc(pupil); axis xy equal tight; title('Input Pupil','Fontsize',20); colorbar;

#figure(3); imagesc(pupil(2:end,2:end)-fliplr(pupil(2:end,2:end))); axis xy equal tight; title('Symmetry Check: Differencing','Fontsize',20); colorbar


