import numpy as np
import matplotlib.pyplot as plt
import falco

Nbeam = 300

inputs = {}
inputs["Nbeam"] = Nbeam
pupil = falco.mask.falco_gen_pupil_LUVOIR_B(inputs)

plt.imshow(falco.util.pad_crop(pupil, 308))
plt.colorbar()
plt.pause(0.1)

# Check centering and symmetry of the pupil
plt.imshow(pupil[1::, 1::] - np.fliplr(pupil[1::, 1::]))
plt.colorbar()
plt.pause(0.1)
