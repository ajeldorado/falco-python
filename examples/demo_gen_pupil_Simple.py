"""Check visually the output of falco_gen_pupil_Simple."""
import sys
sys.path.insert(0,"../")
import falco
import numpy as np

import matplotlib.pyplot as plt

Nbeam = 250

inputs = {}  # initialize
inputs["Nbeam"] = 500  # Number of samples across the beam 
inputs["Npad"] = 700  # Number of pixels across square output array
inputs["OD"] = 1.00  # Inner diameter (fraction of Nbeam)

# OPTIONAL INPUTS
inputs["ID"] = 0.20  # Outer diameter (fraction of Nbeam)
inputs["angStrut"] = np.array([0, 90, 180, 270])  # Array of strut angles [deg]
inputs["wStrut"] = 0.01  # Strut widths (fraction of Nbeam)
inputs["stretch"] = 1.  # Create an elliptical aperture by changing Nbeam along the horizontal direction by a factor of stretch
inputs["centering"] = 'pixel'
inputs["clocking"] = 15  # CCW rotation. Doesn't work with flag HG. [degrees]
inputs["xShear"] = 0.1  # [pupil diameters]
inputs["yShear"] = -0.15  # [pupil diameters]
# inputs["flagHG"] = True  # Cannot do lateral shear or clocking

pupil = falco.mask.falco_gen_pupil_Simple(inputs)

plt.figure(); plt.imshow(pupil); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)

# if("centering" in inputs.keys()):
#     if inputs["centering"]=='pixel':
#         plt.imshow(pupil[1::,1::]-np.fliplr(pupil[1::,1::])); plt.colorbar(); plt.pause(0.1) #-Check centering
#     elif inputs["centering"]=='interpixel':
#         plt.imshow(pupil-np.fliplr(pupil)); plt.colorbar(); plt.pause(0.1) #-Check centering


# %% Simplest pupil--no optional inputs
inputs = {}  # initialize
inputs["Nbeam"] = 500  # Number of samples across the beam
inputs["Npad"] = 700  # Number of pixels across square output array
inputs["OD"] = 0.80  # Inner diameter (fraction of Nbeam)

pupil = falco.mask.falco_gen_pupil_Simple(inputs)

plt.figure(); plt.imshow(pupil); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
