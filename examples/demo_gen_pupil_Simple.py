import sys
sys.path.insert(0,"../")
import falco
import numpy as np

import matplotlib.pyplot as plt

Nbeam = 250

inputs = {} # initialize
#inputs["flagPROPER"] = True
inputs["Nbeam"] = Nbeam #- Number of samples across the beam 
inputs["Npad"] = 256
inputs["ID"] = 0.10 #- Outer diameter (fraction of Nbeam)
inputs["OD"] = 0.80#- Inner diameter (fraction of Nbeam)
#inputs["Nstrut"] = 3 #- Number of struts
#inputs["angStrut"] = np.array([30., 150., 270.])#- Array of struct angles (deg)
#inputs["wStrut"] = 0.01 #- Strut widths (fraction of Nbeam)
#inputs["stretch"] = 1.#- Create an elliptical aperture by changing Nbeam along the horizontal direction by a factor of stretch (PROPER version isn't implemented as of March 2019).
#inputs["centering"] = 'pixel'

pupil = falco.mask.falco_gen_pupil_Simple(inputs)

plt.imshow(pupil); plt.colorbar(); plt.pause(0.1)

if("centering" in inputs.keys()):
    if inputs["centering"]=='pixel':
        plt.imshow(pupil[1::,1::]-np.fliplr(pupil[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering
    elif inputs["centering"]=='interpixel':
        plt.imshow(pupil-np.fliplr(pupil)); plt.colorbar(); plt.pause(0.1) #--Check centering

