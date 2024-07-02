import sys
sys.path.insert(0,"../")
import falco
import numpy as np

import matplotlib.pyplot as plt


inputs = {}
inputs["FPMampFac"] = 0.
inputs["pixresFPM"] = 3
inputs["rhoInner"] = 6.5
inputs["centering"] = 'pixel'

# %% With Outer Ring

inputs["rhoOuter"] = 20.0
fpm = falco.mask.falco_gen_annular_FPM(inputs)

plt.imshow(fpm); plt.colorbar(); plt.pause(0.1)
if("centering" in inputs.keys()): # Check symmetry
    if inputs["centering"]=='pixel':
        plt.imshow(fpm[1::,1::]-np.fliplr(fpm[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering
    elif inputs["centering"]=='interpixel':
        plt.imshow(fpm-np.fliplr(fpm)); plt.colorbar(); plt.pause(0.1) #--Check centering

# %% Without Outer Ring
        
inputs["rhoOuter"] = np.inf
fpm = falco.mask.falco_gen_annular_FPM(inputs)

plt.imshow(fpm); plt.colorbar(); plt.pause(0.1)
if("centering" in inputs.keys()): # Check symmetry
    if inputs["centering"]=='pixel':
        plt.imshow(fpm[1::,1::]-np.fliplr(fpm[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering
    elif inputs["centering"]=='interpixel':
        plt.imshow(fpm-np.fliplr(fpm)); plt.colorbar(); plt.pause(0.1) #--Check centering
        
