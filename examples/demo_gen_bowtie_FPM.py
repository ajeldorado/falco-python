import sys
sys.path.insert(0,"../")
import falco
import numpy as np

import matplotlib.pyplot as plt


inputs = {} # initialize
inputs["angDegrees"] = 65 # Opening angle on each side of the bowtie
inputs["pixresFPM"] = 6
inputs["rhoInner"] = 2.6
inputs["rhoOuter"] = 9.0
inputs["centering"] = 'pixel'

fpm = falco.mask.falco_gen_bowtie_FPM(inputs)

plt.imshow(fpm); plt.colorbar(); plt.pause(0.1)

if("centering" in inputs.keys()):
    if inputs["centering"]=='pixel':
        plt.imshow(fpm[1::,1::]-np.fliplr(fpm[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering
    elif inputs["centering"]=='interpixel':
        plt.imshow(fpm-np.fliplr(fpm)); plt.colorbar(); plt.pause(0.1) #--Check centering

