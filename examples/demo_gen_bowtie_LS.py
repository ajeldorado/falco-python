import sys
sys.path.insert(0,"../")
import falco
import numpy as np

import matplotlib.pyplot as plt

inputs = {}
inputs["Nbeam"] = 100
inputs["ID"] = 0.38
inputs["OD"] = 0.92
inputs["ang"] = 115

#--Optional Inputs
#inputs['centering'] = 'pixel'
#inputs['xShear'] = 0. #--x-axis shear of mask [pupil diameters]
#inputs['yShear'] = 0.5 #--y-axis shear of mask [pupil diameters]
#inputs['clocking'] = 30  #--Clocking of the mask [degrees]
#inputs['magfac'] = 1.5 #--magnification factor of the pupil diameter

LS = falco.mask.falco_gen_bowtie_LS(inputs)

plt.imshow(LS); plt.colorbar(); plt.pause(0.1)

if("centering" in inputs.keys()):
    if inputs["centering"]=='pixel':
        plt.imshow(LS[1::,1::]-np.fliplr(LS[1::,1::])); plt.colorbar(); plt.pause(0.1) #--Check centering
    elif inputs["centering"]=='interpixel':
        plt.imshow(LS-np.fliplr(LS)); plt.colorbar(); plt.pause(0.1) #--Check centering
