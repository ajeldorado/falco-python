import numpy as np
import matplotlib.pyplot as plt

import falco


## Pupil Mode

Nbeam = 1045
centering = 'pixel'
changes = {}
changes['clock_deg'] = 15
changes['xShear'] = 0.1
changes['yShear'] = 0.2
changes['magFac'] = 0.5
pupil = falco.mask.falco_gen_pupil_WFIRST_CGI_20191009(Nbeam, centering, changes)

plt.figure(1); plt.imshow(pupil); plt.colorbar(); plt.pause(1)

## Lyot stop mode

Nbeam = 309
centering = 'pixel'

del changes
changes = {}
changes['flagLyot'] = True
changes['ID'] = 0.50
changes['OD'] = 0.80
changes['wStrut'] = 0.04
LS = falco.mask.falco_gen_pupil_WFIRST_CGI_20191009(Nbeam, centering, changes);

plt.figure(2); plt.imshow(LS); plt.colorbar(); plt.pause(1)

croppedLS = LS[1::,1::]
plt.figure(3); plt.imshow(croppedLS - np.fliplr(croppedLS)); plt.colorbar(); plt.pause(1)
