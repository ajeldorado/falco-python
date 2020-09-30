import sys
sys.path.insert(0,"../")
import numpy as np
import matplotlib.pyplot as plt
# from astropy.io import fits

import falco


# Test magnification, rotation, and shear of pupil

DeltaY = 2  # pixels
Nbeam = 1000
centering = 'interpixel'
changes = {}
changes['yShear'] = -DeltaY / Nbeam
changes['magFac'] = 0.7
pupilA = falco.mask.falco_gen_pupil_Roman_CGI_20200513(Nbeam, centering, changes)
pupilBprime = np.roll(np.rot90(pupilA, 2), (-2*DeltaY, 0), axis=(0, 1))

changes['clock_deg'] = 180
pupilB = falco.mask.falco_gen_pupil_Roman_CGI_20200513(Nbeam, centering, changes)

plt.figure(1); plt.imshow(pupilB); plt.gray(); plt.colorbar(); plt.pause(1)
plt.figure(2); plt.imshow(pupilBprime); plt.colorbar(); plt.pause(1)
plt.figure(3); plt.imshow(pupilB - pupilBprime); plt.colorbar(); plt.pause(1)

# hdu = fits.PrimaryHDU(pupilA)
# hdu.writeto('/Users/ajriggs/Downloads/pupilA_python.fits', overwrite=True)

# hdu = fits.PrimaryHDU(pupilB)
# hdu.writeto('/Users/ajriggs/Downloads/pupilB_python.fits', overwrite=True)


# %% Lyot stop mode

Nbeam = 309
centering = 'pixel'

del changes
changes = {}
changes['flagLyot'] = True
changes['ID'] = 0.50
changes['OD'] = 0.80
changes['wStrut'] = 0.036
lyot = falco.mask.falco_gen_pupil_Roman_CGI_20200513(Nbeam, centering, changes);
lyotCropped = lyot[1::,1::]

plt.figure(4); plt.imshow(lyot); plt.gray(); plt.colorbar(); plt.pause(1)
plt.figure(5); plt.imshow(lyotCropped - np.fliplr(lyotCropped)); plt.colorbar(); plt.pause(1)
