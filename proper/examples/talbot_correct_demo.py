#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import proper
import numpy as np
import matplotlib.pylab as plt

def talbot_correct_demo():
    diam = 0.1                   # beam diameter in meters
    period = 0.04                # period of cosine pattern (meters)
    wavelength_microns = 0.5
    wavelength_m = wavelength_microns * 1.e-6
    n = 128
    
    nseg = 9
    talbot_length = 2 * period**2 / wavelength_m
    delta_length = talbot_length / (nseg - 1.)
    
    z = 0.
    
    plt.close('all')    
    f = plt.figure(figsize = (8, 18))
    
    for i in range(nseg):
        (wavefront, sampling) = proper.prop_run('talbot_correct', 
             wavelength_microns, n, 
             PASSVALUE = {'diam': diam, 'period': period, 'dist': z})
            
        # Extract central cross-section of array
        wavefront = wavefront[:,n//2]
        
        amp = np.abs(wavefront)
        amp -= np.mean(amp)
        phase = np.arctan2(wavefront.imag, wavefront.real)
        phase -= np.mean(phase)
        
        ax1 = f.add_subplot(nseg,2,2*i+1)
        if i == 0:
            ax1.set_title('Amplitude')
        ax1.set_ylim(-0.0015, 0.0015)
        ax1.plot(amp)
        ax2 = f.add_subplot(nseg,2,2*i+2)
        if i == 0:
            ax2.set_title('Phase')
        ax2.set_ylim(-0.25, 0.25)
        ax2.plot(phase)
        
        z += delta_length
        
    plt.show()
    
    return
    
    
if __name__ == '__main__':
    talbot_correct_demo()
