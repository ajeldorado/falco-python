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


def coronagraph_demo():
    
    n = 512             # grid size
    lamda = 0.55        # wavelength (microns)

    (no_errors, no_errors_sampl) = proper.prop_run("run_coronagraph_dm", lamda, n, PASSVALUE = {'use_errors': False, 'use_dm': False, 'occulter_type': '8TH_ORDER'}, VERBOSE = False)
    
    (with_errors, with_errors_sampl) = proper.prop_run("run_coronagraph_dm", lamda, n, PASSVALUE = {'use_errors': True, 'use_dm': False, 'occulter_type': '8TH_ORDER'}, VERBOSE = False)

    (with_dm, with_dm_sampl) = proper.prop_run("run_coronagraph_dm", lamda, n, PASSVALUE = {'use_errors': True, 'use_dm': True, 'occulter_type': '8TH_ORDER'}, VERBOSE = False)
    
    nd = 256
    psfs = np.zeros([3,nd,nd], dtype = np.float64)
    psfs[0,:,:] = no_errors[int(n/2-nd/2):int(n/2+nd/2),int(n/2-nd/2):int(n/2+nd/2)]
    psfs[1,:,:] = with_errors[int(n/2-nd/2):int(n/2+nd/2),int(n/2-nd/2):int(n/2+nd/2)]
    psfs[2,:,:] = with_dm[int(n/2-nd/2):int(n/2+nd/2),int(n/2-nd/2):int(n/2+nd/2)]
    
    plt.figure(figsize = (14,7))
    plt.suptitle("PSFs", fontsize = 18, fontweight = 'bold')
    
    plt.subplot(1,3,1)
    plt.title('No errors')
    plt.imshow(psfs[0,:,:]**0.25, origin = "lower", cmap = plt.cm.gray)
    plt.subplot(1,3,2)
    plt.title('With errors')
    plt.imshow(psfs[1,:,:]**0.25, origin = "lower", cmap = plt.cm.gray)
    plt.subplot(1,3,3)
    plt.title('DM corrected')
    plt.imshow(psfs[2,:,:]**0.25, origin = "lower", cmap = plt.cm.gray)
    plt.show()
    
    print("Maximum speckle flux / stellar flux :")
    print("  No wavefront errors = {0:0.3E}".format(np.max(no_errors), np.min(no_errors)))
    print("  With wavefront errors = {0:0.3E}".format(np.max(with_errors)))
    print("  With DM correction = {0:0.3E}".format(np.max(with_dm)))
    
    
if __name__ == '__main__':
    coronagraph_demo()
