"""FALCO plotting."""
import numpy as np
import matplotlib.pyplot as plt

from . import check


def plot_trial_output(out):
    """
    Plot a FALCO trial's data using a FALCO object as the input.

    Parameters
    ----------
    out : FALCO object
        Object containing performance data from the FALCO trial.

    Returns
    -------
    None
    """

    plt.figure()
    plt.plot(range(out.Nitr+1), out.thput)
    plt.xlabel('Iteration')
    plt.ylabel('Throughput')
    
    plt.figure()
    plt.semilogy(range(out.Nitr+1), out.InormHist)
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Intensity')
    
    plt.figure()
    plt.plot(range(out.Nitr), out.log10regHist)
    plt.xlabel('Iteration')
    plt.ylabel('log10 Regularization')
    
    plt.figure()
    plt.plot(range(out.Nitr), 1e9*out.dm1.Srms, '-r',
             range(out.Nitr), 1e9*out.dm2.Srms, '-b')
    plt.xlabel('Iteration')
    plt.ylabel('RMS DM Surface (nm)')
    
    plt.figure()
    plt.imshow(out.DM1V)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Final DM1 Voltages')
    
    plt.figure()
    plt.imshow(out.DM2V)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Final DM2 Voltages')
    
    iterCount = 1
    plt.figure()
    plt.imshow(out.dm1.Vall[:, :, iterCount])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(('DM1 Voltages at Iteration %d' % (iterCount)))

    return None


def plot_trial_output_from_pickle(fnPickle):
    """
    Plot a FALCO trial's pickled output data.

    Parameters
    ----------
    fnPickle : str
        Filename of pickle containing performance data from the FALCO trial.

    Returns
    -------
    None
    """
    
    with np.load(fnPickle) as data:
        out = data['out']
    
    plt.figure()
    plt.plot(range(out.Nitr+1), out.thput)
    plt.xlabel('Iteration')
    plt.ylabel('Throughput')
    
    plt.figure()
    plt.semilogy(range(out.Nitr+1), out.InormHist)
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Intensity')
    
    plt.figure()
    plt.plot(range(out.Nitr), out.log10regHist)
    plt.xlabel('Iteration')
    plt.ylabel('log10 Regularization')
    
    plt.figure()
    plt.plot(range(out.Nitr), 1e9*out.dm1.Srms, '-r',
             range(out.Nitr), 1e9*out.dm2.Srms, '-b')
    plt.xlabel('Iteration')
    plt.ylabel('RMS DM Surface (nm)')
    
    plt.figure()
    plt.imshow(out.DM1V)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Final DM1 Voltages')
    
    plt.figure()
    plt.imshow(out.DM2V)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Final DM2 Voltages')
    
    iterCount = 1
    plt.figure()
    plt.imshow(out.dm1.Vall[:, :, iterCount])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(('DM1 Voltages at Iteration %d' % (iterCount)))