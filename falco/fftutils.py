try:
    from mkl_fft import (fft2, ifft2, fftshift, ifftshift)
    print("Loaded FFT utilities from 'MKL_FFT'")
except:
    print("Loaded FFT utilities from 'numpy.fft'")
    from numpy.fft import (fft2, ifft2, fftshift, ifftshift)


from numpy.fft import fftfreq
