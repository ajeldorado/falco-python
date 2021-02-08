Python PROPER
=============

Implementation of John Krist's PROPER optical propagation library for Python
version 3.x. PROPER is a library of optical propagation procedures and functions
for the IDL (Interactive Data Language) environment. PROPER is intended for
exploring diffraction effects in optical systems. It is a set of wavefront
propagation tools – it is not a ray tracing system and thus is not suitable for
detailed design work.

+ Authors: Navtej Singh, Nikta Amiri, Luis Marchen
+ Contact: nsaini@jpl.nasa.gov
+ Organization: NASA Jet Propulsion Laboratory
                California Institute of Technology

+ Following requirements should be met to run pyPROPER3 -
  + Numpy >= 1.8
  + Scipy >= 0.14
  + astropy >= 1.3

  To use FFTW, you will also need pyfftw >= 0.1
+ To install PROPER, execute the following command in downloaded PyPROPER
  directory -  

    python setup.py install

+ To use PROPER without installing, issue following commands
  in python/ipython shell

     import
     sys.path.insert(0, /path/to/PROPER)

+ To run a prescription in interactive mode -
  1. Open python or ipython shell and change directory to where the prescription
     is lying.
  2. Import proper package -   
       import proper
  3. Execute the prescription -  
       (psf, sampling) = proper.prop_run('prescription_name', wavelength, grid_size)  

     where the first parameter is name of prescription (without file extension),
     second parameter is wavelength in micrometer and third argument is grid
     dimension. You can also pass key-value pairs as optional parameters. To run
     multiple cases at once in parallel, use prop_run_multi instead.

     Please refer to PROPER user manual for more details.
  4. One can display the generated point spread function (psf) using
     matplotlib package -  
       import numpy as np  
       import matplotlib.pylab as plt  

       plt.imshow(np.log10(psf), origin = 'lower')  
       plt.show()  
  5. The PSF can be saved as FITS image -  
       proper.prop_fits_write("example.fits", psf)  

     where the first parameter is FITS image file name and second parameter is
     2D numpy array. prop_fits_write also accepts optional key-value parameters
     (check doc-string in prop_fits_write for more details). This function will
     overwrite an existing FITS image with the same name.
