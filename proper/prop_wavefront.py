#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri



import numpy as np
import proper

class WaveFront(object):
    """Wavefront class to define wavefront of an optical system.
    
    Wavefront array structure is created by this routine.
    """    
    # Class variables
    nlist = 1500
     
           
    def __init__(self, beam_diam, ndiam, wavelength, ngrid, w0, z_ray):
        """WaveFront object constructor
        
        Parameters
        ----------
        beam_diam : float
            Initial diameter if beam in meters
            
        wavelength : float
            Wavelength in meters
            
        ngrid : float
            Wavefront gridsize in pixels (n by n)
            
        beam_diam_fraction : float
            Fraction of the grid width corresponding to the beam diameter. 
            If not specified, it is assumed to be 0.5.
            
        Returns
        -------
        wfo : obj
            Wavefront class object
        """
        # wavefront structure
        self._wfarr = np.ones([ngrid, ngrid], dtype = np.complex128)     # wavefront array
        self._lamda = float(wavelength)                                  # wavelength in meters
        self._dx = beam_diam/ndiam                                       # grid sampling (meters/pixel)
        self._beam_type_old = "INSIDE_"                                  # beam location (inside or outside beam waist)
        self._reference_surface = "PLANAR"                               # reference surface type
        self._R_beam = 0.                                                # beam radius of curvature
        self._R_beam_inf = 1                                             # beam starts out with infinite curvature radius
        self._z = 0.                                                     # current location along propagation direction (meters)
        self._z_w0 = 0.                                                  # beam waist location (meters)
        self._w0 = w0                                                    # beam waist radius (meters)
        self._z_Rayleigh = z_ray                                         # Rayleigh distance for current beam
        self._propagator_type = "INSIDE__TO_INSIDE_"                     # inside_to_outside (or vice-versa) or inside_to_inside
        self._current_fratio = 1.e9                                      # current F-ratio 
        self._diam = beam_diam                                           # initial beam diameter in meters    
        self._ngrid = ngrid
        
        return


    @property
    def wfarr(self):
        """Method returns current complex-values wavefront array.
        
        Parameters
        ----------
            None
            
        Returns
        -------
        wfarr : numpy ndarray
            A 2D, complex valued wavefront array centered in the array
        """
        return self._wfarr

        
    @wfarr.setter
    def wfarr(self, value):
        self._wfarr = value
                                
        
    @property
    def lamda(self):
        """Method returns wavelength in meters.
        
        Parameters
        ----------
            None
            
        Returns
        -------
        lamda : float
            Wavelength in meters
        """
        return self._lamda

                
    @lamda.setter
    def lamda(self, value):
        self._lamda = float(value)
        
                                                                                        
    @property
    def dx(self):
        """Method returns grid sampling
        
        Parameters
        ----------
            None
            
        Returns
        -------
        dx : float
          Grid sampling  
        """
        return self._dx       

                
    @dx.setter
    def dx(self, value):
        self._dx = float(value)
                                                        

    @property
    def beam_type_old(self):
        """Method returns beam location
        
        Parameters
        ----------
          None
            
        Returns
        -------
        beam_type_old : str
          Beam location  
        """
        return self._beam_type_old                            

                                                        
    @beam_type_old.setter
    def beam_type_old(self, value):
        self._beam_type_old = value
                                                                                                                                                                                

    @property
    def reference_surface(self):
        """Method returns reference surface type
        
        Parameters
        ----------
            None
            
        Returns
        -------
        beam_type_old : str
          Reference surface type  
        """
        return self._reference_surface


    @reference_surface.setter
    def reference_surface(self, value):
        self._reference_surface = value
        
                                                        
    @property
    def R_beam(self):
        """Method returns beam radius of curvature
        
        Parameters
        ----------
            None
            
        Returns
        -------
        R_beam : float
          Beam radius of curvature         
        """
        return self._R_beam


    @R_beam.setter
    def R_beam(self, value):
        self._R_beam = float(value)
        
        
    @property
    def R_beam_inf(self):
        """Method returns beam infinite radius of curvature
        
        Parameters
        ----------
            None
            
        Returns
        -------
        R_beam : float
          Beam infinite radius of curvature         
        """
        return self._R_beam_inf
        

    @R_beam_inf.setter
    def R_beam_inf(self, value):
        self._R_beam_inf = float(value)
        
        
    @property
    def z(self):
        """Method returns current location along propagation direction
        
        Parameters
        ----------
            None
            
        Returns
        -------
        z : float
          Current location along propagation direction         
        """
        return self._z

        
    @z.setter
    def z(self, value):
        self._z = float(value)
                                

    @property
    def z_w0(self):
        """Method returns beam waist location
        
        Parameters
        ----------
            None
            
        Returns
        -------
        z_w0 : float
          Beam waist location (meters)         
        """
        return self._z_w0


    @z_w0.setter
    def z_w0(self, value):
        self._z_w0 = float(value)
        
        
    @property
    def w0(self):
        """Method returns beam waist radius (meters)
        
        Parameters
        ----------
            None
            
        Returns
        -------
        w0 : float
          Beam waist radius (meters)         
        """
        return self._w0

        
    @w0.setter
    def w0(self, value):
        self._w0 = float(value)
                                

    @property
    def z_Rayleigh(self):
        """Method returns Rayleigh distance from current beam
        
        Parameters
        ----------
            None
            
        Returns
        -------
        z_rayleigh : float
          Rayleigh distance from current beam         
        """
        return self._z_Rayleigh
        

    @z_Rayleigh.setter
    def z_Rayleigh(self, value):
        self._z_Rayleigh = float(value)
        
        
    @property
    def propagator_type(self):
        """Method returns propagator type
        
        Parameters
        ----------
            None
            
        Returns
        -------
        propagator_type : str
          Propagator type (inside_to_outside (or vice-versa) or inside_to_inside         
        """
        return self._propagator_type


    @propagator_type.setter
    def propagator_type(self, value):
        self._propagator_type = value
        

    @property
    def current_fratio(self):
        """Method returns current f-ratio
        
        Parameters
        ----------
            None
            
        Returns
        -------
        current_fratio : float
          Current f-ratio         
        """
        return self._current_fratio

                
    @current_fratio.setter
    def current_fratio(self, value):
        self._current_fratio = float(value)


    @property    
    def diam(self):
        """Method returns beam diameter 
        
        Parameter
        ---------
            None
            
        Returns
        -------
        ngrid : float
            Beam diameter in meters
        """
        return self._diam
        

    @diam.setter    
    def diam(self, value):
        self._diam = float(value)
                
                        
    @property    
    def ngrid(self):
        """Method returns grid size 
        
        Parameter
        ---------
            None
            
        Returns
        -------
        ngrid : float
            Grid size in pixels
        """
        return self._ngrid
