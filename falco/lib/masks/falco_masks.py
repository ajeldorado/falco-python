import numpy as np
from skimage.io import imread

def falco_gen_pupil_WFIRST_20180103(Nbeam,centering,**kwargs):
    """
    Script to generate a WFIRST CGI input pupil, ID # 20180103. (ID # is the
    date it was received.)
    -Has non-circular, offset secondary mirror.
    -Has 3.22# OD) strut scraper width.
    """
    # Set default values of input parameters
    flagRot180deg = False;
    #--Look for Optional Keywords
    icav = 0; # index in cell array varargin
    for key, value in kwargs.items():
        if value.lower() == 'rot180':
            flagRot180deg = True; # For even arrays, beam center is in between pixels.
            pass
        else:
            #print('Error:  falco_gen_pupil_WFIRST_20180103: Unknown keyword: '%(value))
            raise ValueError('falco_gen_pupil_WFIRST_20180103: Unknown keyword: '%(value))

    #--Load the WFIRST Pupil
#    pupil0 = imread('pupil_WFIRST_CGI_20180103.png'); 
#    pupil0 = np.rot90(pupil0,2);
    
#    pupil1 = np.sum(pupil0,3);
#    pupil1 = pupil1/np.max(pupil1(:));
#    if(flagRot180deg)
#        pupil1 = np.rot90(pupil1,2);
#    end
#    
#    ##--Resize
#    Npup = len(pupil1);
#    xs0 = ( -(Npup-1)/2:(Npup-1)/2 )/Npup; #--original coordinates, normalized to the pupil diameter. True for the 20180103 design, which is interpixel centered.
#    Xs0 = meshgrid(xs0);
#    
#    
#    switch centering
#        case{'interpixel','even'}
#            xs1 = ( -(Nbeam-1)/2:(Nbeam-1)/2 )/Nbeam;
#            Xs1 = meshgrid(xs1);
#            mask = interp2(Xs0,Xs0.',pupil1,Xs1,Xs1.','spline',0); #--interp2 does not get the gray edges as well, but imresize requires the Image Processing Toolbox
#        case{'pixel','odd'}
#            xs1 = ( -(Nbeam)/2:(Nbeam)/2 )/Nbeam;
#            Xs1 = meshgrid(xs1);
#            temp = interp2(Xs0,Xs0.',pupil1,Xs1,Xs1.','linear',0); #--interp2 does not get the gray edges as well, but imresize requires the Image Processing Toolbox
#            mask = zeros(Nbeam+2,Nbeam+2); #--Initialize
#            mask(2:end,2:end) = temp; #--Fill in values
#    end
#    
#        pass
    return np.zeros((1,1))
