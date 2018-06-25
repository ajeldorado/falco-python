import falco.masks as masks
import numpy as np


def test_annular_fpm():
    """ Quick and dirty spot check of annular_fpm
    This test is very partial at best.
    """

    # test some semi-random cases - is the array size as expected? 
    assert masks.annular_fpm(3, 2, np.inf).shape == (3*2*2, 3*2*2)
    assert masks.annular_fpm(3, 5, np.inf).shape == (3*5*2, 3*5*2)
    assert masks.annular_fpm(3, 5, 10).shape == (3*10*2, 3*10*2)
    assert masks.annular_fpm(3, 5, 11).shape == (3*11*2, 3*11*2)

    # test some pixel values are as expected. 
    mask = masks.annular_fpm(3, 2, 10)
    assert mask[0,0]==0 # corner is black
    assert mask[5*10, 5*10]==1 # in between is white
    assert mask[3*10, 3*10]==0 # center is black

def test_falco_gen_DM_stop():
    #using a MATLAB generated data stored in _LC_single_trial_mp_data to generate a mask and compare it to MATLAB generated mask
    from falco.tests._LC_single_trial_mp_data import mp

    mask = masks.falco_gen_DM_stop(mp.P2.full.dx,mp.dm1.Dstop,mp.centering)
    assert(np.allclose(mp.dm1.full.mask, mask))
    return mp, mask

def test_falco_gen_pupil_WFIRST_20180103():
    #using a MATLAB generated data stored in _LC_single_trial_mp_data to generate a mask and compare it to MATLAB generated mask
    from falco.tests._LC_single_trial_mp_data import mp

    mask = masks.falco_gen_pupil_WFIRST_20180103(mp.P1.full.Nbeam, mp.centering)
    assert(np.allclose(mp.P1.full.mask, mask))
    return mp, mask

def test_falco_gen_pupil_WFIRSTcycle6_LS():
    #using a MATLAB generated data stored in _LC_single_trial_mp_data to generate a mask and compare it to MATLAB generated mask
    from falco.tests._LC_single_trial_mp_data import mp

    mask = masks.falco_gen_pupil_WFIRSTcycle6_LS(mp.P4.full.Nbeam, mp.P4.D, mp.P4.IDnorm, mp.P4.ODnorm, mp.LS_strut_width, mp.centering, True)
    assert(np.allclose(mp.P4.full.mask, mask))
    return mp, mask
