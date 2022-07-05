import numpy as np
from math import isclose

from falco.thinfilm import calc_complex_trans_matrix


def test_transmission_pmgi():
    substrate = 'fs'
    metal = 'nickel'
    dielectric = 'pmgi'
    lam = 600e-9
    d0 = 4*lam
    aoi = 0
    t_Ti_base = 0
    t_Ni_vec = [0]
    t_diel_vec = [100e-9]
    pol = 2
    [tCoef, _] = calc_complex_trans_matrix(substrate, metal, dielectric,
                                           lam, aoi, t_Ti_base, t_Ni_vec,
                                           t_diel_vec, d0, pol)
    T_FALCO = np.abs(tCoef)**2  # Value from FALCO: 0.94313153
    T_Macleod = 0.9431006949  # Value from Essential Macleod
    assert isclose(T_FALCO, T_Macleod, rel_tol=0.0001)


def test_transmission_nickel():
    substrate = 'fs'
    metal = 'nickel'
    dielectric = 'pmgi'
    lam = 400e-9
    d0 = 4*lam
    aoi = 10
    t_Ti_base = 0
    t_Ni_vec = [95e-9]
    t_diel_vec = [0]
    pol = 0
    [tCoef, _] = calc_complex_trans_matrix(substrate, metal, dielectric,
                                           lam, aoi, t_Ti_base, t_Ni_vec,
                                           t_diel_vec, d0, pol)
    T_FALCO = np.abs(tCoef)**2  # Value from FALCO: 0.00087847
    T_Macleod = 0.00087848574  # Value from Essential Macleod
    assert isclose(T_FALCO, T_Macleod, rel_tol=0.0001)


def test_transmission_pmgi_on_nickel_a():
    substrate = 'fs'
    metal = 'nickel'
    dielectric = 'pmgi'
    lam = 450e-9
    d0 = 4*lam
    aoi = 10
    t_Ti_base = 0
    t_Ni_vec = [95e-9]
    t_diel_vec = [30e-9]
    pol = 1
    [tCoef, _] = calc_complex_trans_matrix(substrate, metal, dielectric,
                                           lam, aoi, t_Ti_base, t_Ni_vec,
                                           t_diel_vec, d0, pol)
    T_FALCO = np.abs(tCoef)**2  # Value from FALCO: 0.00118379
    T_Macleod = 0.00118382732  # Value from Essential Macleod
    assert isclose(T_FALCO, T_Macleod, rel_tol=0.0001)


def test_transmission_pmgi_on_nickel_b():
    substrate = 'fs'
    metal = 'nickel'
    dielectric = 'pmgi'
    lam = 550e-9
    d0 = 4*lam
    aoi = 10
    t_Ti_base = 0
    t_Ni_vec = [95e-9]
    t_diel_vec = [600e-9]
    pol = 1
    [tCoef, _] = calc_complex_trans_matrix(substrate, metal, dielectric,
                                           lam, aoi, t_Ti_base, t_Ni_vec,
                                           t_diel_vec, d0, pol)
    T_FALCO = np.abs(tCoef)**2  # Value from FALCO: 0.001216750339
    T_Macleod = 0.00121675706  # Value from Essential Macleod
    assert isclose(T_FALCO, T_Macleod, rel_tol=0.0001)


def test_no_errors_mgf2_on_nickel():
    substrate = 'fs'
    metal = 'nickel'
    dielectric = 'mgf2'
    lam = 550e-9
    d0 = 4*lam
    aoi = 10
    t_Ti_base = 0
    t_Ni_vec = [95e-9]
    t_diel_vec = [600e-9]
    pol = 1
    [tCoef, _] = calc_complex_trans_matrix(substrate, metal, dielectric,
                                           lam, aoi, t_Ti_base, t_Ni_vec,
                                           t_diel_vec, d0, pol)


if __name__ == '__main__':
    test_transmission_pmgi()
    test_transmission_nickel()
    test_transmission_pmgi_on_nickel_a()
    test_transmission_pmgi_on_nickel_b()
    test_no_errors_mgf2_on_nickel()
