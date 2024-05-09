#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:02:18 2022

@author: ajriggs
"""
import os
import numpy as np
from math import isclose

from falco.thinfilm import calc_complex_occulter

# substrate = 'fs'
# metal = 'nickel'
# dielectric = 'mgf2'
# lam = 550e-9
# d0 = 4*lam
# aoi = 10
# t_Ti_base = 0
# t_Ni_vec = [95e-9]
# t_diel_vec = [600e-9]
# pol = 1
# [tCoef, _] = calc_complex_occulter(substrate, metal, dielectric,
#                                    lam, aoi, t_Ti_base, t_Ni_vec,
#                                    t_diel_vec, d0, pol)

# localpath = '/Users/ajriggs/Repos/falco-python/falco'
# fn_mgf2 = os.path.join(
#     localpath, 'data',
#     'MgF2_data_from_Rodriguez-deMarcos_wvlUM_n_k.txt')
# dataMgF2 = np.loadtxt(fn_mgf2, delimiter=" ", unpack=False,
#                       comments="#")
# lamUM_mgf2_0 = dataMgF2[:, 0]  # nm
# lam_mgf2_0 = lamUM_mgf2_0 * 1e3  # [nm]
# n_mgf2_0 = dataMgF2[:, 1]
# k_mgf2_0 = dataMgF2[:, 2]
# n_diel = np.interp(lam_nm, lam_mgf2_0, n_mgf2_0)
# k_diel = np.interp(lam_nm, lam_mgf2_0, k_mgf2_0)

# print(n_diel)
# print(k_diel)