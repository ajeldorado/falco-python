#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:53:15 2024

@author: gregorywa
"""

import numpy as np
import matplotlib.pyplot as plt

import falco

def gen_test_pattern():
    dimsTest = 100

    x = falco.util.create_axis(dimsTest, 1)
    y = falco.util.create_axis(dimsTest, 1)
    
    dx = 0;
    dy = 0;

    X,Y = np.meshgrid(x,y)

    fxPattern = 1/20
    fyPattern = 1/20

    #pattern = 5 + np.cos(2* np.pi * fxPattern * X) * np.cos(2* np.pi * fyPattern * Y)
    pattern = 5 + np.sinc(2* np.pi * fxPattern * (X-dx)) * np.sinc(2* np.pi * fyPattern * (Y-dy))
    
    return pattern
        
        

def test_fourier_resample(test_input):
    zoom = 1.51
    fprime = falco.diff_dm.fourier_resample(test_input,zoom)
    fprimeprime = falco.diff_dm.fourier_resample(fprime,1/zoom)
    
    abs_tol = 0.005*np.max(test_input)

    plt.figure()
    plt.imshow(test_input)
    plt.colorbar()
    plt.title('Input')
    
    plt.figure()
    plt.imshow(fprime)
    plt.colorbar()
    plt.title('Output1')
    
    plt.figure()
    plt.imshow(fprimeprime)
    plt.colorbar()
    plt.title('Output2')
    
    plt.figure()
    plt.imshow(fprimeprime-test_input)
    plt.colorbar()
    plt.title('Difference')

    plt.show()
    
    indMaxf = np.unravel_index(np.argmax(test_input),test_input.shape)
    indMaxfp = np.unravel_index(np.argmax(fprime),fprime.shape)
    indMaxfpp = np.unravel_index(np.argmax(fprimeprime),fprimeprime.shape)
    
    assert( np.all( indMaxf==np.floor(np.array(test_input.shape)/2) ) )
    assert( np.all( indMaxfp==np.floor(np.array(fprime.shape)/2) ) )
    assert( np.all( indMaxfpp==np.floor(np.array(fprimeprime.shape)/2) ) )
    
    maxAbsDiff = np.max(np.abs(fprimeprime - test_input))
    assert maxAbsDiff < abs_tol


if __name__ == '__main__':
    pattern = gen_test_pattern()
    test_fourier_resample(pattern)