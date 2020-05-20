import cupy as cp
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:07:03 2020

@author: ajriggs
"""

#--Required Icp.ts
#Nbeam = icp.ts['Nbeam']  # max aperture radius in samples
#Narray = icp.ts['Narray'] # Number of samples across in square output array
#radiusX = icp.ts['radiusX'] # x-radius of ellipse [pupil diameters]
#radiusY = icp.ts['radiusY'] # y-radius of ellipse [pupil diameters]
#clockingRadians = cp.pi/180.*icp.ts['clockingDegrees']
#
##--Optional icp.ts
#if not 'centering' in icp.ts.keys(): icp.ts['centering'] = 'pixel'
#if not 'xShear' in icp.ts.keys(): icp.ts['xShear'] = 0.
#if not 'yShear' in icp.ts.keys(): icp.ts['yShear'] = 0.
#if not 'magFac' in icp.ts.keys(): icp.ts['magFac'] = 1.
#centering = icp.ts['centering']
#xShear = icp.ts['xShear']
#yShear = icp.ts['yShear']
#magFac = icp.ts['magFac']

import numpy as np
import matplotlib.pyplot as plt

Nbeam = 100
Narray = 102
radiusX = 0.5
radiusY = 0.3
clockingRadians = 0#cp.pi/180.*20.0

centering = 'pixel'
xShear = 0 #0.3
yShear = 0#0.3
magFac = 1.0


if centering == 'pixel':
    x = cp.linspace(-Narray/2., Narray/2. - 1, Narray)/float(Nbeam)
elif centering == 'interpixel':
    x = cp.linspace(-(Narray-1)/2., (Narray-1)/2., Narray)/float(Nbeam)
    
y = x
x = x - xShear
y = y - yShear
[X, Y] = cp.meshgrid(x,y)
dx = x[1] - x[0]
radius = 0.5

RHO = 1/magFac*0.5*cp.sqrt(
    1/(radiusX)**2*(cp.cos(clockingRadians)*X + cp.sin(clockingRadians)*Y)**2
    + 1/(radiusY)**2*(cp.sin(clockingRadians)*X - cp.cos(clockingRadians)*Y)**2
    )

halfWindowWidth = cp.max(cp.abs((RHO[1, 0]-RHO[0, 0], RHO[0, 1] - RHO[0, 0])))
pupil = -1*cp.ones(RHO.shape)
pupil[cp.abs(RHO) < radius - halfWindowWidth] = 1
pupil[cp.abs(RHO) > radius + halfWindowWidth] = 0
grayInds = cp.array(cp.nonzero(pupil==-1))
# print('Number of grayscale points = %d' % grayInds.shape[1])

upsampleFactor = 100
dxUp = dx/float(upsampleFactor)
xUp = cp.linspace(-(upsampleFactor-1)/2., (upsampleFactor-1)/2., upsampleFactor)*dxUp
#xUp = (-(upsampleFactor-1)/2:(upsampleFactor-1)/2)*dxUp
[Xup, Yup] = cp.meshgrid(xUp, xUp)

subpixel = cp.zeros((upsampleFactor,upsampleFactor))

for iInterior in range(grayInds.shape[1]):

    subpixel = 0*subpixel

    xCenter = X[grayInds[0, iInterior], grayInds[1, iInterior]]
    yCenter = Y[grayInds[0, iInterior], grayInds[1, iInterior]]
    RHOup = 0.5*cp.sqrt(
    1/(radiusX)**2*(cp.cos(clockingRadians)*(Xup+xCenter) + cp.sin(clockingRadians)*(Yup+yCenter))**2
    + 1/(radiusY)**2*(cp.sin(clockingRadians)*(Xup+xCenter) - cp.cos(clockingRadians)*(Yup+yCenter))**2 
    )

    subpixel[RHOup <= radius] = 1
    pixelValue = cp.sum(subpixel)/float(upsampleFactor**2)
    pupil[grayInds[0, iInterior], grayInds[1, iInterior]] = pixelValue

plt.figure(2); plt.imshow(pupil); plt.colorbar(); plt.pause(0.1)
