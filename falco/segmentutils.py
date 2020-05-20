import cupy as cp
import numpy as np
import os
import proper
import math
from . import utils
import falco

def hexSegMirror_addHexagon( centerRow, centerColumn, hexFlatDiam, arrayIn):
#%   This function adds hexagon to arrayIn, centered at (centerRow, 
#    centerColumn), with segment width 'hexFlatDiam'.
#%   Icp.ts:
#%   centerRow - row of hexagon center (samples)
#%   centerColumn - column of hexagon center (samples)
#%   hexFlatDiam - flat to flat width of the hexagonal segment (samples)
#%   arrayIn - Icp.t array
#%   
#%   Coordinate system origin: (rows/2+1, cols/2+1)

    [rows,cols] = arrayIn.shape

    [X,Y] = cp.meshgrid(cp.arange(-cols/2.,cols/2.), cp.arange(-rows/2.,rows/2.)) # Grids with Cartesian (x,y) coordinates 

    [RHOprime,THETA] = falco.utils.cart2pol(X-centerColumn,Y-centerRow)
#    RHOprime = cp.sqrt((X-centerColumn)**2+(Y-centerRow)**2)
#    THETA = cp.arctan2(Y-centerRow,X-centerColumn)

    power = 1000
    a = cp.exp(-(RHOprime*cp.sin(THETA)/(hexFlatDiam/2.))**power)
    b = cp.exp(-(RHOprime*cp.cos(THETA-cp.pi/6)/(hexFlatDiam/2.))**power)
    c = cp.exp(-(RHOprime*cp.cos(THETA+cp.pi/6)/(hexFlatDiam/2.))**power)
    HEX = a*b*c

    return arrayIn + HEX

def hexSegMirror_addHexSegment( centerRow, centerColumn, numRings, apDia, wGap, piston, tiltx, tilty, arrayIn):
#%hexSegMirror_addHexSegment Adds hexagonal mirror segment to arrayIn, 
#% centered at (centerRow, centerColumn). The full mirror have numRings rings of 
#% hexagonal segments, flat-to-flat diameter (in samples) of apDia, 
#% wGap (in samples) between segments, and piston, tiltx, and tilty
#% phase offsets. Piston is in units of waves. tiltx and tilty are waves
#% across the full flat-to-flat pupil diameter apDia. 
#%
#%   Icp.ts:
#%   centerRow - row of hexagon center (samples)
#%   centerColumn - column of hexagon center (samples)
#%   numRings - number of rings in the segmented mirror (samples)
#%   apDia - flat to flat aperture diameter (samples)
#%   wGap - width of the gap between segments (samples)
#%   piston - Segment piston in waves
#%   tiltx - Tilt on segment in horizontal direction (waves/apDia)
#%   tilty - Tilt on segment in vertical direction (waves/apDia)
#%   arrayIn - Icp.t array
#%   
#%   Coordinate system origin: (rows/2+1, cols/2+1)

    hexFlatDiam = (apDia-numRings*2*wGap)/(2*numRings+1)
    # hexRad = hexFlatDiam/sqrt(3);% center to vertex
    hexSep = hexFlatDiam + wGap

    [rows,cols] = arrayIn.shape

    [X,Y] = cp.meshgrid(cp.arange(-cols/2.,cols/2.), cp.arange(-rows/2.,rows/2.)) # Grids with Cartesian (x,y) coordinates 

    RHOprime = cp.sqrt((X-centerColumn)**2+(Y-centerRow)**2)
    THETA = cp.arctan2(Y-centerRow,X-centerColumn)

    power = 1000
    a = cp.exp(-(RHOprime*cp.sin(THETA)/(hexFlatDiam/2))**power)
    b = cp.exp(-(RHOprime*cp.cos(THETA-cp.pi/6)/(hexFlatDiam/2))**power)
    c = cp.exp(-(RHOprime*cp.cos(THETA+cp.pi/6)/(hexFlatDiam/2))**power)
    HEXamp = a*b*c
    
#    HEXphz = and(and(RHOprime*sin(THETA)<=(hexSep/2),RHOprime*cos(THETA-pi/6)<=(hexSep/2)),RHOprime*cos(THETA+pi/6)<=(hexSep/2))
#            *cp.exp(1i*2*pi*piston)
#            *cp.exp(1i*2*pi*tiltx/apDia*(X-centerColumn))
#            *cp.exp(1i*2*pi*tilty/apDia*(Y-centerRow))
    HEXphz = 1.
            
    return arrayIn + HEXamp*HEXphz


def hexSegMirror_getField( hexMirrorDict ):
#% This function returns the complex reflectance of the pupil function defined 
#% by a hexagonally segmented mirror 
#%   Icp.t: hexMirrorDict - Structure with the following variables 
#%   apDia - flat to flat aperture diameter (samples)
#%   wGap - width of the gap between segments (samples)
#%   numRings - number of rings in the segmented mirror (samples)
#%   N - size of NxN computational grid 
#%   pistons - Segment pistons in waves
#%   tiltxs - Tilts on segment in horizontal direction (waves/apDia)
#%   tiltys - Tilts on segment in vertical direction (waves/apDia)

    apDia = hexMirrorDict["apDia"] # flat to flat aperture diameter (samples)
    wGap = hexMirrorDict["wGap"] # samples
    numRings = hexMirrorDict["numRings"] # Number of rings in hexagonally segmented mirror 
    N = hexMirrorDict["Npad"]
    pistons = hexMirrorDict["pistons"]
    tiltxs = hexMirrorDict["tiltxs"]
    tiltys = hexMirrorDict["tiltys"]
    
    if('missingSegments' in hexMirrorDict.keys()):
        missingSegments = hexMirrorDict["missingSegments"]
    else:
        missingSegments = cp.ones(hexSegMirror_numSegments( numRings ),)
    
    N1 = 2**falco.utils.nextpow2(apDia)
    OUT = cp.zeros((N1,N1))
    
    hexFlatDiam = (apDia-numRings*2*wGap)/(2*numRings+1)
    hexSep = hexFlatDiam + wGap
    
    segNum = 0 # initialize the counter
    for ringNum in range(numRings+1): #= 0:numRings
    
        cenrow = ringNum*hexSep
        cencol = 0
        
        if(missingSegments[segNum]==0):
            OUT = hexSegMirror_addHexSegment( cenrow, cencol, numRings, apDia, 
                                             wGap, pistons[segNum], tiltxs[segNum], tiltys[segNum], OUT)
        segNum = segNum + 1
        
        for face in range(1,7): #= 1:6
            
            step_dir = cp.pi/6*(2*face+5)
            steprow = hexSep*cp.sin(step_dir)
            stepcol = hexSep*cp.cos(step_dir)
            
            stepnum = 1
    
            while(stepnum<=ringNum):
                cenrow = cenrow + steprow
                cencol = cencol + stepcol
                if(face==6 and stepnum==ringNum):
                    #%disp(['Finished ring ',num2str(ringNum)]);
                    pass
                else:
                    if(missingSegments[segNum]==0):
                        OUT = hexSegMirror_addHexSegment( cenrow, cencol, numRings, apDia, 
                                                         wGap, pistons[segNum], tiltxs[segNum], tiltys[segNum], OUT);
                        pass
                    segNum += 1
                stepnum += 1

    return falco.utils.padOrCropEven(OUT,N)


def hexSegMirror_getSupport( hexMirrorDict ):
#%hexSegMirror_getSupport Returns the support of the pupil function defined
#%by a hexagonally segmented mirror 
#%   Icp.t: hexMirrorDict - Dictionary with the following variables 
#%   apDia - flat to flat aperture diameter (samples)
#%   wGap - width of the gap between segments (samples)
#%   numRings - number of rings in the segmented mirror (samples)
#%   N - size of NxN computational grid 
#%   missingSegments - list of zeros and ones indicating if each segment is present 
#%   offset - centering offset vector [N/2+1+offset(1), N/2+1+offset(2)]

    apDia = hexMirrorDict["apDia"] # flat to flat aperture diameter (samples)
    wGap = hexMirrorDict["wGap"] # samples
    numRings = hexMirrorDict["numRings"] # Number of rings in hexagonally segmented mirror 
    N = hexMirrorDict["Npad"]
        
    if('missingSegments' in hexMirrorDict.keys()):
        missingSegments = hexMirrorDict["missingSegments"]
    else:
        missingSegments = cp.ones(hexSegMirror_numSegments( numRings ),)
        
    OUT = cp.zeros((N,N))
    
    hexFlatDiam = (apDia-numRings*2*wGap)/(2*numRings+1)
    hexSep = hexFlatDiam + wGap
    
    count = 0 #1
    for ringNum in range(numRings+1): #= 0:numRings
    
        cenrow = ringNum*hexSep
        cencol = 0
        
        if('offset' in hexMirrorDict.keys()):
            offset = hexMirrorDict["offset"]
            cenrow = cenrow + offset[0]
            cencol = cencol + offset[1]
        
        if(missingSegments[count]==1):
            OUT = hexSegMirror_addHexagon( cenrow,cencol, hexFlatDiam, OUT )

        count += 1
        
        for face in range (1,7): #= 1:6
            
            step_dir = cp.pi/6.*(2.*face+5.)
            steprow = hexSep*cp.sin(step_dir)
            stepcol = hexSep*cp.cos(step_dir)
            
            stepnum = 1
    
            while(stepnum<=ringNum):
                cenrow = cenrow + steprow
                cencol = cencol + stepcol
                if(face==6 and stepnum==ringNum):
                    #disp(['Finished ring ',num2str(ringNum)]);
                    pass
                else:
                    if(missingSegments[count]==1):
                        OUT = hexSegMirror_addHexagon( cenrow,cencol, hexFlatDiam, OUT )
                    count = count + 1
                stepnum = stepnum + 1

    return OUT


def hexSegMirror_numSegments( numRings ):
#%hexSegMirror_numSegments Returns the number of segments in a hexagonal
#%mirror with a given number of rings, 'numRings'
#%   numRings - number of rings in the mirror
#
#% loop through rings and add up the number of segments
    numOfSegments = 0
    for ringNum in range(numRings+1): #= 0:numRings
        numOfSegments = numOfSegments + 1 
        for face in range(1,7): #= 1:6
    
            stepnum = 1
    
            while(stepnum<=ringNum):
                if(face==6 and stepnum==ringNum):
                    #disp(['Finished ring ',num2str(ringNum)]);
                    pass
                else:
                    numOfSegments = numOfSegments + 1;
                    pass

                stepnum = stepnum + 1;
                
    return numOfSegments