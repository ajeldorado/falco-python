# falco-python
Fast Linearized Coronagraph Optimizer (FALCO) in Python 3

**********************************************
              UNDER DEVELOPMENT
- Start from the master branch for the most stable code.
- The Lyot coronagraph, vortex coronagraph, and shaped pupil 
Lyot coronagraph modes are now operational and have been 
validated against the MATLAB versions.
**********************************************

This branch is the result of my trying to get FALCO to use cupy (https://cupy.chainer.org/).  Cupy is a CUDA-accelerated library basically intended as a drop-in replacement for numpy.  I say "basically" because it is not that yet; it does not have all of numpy's functionality built in at this point, which causes some problems in a program as complex as FALCO.  But, as far as MOST array operations and math routines in numpy are concerned, this is pretty much true.  However, FALCO doesn't just use numpy - it also uses the interpolation routines from scipy as well as partially integrating PROPER, neither of which are built to be GPU-accelerated.  This means that there is a lot of shuttling stuff back and forth between the GPU and CPU, costing computation time.

This first attempt at GPU accelerating FALCO with cupy is pretty hacky.  The first thing I did was just do a find and replace of "np." with "cp.".  Then I cleaned up the resulting runtime errors, which were many.  This is not the way to do it properly, but it's a start.  This implementation of cupy-based FALCO is:

-More than twice as slow as the CPU-only version (great job achieving the original goal!).

-Shuttles stuff between the CPU and GPU WAY too much for its own good.

-Worsens the contrast instead of improving it.

-REALLY worsens the throughput instead of merely worsening it.

-Is untested with the vortex pathway.

-Prone to giving confusing errors when writing it or routines for it, mostly because some array or other is on GPU when it needs to be on the CPU and vice versa.

All in all, a smashing success, I'd say.  'Least it runs.  Anyway, making this work properly and produce usable wavefront solutions will be much more work than I can give to the project in my remaining two weeks at JPL, so here we are.  Seeing any sort of performance improvement by using the GPU is likely to require a ground-up rewrite of FALCO as a whole, as well as writing our own interpolation routines on the GPU and writing GPU versions of the PROPER routines used in FALCO.  If people want to optimize FALCO in this way, a high level of attention to detail is going to be necessary to make sure that what should be on the CPU is on the CPU and what should be on the GPU is on the GPU; it's really the shuttling between the two which is very slow and ultimately what is sinking the current implementation.

TL;DR Don't ever merge this crap into master without putting in A LOT of extra work.

Oh God, what have I done.

.

.

.

.

.

.

.

.

P.S.  If for godforsaken reason you ACTUALLY want to run this, start with EXAMPLE_main_WFIRST_LC.py.  That's what I used for all my testing, and you should at least make it through with no errors.
