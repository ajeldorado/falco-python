# FALCO: Fast Linearized Coronagraph Optimizer in Python 3
[![Build Status](https://dev.azure.com/highcontrast/falco-python/_apis/build/status/ajeldorado.falco-python?branchName=master)](https://dev.azure.com/highcontrast/falco-python/_build/latest?definitionId=2&branchName=master)

The Fast Linearized Coronagraph Optimizer (FALCO) is an open-source package of routines and example scripts for coronagraphic focal plane wavefront correction. The goal of FALCO is to provide a free, modular framework for the simulation or testbed operation of several common types of coronagraphs, and the design of coronagraphs that use wavefront control algorithms to shape deformable mirrors (DMs) and masks. FALCO includes routines for pair-wise probing estimation of the complex electric field and Electric Field Conjugation (EFC) control, and we ask the community to contribute other wavefront correction algorithms and optical layouts. FALCO utilizes and builds upon PROPER, an established optical propagation library. The key innovation in FALCO is the rapid computation of the linearized response matrix for each DM, which facilitates re-linearization after each control step for faster DM-integrated coronagraph design and wavefront correction experiments. FALCO is freely available as source code in MATLAB at github.com/ajeldorado/falco-matlab and in Python 3 at github.com/ajeldorado/falco-python.

Developed by A.J. Riggs at the Jet Propulsion Laboratory, California Institute of Technology.
Major contributions and testing were provided by Garreth Ruane, Luis Marchen, Santos (Felipe) Fregoso, Erkin Sidick, Carl Coker, Navtej Saini, and Jorge Llop-Sayson.

**********************************************
### DOCUMENTATION

* The only non-standard library you need is PROPER, available only via download here: https://sourceforge.net/projects/proper-library/
* To get started, add PROPER and falco-python to your PYTHONPATH. Then try running some scripts in the falco-python/examples/ folder starting with EXAMPLE_main* or demo_*.
* Documentation on specific usage cases is available at the Matlab version's Github Wiki at https://github.com/ajeldorado/falco-matlab/wiki.
* For an overview of FALCO and its uses, refer to the SPIE conference paper "Fast Linearized Coronagraph Optimizer (FALCO) I: A software toolbox for rapid coronagraphic design and wavefront correction". 
DOI: 10.1117/12.2313812
**********************************************
