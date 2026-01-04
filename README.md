# topoptlab 
[![Documentation Status](https://readthedocs.org/projects/topoptlab/badge/?version=latest)](https://topoptlab.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/topoptlab?color=blue&label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/topoptlab/)
[![CI](https://github.com/stefanhiemer/topoptlab/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/stefanhiemer/topoptlab/actions/workflows/ci.yaml)

This project provides a collection of topology optimization techniques, many of 
which were originally published MATLAB scripts. The goal is to make these methods 
broadly accessible in Python, in a style similar to the well-known 
[`88 line code`](https://link.springer.com/article/10.1007/s00158-010-0594-7) 
by Andreassen and Sigmund. The package can be used in two ways:

- As a library:  
  Write your own scripts in the concise style of the 88 line MATLAB code. A simple 
  re-write with extension to 3D, variable boundary conditions and changeable 
  physics (lin. elasticity, heat conduction) can be found in 
  [`TO_from_scratch.py`](https://github.com/stefanhiemer/topoptlab/blob/main/examples/topology_optimization/compliance_minimization/TO_from_scratch.py)

- As a modular “black-box” routine:  
  Use the main function in [`topology_optimization.py`](https://github.com/stefanhiemer/topoptlab/blob/main/topoptlab/topology_optimization.py) as a black box function that returns an optimal 
  design once provided with a set of boundary conditions and parameters to run topology optimization directly. If the boundary conditions are already available, then running an optimization can be quite simple. 
  E. g. if one wants to reproduce the famous MBB beam in 2D it amounts to:
```
from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import mbb_2d

main(nelx=60, nely=20, 
     volfrac=0.5,
     rmin=2.4, 
     optimizer="oc",
     bcs=mbb_2d)
```
For tutorials, explanations, and full documentation, see the 
[online documentation](https://topoptlab.readthedocs.io/en/latest/).

## Features
Here is an (incomplete) list of features:

- Topology Optimization
  - Material interpolations
    - Modified SIMP
    - RAMP
    - (Hashin–Shtrikman) bound-based
  - Objectives
    - Compliance minimization (stiffness maximization) / control
    - Displacement maximization / control
  - Filters
    - Sensitivity
    - Density
    - Additive manufacturing filter (Langelaar, 2D)
    - Projections
      - Guest (2004)
      - Sigmund (2007)
      - Volume-conserving projection (Xu, 2010)
  - Constrained optimizers
    - MMA / GCMMA via `mmapy` by Arjen Deetman
    - Optimality criteria method
  - Unconstrained optimizers
    - Gradient descent
    - Barzilai–Borwein
  - Design analysis
    - Gray level indicator
    - Lengthscale violations

- FEM
  - Linear elasticity
  - Heat conduction
  - Heat expansion
  - Cahn–Hilliard
  - analytic element integration via `symfem` by Matthew Scroggs

- Example cases
  - MBB beam 2D / 3D
  - Cantilever 2D / 3D
  - Compliant mechansim (force inverter) 2D / 3D
  - Heat plate cooling 2D (stationary and transient)

- Solver / preconditioner for linear systems (beyond `scipy.sparse`)
  - Sparse Cholesky decomposition of CHOLMOD via CVXOPT
  - Algebraic multigrid (via `pyAMG` and self-written)
  - Geometric multigrid
  - Block-sparse preconditioner


For a list of upcoming features, look at the 
[ROADMAP.md](https://github.com/stefanhiemer/topoptlab/blob/main/ROADMAP.md).


## Monolithic codes

In the monolithic_code directory, you can find codes that are self contained 
and do not use the topoptlab module written here. These codes are purely there 
to either test new frameworks (e. g. JAX or PETSC) or for 
teaching/demonstration purposes. If you are completely new to topology 
optimization, this is where you start and I suggest to start with the 
[`topopt88.py`](https://github.com/stefanhiemer/topoptlab/blob/main/monolithic_codes/topopt88.py).

# Installation, documentation and tests

## Installation with pip from PyPI
Install everything needed for the basic installation, documentation and tests:
```
pip install topoptlab[tests,docs]
```
## Installation with pip from the Github Repository
Clone the repository 
```
git clone https://github.com/stefanhiemer/topoptlab
```
and execute
```
pip install .[tests,docs]
```
in top directory. Editable installation (recommended if you want to edit 
something in the code) 
```
pip install -e .[tests,docs]
```

## Run tests
Run fast tests (finish in under one minute)
```
pytest
```
Run slow tests (take a few minutes)
```
pytest -m slow
```

## Make documentation

The documentation can be build via Sphinx 

```
cd docs/
sphinx-apidoc -o source/ ../topoptlab/ ../topoptlab/legacy --force --no-toc --separate
make html
```
and displayed in your browser by drag and drop or if you are on Linux
```
xdg-open build/html/index.html
```

## Build package and release on PyPI (only for maintainers and developers)

Make sure the necessary packages have been installed:
```
pip install -e .[pypi]
```
Build the package
```
python -m build
```
and upload via twine
```
python -m twine upload dist/*
```
For testing reasons it may be smarter to first upload on TestPyPI 
```
python -m twine upload --repository testpypi dist/*
```
and then install the package via 
```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple \ topoptlab
```

# Getting report bugs, and suggest enhancements

If you found a bug or you want to suggest a new feature/enhancement, submit it 
on the [issue tracker](https://github.com/stefanhiemer/topoptlab/issues).

# How to contribute

Contact the maintainers via an enhancement suggestion in the 
[issue tracker](https://github.com/stefanhiemer/topoptlab/issues) and 
look at the [ROADMAP.md](https://github.com/stefanhiemer/topoptlab/blob/main/ROADMAP.md)
for a list of upcoming features. If you decide to contribute, fork the 
repository and open a pull request. For detailed instructions check out 
[CONTRIBUTING.md](https://github.com/stefanhiemer/topoptlab/blob/main/CONTRIBUTING.md).

# Acknowledgments

We acknowledge the support by the Humboldt foundation through the Feodor-Lynen 
fellowship. We acknowledge partial support from the ARCHIBIOFOAM project which 
received funding from the European Union’s Horizon Europe research and 
innovation programme under grant agreement No 101161052. Views and opinions 
expressed are however those of the author(s) only and do not necessarily 
reflect those of the European Union or European Innovation Council and SMEs 
Executive Agency (EISMEA). Neither the European Union nor the granting 
authority can be held responsible for them.

Special thanks are extended to Christof Schulze for his valuable suggestions on 
code improvements, to Stefano Zapperi for hosting at the University of Milan 
and for his continuous support, and to Peter Råback for sharing his vast 
knowledge on FEM related implementation details.