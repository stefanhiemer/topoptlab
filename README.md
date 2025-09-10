# topoptlab 

This project is a collection of topology optimization techniques many of which
are already published in Matlab. This project is there to make these techniques
available to people without Matlab licenses but also to offer a basis on which
new methods can be implemented quickly. Please be aware that Matlab is 
optimized for tasks like topology optimization while Python is a general 
purpose language, so you will likely see slower performance here than in 
equivalent Matlab scripts.

# Features
Here is an (incomplete) list of features:

- Topology Optimization
  - Material interpolations
    - Modified SIMP
    - RAMP
    - (Hashin–Shtrikman) bound-based
  - Objectives
    - Compliance / stiffness minimization / control
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
    - MMA / GCMMA (Arjen Deetman)
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

- Example cases
  - MBB beam 2D / 3D
  - Cantilever 2D
  - Compliant mechanism 2D
  - Heat plate cooling 2D (stationary and transient)

- Solver / preconditioner for linear systems (beyond `scipy.sparse`)
  - Sparse Cholesky decomposition of CHOLMOD via CVXOPT
  - Algebraic multigrid
  - Geometric multigrid
  - Block-sparse preconditioner

# How to use 

Topoptlab can be used either as a black box function that returns an optimal 
design to you once provided with a set of boundary conditions and parameter or 
as a general purpose topology optimization and finite element library. If the 
boundary conditions are already available, then running an optimization can be 
quite simple. E. g. if one wants to reproduce the famous MBB beam in 2D it 
amounts to:

```
from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import mbb_2d

main(nelx=60, nely=20, volfrac=0.5, penal=3.,
     rmin=2.4, 
     optimizer="oc",
     file="mbb_2d",
     bcs=mbb_2d,
     display=True,
     export=True)
```

# Monolithic codes

In the monolithic_code directory, you can find codes that are self contained 
and do not use the topoptlab module written here. These codes are purely there 
to either test new frameworks (e. g. JAX or PETSC) or for 
teaching/demonstration purposes. If you are completely new to topology 
optimization, this is where you start and I suggest to start with the 
topopt88.py.

# Installation
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

# Run tests
Run fast tests (finish in under one minute)
```
pytest
```
Run slow tests (take a few minutes)
```
pytest -m slow
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

# Roadmap

See [ROADMAP.md](./ROADMAP.md) for a list of upcoming features.

# Make documentation

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

# Getting report bugs, and suggest enhancements

If you found a bug or you want to suggest a new feature/enhancement, submit it 
on the [issue tracker](https://github.com/stefanhiemer/topoptlab/).

# How to contribute

If you want to contribute, fork the repository and open a pull request. 
However before doing that contact the maintainers via an enhancement suggestion 
in the [issue tracker](https://github.com/stefanhiemer/topoptlab/).

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