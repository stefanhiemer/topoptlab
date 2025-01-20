# toptopt lab 
This project is a collection of topology optimization techniques many of which
are already published in Matlab. This project is there to make these techniques
available to people without Matlab licenses but also to offer a basis on which
new methods can be implemented quickly. Please be aware that Matlab is 
optimized for tasks like topology optimization while Python is a general 
purpose language, so you will likely see slower performance here than in 
equivalent Matlab scripts.

# Monolithic Codes

In the monolithic_code directory, you can find codes that are self contained 
and do not use the topoptlab module written here (at the moment not true). 
These codes are purely there to either test new frameworks (e. g. JAX or PETSC)
or for teaching/demonstration purposes. If you are completely new to topology 
optimization, this is where you start and I suggest to start with the topopt.py.

# Installation
Basic installation and run tests by exectuing
```
pip install .[tests]
```
in top directory. Editable installation (recommended if you want to edit 
something in the code) 
```
pip install -e .[tests]
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

# Things left to do:

features in topopt.py 
- [x] vectorize the simulation set up
- [x] fix live animation
- [x] check whether iK,jK,sK can be filtered by masking before setting up the
      stiffness matrix.
- [x] Helmholtz filter (with dense cholesky as scipy has no sparse cholesky)
- [x] Helmholtz filter with LU decomposition (tested it, but gives different
      results. I guess that Super-LU uses a different precision.)
- [x] different examples than just MBB beam
- [x] multiload cases
- [x] active/passive elements
- [x] compliant mechanism
- [x] heat conduction
- [x] Haevisde filter
- [x] volume conserving eta projection
- [x] include Method of Moving symptotes (MMA)
- [x] additive manufacturing filter by Langelaar
- [ ] clean up filter by Langelaar (vectorization etc. pp.)
- [ ] include Globally Convergent Method of Moving symptotes (GCMMA)
- [ ] FEM with PETSC
- [ ] algebraic multigrid (AMG)
- [ ] stress constraints
- [ ] get rid of strange indexing for density/sensitivity filter
- [ ] kick out from __future__ import division
- [ ] np.union1d for setting up K is expensive. check if necessary.
- [ ] accelerated optimality criteria (aOC)
- [ ] 3D