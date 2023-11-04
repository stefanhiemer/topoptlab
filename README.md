Things left to do:

topopt.py
- [x] docstrings for documentation
- [x] vectorize the simulation set up
- [ ] animation
- [x] check whether iK,jK,sK can be filtered by masking before setting up the
      stiffness matrix.

topopt_cholmod.py
- [x] docstrings for documentation
- [x] vectorize the simulation set up
- [ ] animation
- [x] row and column deletion uses np routines on scipy.sparse matrices.
      usually this is a bad idea as np.arrays and sparse matrices do not mix.
- [x] lines 112 to 121 to conversion from coo to csc and back just to convert 
      it to the cvx sparse format. I suspect that this is unnecessary.