symbolic features
- [ ] handle material properties consistently in API
- [ ] allow direct file creation of analytic elements
- [ ] convert integrals to common form, split common form into parts, solve in parallel, reverse substitution, simplify
- [ ] extract prefactors from VectorFunctions and MatrixFunctions
- [ ] for solving integrals, pass from sympy to FRICAS, integrate, pass back  
- [ ] use quadrature rules


features

- [ ] inverse homogenization
- [ ] multimaterial optimization
- [ ] anisotropic / orientation variable optimization
- [ ] bodyfitted meshes e. g. via ParaView

into TO main
- [ ] unstructured meshes from GMSH
- [ ] add aggregation functions

generalizations
- [ ] elements to same form
- [ ] add 1d elements
- [ ] add 1D to topology optimization
- [ ] generalize stress constraints to arbitrary failure criteria
- [ ] wrap different physical phenomena in FEM_Phys class
- [ ] generalize boundary conditions
- [ ] generalize filters to multimaterial optimization
- [ ] generalize filters to orientation variables
- [ ] generalize filters to arbitrary filter combinations
- [ ] wrap physical problems in solvers
- [ ] add 2D Hashin-Shtrikman bounds
- [ ] add anisotropic bounds
- [ ] add constraints as optional functions
- [ ] generalize export function to divide between nodal and elemental
      variables

clean-ups and streamlining/refactoring
- [ ] exchange scipy.sparse matrices for sparse arrays
- [ ] clean up filter length scales
- [ ] clean up filter by Langelaar (vectorization, 3D)
- [ ] construct analytical elements with np.column_stack and reshape
- [ ] construct analytical elements with common subexpressions elimination

documentation 
- [ ] add the shape commentaries which now are being ignored.
- [ ] document MMA and GCMMA.
- [ ] document general constraints.
- [ ] add FAQ
