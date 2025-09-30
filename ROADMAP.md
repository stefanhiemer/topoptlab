features

- [ ] add source for scalar field
- [ ] get rid of manual determinant calculation in elements with quadrature
- [ ] inverse homogenization
- [ ] stress constraints
- [ ] isotropic hyperelasticity
- [ ] Neo-Hooke
- [ ] multimaterial optimization
- [ ] anisotropic / orientation variable optimization
- [ ] anisotropic hyperelasticity
- [ ] include Globally Convergent Method of Moving symptotes (GCMMA) into TO main
- [ ] geometric multigrid (GMG)
- [ ] algebraic multigrid (AMG)
- [ ] optimality criteria with analytical update
- [ ] unstructured meshes from GMSH
- [ ] add aggregation functions

generalizations
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
- [ ] clean up filter length scales
- [ ] clean up filter by Langelaar (vectorization, 3D, etc. pp.)
- [ ] construct analytical elements with np.column_stack and reshape
- [ ] construct analytical elements with common subexpressions elimination

documentation 
- [ ] add the shape commentaries which now are being ignored.
- [ ] long usage example in README.md
- [ ] document MMA and GCMMA.
- [ ] document general constraints.
- [ ] How to use
- [ ] add FAQ
