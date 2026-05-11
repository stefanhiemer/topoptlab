features

- [ ] inverse homogenization
- [ ] stress constraints
- [ ] multimaterial optimization
- [ ] anisotropic / orientation variable optimization
- [ ] anisotropic hyperelasticity
- [ ] include Globally Convergent Method of Moving symptotes (GCMMA) into TO main
- [ ] unstructured meshes from GMSH
- [ ] add aggregation functions

generalizations
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
