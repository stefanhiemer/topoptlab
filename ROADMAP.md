features

- [ ] add source for scalar field
- [ ] assemble documentation
- [ ] inverse homogenization
- [ ] add bodyforce
- [ ] stress constraints
- [ ] isotropic hyperelasticity 
- [ ] Neo-Hooke
- [ ] multimaterial optimization
- [ ] anisotropic / orientation variable optimization
- [ ] anisotropic hyperelasticity
- [ ] include Globally Convergent Method of Moving symptotes (GCMMA)
- [ ] FEM with PETSC
- [ ] geometric multigrid (GMG)
- [ ] algebraic multigrid (AMG)
- [ ] optimality criteria with analytical update
- [ ] unstructured meshes from GMSH
- [ ] add aggregation functions
- [ ] add trapecoidal function for time series integration of objective
- [ ] smooth envelope function
- [ ] volume preserving minimum length scale filter / constraint 

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
- [ ] clean up filter by Langelaar (vectorization, 3D, etc. pp.)
- [ ] construct analytical elements with np.column_stack and reshape
- [ ] construct analytical elements with common subexpressions elimination