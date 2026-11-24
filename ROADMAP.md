symbolic features
- [ ] handle material properties consistently in API
- [ ] allow direct file creation of analytic elements
- [ ] convert integrals to common form, split common form into parts, solve in parallel, reverse substitution, simplify
- [ ] extract prefactors from VectorFunctions and MatrixFunctions
- [ ] for solving integrals, pass from sympy to FRICAS, integrate, pass back

elements beyond Q1 (higher order and other symfem elements)
currently only nodal Lagrange/Q elements fit the workflow (node_ordering
rejects everything else). create_edofMat only exists for linear elements, so
even Q2 Lagrange cannot be assembled yet.
- phase 0: foundation (needed for every element beyond Q1)
  - [ ] describe dofs per entity (corner, edge, face, interior) with entity
        number and multiplicity instead of per node (info already in
        node_ordering)
  - [ ] entity-based edofMat: global numbering of corners, edges, faces and
        cells of the structured grid; for gmsh meshes extract edges/faces
        from elnodes
  - [ ] geometry element as explicit option (e.g. geometry_order) of element
        routines and jacobian, independent of the field element. default
        stays isoparametric, sub-/superparametric is a conscious choice of
        the user. elements that cannot be isoparametric (no corner nodes or
        no partition of unity, e.g. bubble, dPc, Rannacher-Turek) raise an
        error asking for an explicit geometry element.
- phase 1: nodal elements with existing weak forms
  - [ ] classical serendipity CPS8/C3D20 as own nodal element (symfem's
        serendipity uses integral moments for order >= 2)
  - [ ] Rannacher-Turek (point values at face midpoints, no orientation)
  - [ ] bubble enrichment (P1+bubble, MINI for Stokes): combined element
        spaces, static condensation of interior dofs
  - [ ] verify C3D27 face node order (nodes 21-26) against ABAQUS
        documentation
- phase 2: C1 elements (new weak forms)
  - [ ] Bogner-Fox-Schmit, Hermite: several dofs per node (value +
        derivatives), scaling of derivative dofs with l. BFS is C1 only on
        rectangles, not on sheared parallelograms.
  - [ ] Kirchhoff plates, Euler-Bernoulli beams, conforming huhu
- phase 3: discontinuous elements (dPc, DG)
  - [ ] symbolic face integrals over pairs of elements (jumps, averages,
        penalty)
  - [ ] loop over interior faces of the grid for assembly
- phase 4: H(div)/H(curl) (BDFM, NCE/NCF, Regge, Hu-Zhang, ...)
  - [ ] mixed formulations: Darcy/mixed Poisson, mixed elasticity, Maxwell
  - [ ] Piola transformations (contravariant/covariant)
  - [ ] orientation/signs of edge and face dofs in assembly
  - [ ] block preconditioners for saddle point problems
  - [ ] application e. g. Darcy/Brinkman flow topology optimization
- not planned: non-Ciarlet elements (enriched Galerkin, direct serendipity)

features

- [ ] inverse homogenization
- [ ] add buckling constraints
- [ ] multimaterial optimization
- [ ] anisotropic / orientation variable optimization
- [ ] bodyfitted meshes e. g. via ParaView
- [ ] unstructured meshes read by GMSH
- [ ] add aggregation functions

generalizations
- [ ] cast elements to same form
- [ ] different 
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
