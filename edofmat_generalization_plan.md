# Plan: Generalized element dof matrix for the regular mesh

## Goal

Create the element degree of freedom matrix (`edofMat`) for Lagrange/Q
elements of arbitrary order on the regular mesh, following our node
conventions, derived from the (correctly sorted) symfem nodes of
`topoptlab.symbolic.cell.node_ordering`.

## Current state

`create_edofMat` exists only for linear elements:

- `topoptlab/elements/linear_interval.py`
- `topoptlab/elements/bilinear_quadrilateral.py`
- `topoptlab/elements/trilinear_hexahedron.py`

Conventions of the existing 2D version:

    n1 = (nely+1)*elx + ely
    n2 = (nely+1)*(elx+1) + ely
    edofMat = column_stack((n1+1, n2+1, n2, n1)) * nnode_dof (+ dof offset)

- global nodes: y is the fast index, x the slow one (3D: z slowest)
- the y index is flipped: corner (-1,-1) of the element is `n1+1`, i.e. the
  row below `ely`
- dofs: node index * `nnode_dof` + local dof number

The helper indices `n1`-`n4` returned by the old functions are a relic. They
are unpacked in about 20 places but never used (one example passes them to
`assemble_helmholtz_filter`, where they end up unused in `**kwargs`).

## Decisions

- The existing `create_edofMat` functions stay untouched (test case).
- The new function returns only `edofMat` (no helper indices).
- Only complete tensor-product elements (Lagrange/Q). Serendipity etc. would
  leave unused lattice points and need a compressed numbering (later, see
  ROADMAP.md).

## Approach

For order p, the nodes of the regular mesh form a finer lattice with
(p*nelx+1) x (p*nely+1) (x (p*nelz+1)) points. Each local node with reference
coordinates xi in {-1, -1+2/p, ..., 1} has the integer lattice offset

    o = p*(xi+1)/2

which follows exactly from the node coordinates of `node_ordering`. The
global lattice position of local node (o_x, o_y, o_z) in element
(elx, ely, elz) is

    ix = p*elx + o_x
    iy = p*ely + (p - o_y)      (flipped y, as in the existing convention)
    iz = p*elz + o_z

and the global node index

    2D: (p*nely+1)*ix + iy
    3D: (p*nelx+1)*(p*nely+1)*iz + (p*nely+1)*ix + iy

Only the offset table (local node -> (o_x, o_y, o_z)) is element dependent;
it is generated from symfem (e.g. as constant in the generated element
module of `generated_elements_plan.md`). The function itself is generic.

## Phase 1: generic function

- [ ] `create_edofMat(nelx, nely, nelz, nnode_dof, offsets, order,
      dtype=np.int32)` (name/location to be decided) computing the global
      node indices from the offset table, vectorized over elements.
- [ ] total number of nodes: prod(p*nel+1).
- [ ] test: for p=1 identical to the old `create_edofMat` in 1D/2D/3D and
      for several `nnode_dof`.
- [ ] test: for p=2,3 every global node is used, shared nodes of
      neighbouring elements coincide, and the node coordinates computed
      from the lattice match the mapped reference coordinates.

## Phase 2: offsets from symfem

- [ ] function that converts the node coordinates of `node_ordering` to the
      integer offset table (exact via sympy Rationals).
- [ ] write the offset table into the generated element modules.

## Phase 3: periodic boundary conditions

- [ ] generalize `apply_pbc` to order p (identify lattice nodes on opposite
      boundaries).
- [ ] test: p=1 identical to the old `apply_pbc`.

## Phase 4: consumers (later)

- [ ] places that assume `ndof = prod(nel+1)*nnode_dof` or
      `n_nodaldof = KE.shape[-1]/2**ndim` (e.g.
      `topology_optimization.prepare_phys_problems`) must use the
      generalized node count.
- [ ] removing the unused helper indices from the old functions and their
      callers goes to the clean-up list of ROADMAP.md, not into this plan.

## Open questions

- Name and location of the generic function.
- Node coordinates for export (VTK of higher-order elements) also from the
  lattice?
