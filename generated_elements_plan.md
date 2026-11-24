# Plan: Generate numerical element modules from symfem

## Goal

Generate the numerical element modules (shape functions, derivatives, node
table) for Lagrange/Q elements of arbitrary order from symfem, with the same
API as the existing linear modules, and plug them into the numerical element
routines (`_lk_poisson`, mass, body force, advection, ...).

## Current state

The three hand-written modules in `topoptlab/elements/`

- `linear_interval.py`
- `bilinear_quadrilateral.py`
- `trilinear_hexahedron.py`

each contain:

| function | depends on the element? |
|---|---|
| `create_edofMat`, `apply_pbc` | yes (mesh), see `edofmat_generalization_plan.md` |
| `shape_functions`, `shape_functions_dxi`, `shape_functions_hessian` | yes |
| `check_inputs` | no (only number of reference coordinates) |
| `jacobian`, `invjacobian`, `bmatrix` | no (only via `shape_functions_dxi` and `xe`) |
| `jacobian_rectangle/cuboid`, `invjacobian_rectangle/cuboid`, `bmatrix_rectangle/cuboid` | only for affine rectangles/cuboids |

Corner order of these modules (counterclockwise from (-1,...,-1), bottom then
top layer) is identical to the convention of `topoptlab.symbolic.cell.node_ordering`.

The numerical element routines already take the element-dependent parts as
callables, e.g. `_lk_poisson(xe, k, ..., nquad=2, shape_functions_dxi=None,
invjacobian=None)`, defaulting to bilinear/trilinear for `None`.

Consumers of the modules: `elements/poisson.py`, `linear_elasticity.py`,
`nonlinear_elasticity.py`, `mass_scalar.py`, `mass_vector.py`,
`monomial_scalar.py`, `bodyforce.py`, `advection.py`, `intertia.py`, `gmg.py`,
`geometry_parser.py`.

## Decisions

- The existing linear modules stay untouched and serve as test case: the
  generated order-1 modules must reproduce them exactly.
- Only Lagrange/Q elements (the only symfem elements that fit our workflow,
  see ROADMAP.md "elements beyond Q1").
- Geometry: isoparametric by default; sub-/superparametric only as explicit
  choice of the user (ROADMAP.md phase 0).

## Phase 1: generic element-independent routines

- [ ] write `jacobian`, `invjacobian`, `bmatrix`, `check_inputs` once,
      generically, taking `shape_functions_dxi` (and `ndim`) as argument
      instead of duplicating them per element module. Location to be
      decided (e.g. `topoptlab/elements/isoparametric.py`).
- [ ] test: generic versions equal the existing per-module versions for
      interval/quadrilateral/hexahedron.

## Phase 2: code generation from symfem

- [ ] generator function (e.g. in `topoptlab/symbolic/code_conversion.py` or
      a new `topoptlab/symbolic/element_generation.py`):
  - input: `ndim`, `order`
  - basis from `base_cell(ndim, order)` on [-1,1], without `l`, `g`
  - generate `shape_functions`, `shape_functions_dxi`,
    `shape_functions_hessian` with `convert_to_code` (np.column_stack, same
    signatures `xi, eta, zeta` as the existing modules)
  - generate node table from `node_ordering`: node coordinates on [-1,1]
    and entity (corner/edge/face/interior) of each node
  - write a module file with these functions and constants
- [ ] decide naming/location of generated modules (e.g.
      `topoptlab/elements/lagrange_quadrilateral_2.py`)
- [ ] test: generated order-1 modules reproduce `linear_interval`,
      `bilinear_quadrilateral`, `trilinear_hexahedron` (shape functions,
      derivatives, hessians at random points).
- [ ] test: generated order-2/3 modules fulfill the Kronecker property at
      the node table and partition of unity.

## Phase 3: plug into numerical element routines

- [ ] pass the generated callables to the numerical routines
      (`shape_functions`, `shape_functions_dxi`, `invjacobian`).
- [ ] number of quadrature points must follow the order (currently
      `nquad=2` default); e.g. `nquad=order+1` or derived from the element.
- [ ] remove hard-coded Q1 assumptions in the routines (number of nodes
      2**ndim, ...).
- [ ] test: numerical Q2 Poisson/mass/elasticity matrices equal the symbolic
      ones from `topoptlab.symbolic` (Poisson Q2 already verified against an
      independent reference in 1D/2D/3D).

## Open questions

- Location of the generic routines and of the generated modules.
- Should the existing linear modules later be replaced by the generated ones
  (after the tests show they are identical), or stay permanently?
- Do the rectangle/cuboid special cases (`jacobian_rectangle`, ...) need
  generated counterparts, or are they superseded by the generic routines?
