# Plan: Separate main for physical (FE) solvers

## Goal

A main/driver that runs only the finite element problem (no optimization),
to test and apply FE on its own: patch tests, convergence studies (e.g. Q1
vs. Q2), and plain simulations. It is also the test bed for
`generated_elements_plan.md` and `edofmat_generalization_plan.md`.

## Current state

- The full physical workflow is only called inside
  `topoptlab/topology_optimization.py` (`prepare_phys_problems`,
  `solve_phys_problems`), with linear elasticity as fixed default.
- `examples/fem/*` (heat conduction, heat expansion, linear elasticity,
  Cahn-Hilliard, third medium contact) each build their setup by hand:
  `create_edofMat`, element matrices, `create_matrixinds`,
  `assemble_matrix`, `apply_bc`, `solve_lin`, export. Lots of duplicated
  code.
- The `FEMSolver` classes in `topoptlab/fem_solvers/` are unfinished and
  are not the basis of this plan.

## Decisions

- Functional style like `topology_optimization.main`, no class hierarchy.
- Built from the existing building blocks: `create_edofMat` (later the
  generalized version), `topoptlab.fem` (`create_matrixinds`,
  `assemble_matrix`, `apply_bc`), `solve_lin`, BC callables as used in the
  examples, existing export utilities.

## Phase 1: driver

- [ ] `fem_main(mesh, lk, bc, xPhys=None, body_forces=..., lin_solver=...,
      preconditioner=..., assembly_mode=..., export=..., display=...)`
      (name/location to be decided, e.g. `topoptlab/fem_main.py`):
  1. mesh and `edofMat`
  2. element matrices (element callable `lk`, optional material
     interpolation with `xPhys`)
  3. assembly
  4. boundary conditions
  5. solve
  6. derived quantities and export
- [ ] support several load cases and coupled/staggered problems only as far
      as the examples need it (heat expansion: temperature then elasticity).

## Phase 2: share code with topology optimization

- [ ] factor the physical part of `prepare_phys_problems` /
      `solve_phys_problems` so that `fem_main` and
      `topology_optimization.main` run the same physics path; the TO main
      only adds the optimization loop.
- [ ] test: results of `topology_optimization.main` unchanged.

## Phase 3: examples and tests

- [ ] port `examples/fem/*` to `fem_main` (thin scripts).
- [ ] patch tests (linear fields reproduced exactly) for Q1, later Q2.
- [ ] convergence study Q1 vs. Q2 against an analytical solution once the
      generated elements and the generalized `edofMat` exist.

## Open questions

- Name and location of the driver.
- Interface for problems with several fields (heat expansion, Cahn-Hilliard):
  one driver call per field, or a list of fields?
- Return value: dict of fields and derived quantities?
