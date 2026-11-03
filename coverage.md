# Test coverage action list

Status: 2026-11-03, branch `feature/inverse-homogenization`.

## Baseline

- Suite (`pytest -m 'not slow'`): 442 passed, 2 failed, 1 skipped, 3 deselected (about 3 min).
- Reported coverage: **60%** of imported statements. The real figure is lower: about 60 modules are never imported by any test, and coverage does not list them, because 10 subpackages have no `__init__.py`.
- Example tests in `tests/test_examples.py` run the scripts via `subprocess`, so their coverage is not recorded.

## Out of scope for now

- `topoptlab/legacy/`: will be deleted.
- `topoptlab/topology_opt/` (new driver replacing `topology_optimization.py`): not yet wired into the tests (known issue).
- `topoptlab/fem_solvers/` and `problem_solver.py`: unfinished; the new driver depends on them.
- No new tests against the old `topology_optimization.py`. Keep the existing ones until the new driver can replace them.

## 0. Housekeeping (about 1 hour)

- [ ] Add `__init__.py` to `elements`, `filter`, `symbolic`, `topology_opt`, `optimizer`, `bounds`, `example_bc`, `legacy`, `blocksparse` and `material_models`. Without them, modules no test imports are left out of the report.
- [ ] Add a coverage config to `pyproject.toml`:
  ```toml
  [tool.coverage.run]
  source = ["topoptlab"]
  omit = [
      "topoptlab/legacy/*",
      "topoptlab/topology_opt/*",
      "topoptlab/fem_solvers/*",
      "topoptlab/problem_solver.py",
      "topoptlab/draw_functions.py",
      "topoptlab/profile_utils.py",
  ]
  ```
  Add `pytest-cov` to the `tests` extras.
- [ ] Fix the failing `tests/test_examples.py::test_fdm`: `examples/fdm/heateq.py` and `heateq_aniso.py` block on `input("Press any key...")`. Guard that call behind the display flag.
- [ ] Fix the failing `tests/bounds/test_reussvoigt.py::test_reuss_dx[x2-props2]`. Relative error is 2.5e-5, which is finite-difference noise, so loosen `rtol`.
- [ ] Optional: `examples/fdm/allencahn_fdm.py` ignores its command-line arguments and runs to t=4000. Make it respect them so the smoke test is fast.
- [ ] Delete `tests/test_foldingmech.py` together with `legacy/`.

## 1. Gradient and value checks (highest value per test)

- [ ] `objectives.py` (17%, 238 missed): extend the finite-difference harness in `tests/test_objectives.py` to `compliance`, `compliance_squarederror`, `var_maximization`, `var_squarederror`, `stress_pnorm`, `support`, `two_point_correlation*` and `gray_indicator_constraint`.
- [ ] `param_continuation.py` (9%, 137 missed): the update rules (`beta_scaling`, `beta_translation`, `scaling`, `change_constraint`, `dunning_beta_continuation`, `level_beta_continuation`, `update_move_limit`, `scale_move_limit`) are mostly scalar. Test them against hand-computed values.
- [ ] `optimizer/optimality_criterion.py` (32%): call `oc_top88`, `oc_mechanism` and `oc_generalized` directly on small arrays. Check the bounds, move limit and volume constraint.

## 2. Cheap parametrized sweeps

- [ ] `example_bc/lin_elast.py` (26%, 162 missed): one test parametrized over all 21 boundary-condition functions on a 4×3 (or 4×3×2) mesh. Check shapes, DOF ranges, and that fixed and loaded DOFs don't overlap.
- [ ] `log_utils.py` (26%, 149 missed): write with `SimpleLogger` and read back with `parse_simple_logfile` in `tmp_path`. Cover `EmptyLogger`, `init_logging` and `log_material_properties`.
- [ ] `geometries.py` (19%): check output shape, value range and volume fraction.
- [ ] `output_designs.py` (18%): export with `export_vtk` to `tmp_path` and read it back with `meshio`.
- [ ] `rotation_matrices.py` (28%): check R·Rᵀ = I, det(R) = 1, and that a zero angle gives the identity.
- [ ] `design_analysis.py` (29%) and `utils.py` (75%): fill gaps with small known-answer tests.

## 3. Filters and linear solvers (called directly, not through the driver)

- [ ] `filter/matrix_filter.py` (45%), `filter/haeviside_projectors.py` (67%), `filter/haeviside_projection.py` (69%), `filter/workflows.py` (52%): check that a constant field stays constant, that derivatives match finite differences, and that the forward and adjoint operators are consistent.
- [ ] `filter/filter_bc.py` (never imported, 267 lines): same checks.
- [ ] `solve_linsystem.py` (53%), `gmg.py` (34%), `multigrid.py` (never imported): parametrize over solver and preconditioner names on a small SPD system and compare with `scipy.sparse.linalg.spsolve`.
- [ ] `optimizer/mma_utils.py` (40%), `optimizer/stepsize.py` (50%), `optimizer/line_search.py` (never imported), `aggregation.py` (50%): small known-answer tests; finite-difference checks for the aggregation functions.

## 4. Elements, material models, symbolic (lower priority)

- [ ] Never-imported elements (`linear_interval`, `mass_vector`, `monomial_scalar*`, `stress_measures`, `hessian`, `intertia`): reuse the symmetry and rigid-body-mode checks from `tests/elements`.
- [ ] `material_models/invariant_hyperelasticity.py` (never imported, 370 lines): check tangents against finite differences.
- [ ] `symbolic/rotation`, `nonlin_elasticity`, `neohookean`, `huhu`, `buckling`, `heatexpansion`, `inertia`: compare symbolic output with the hard-coded element implementations (pattern in `tests/symbolic`). Also cover the gaps in `symbolic/strain_measures.py` (26%) and `symbolic/matrix_utils.py` (75%).
- [ ] `fdm.py` and `timestepper.py`: decide whether they count as stable; if so, test one small case each.

## Later: once `fem_solvers` is finished

- [ ] Point the driver tests (`test_compliance`, `test_filters`, `test_inverter`, `test_passive`, `test_solver`, `test_accelerators`) at `topoptlab.topology_opt.driver.main`. During the transition, parametrize over both drivers with the same reference objective values, so the new driver has to reproduce the old results.
- [ ] Add one heat-equation case (like `examples/topology_optimization/compliance_minimization/heatplate2d_solver.py`).
- [ ] Add unit tests for the `topology_opt` helpers: `create_mesh`, `initialize_design`, `initialize_history`/`update_history`, `initialize_optimizer`, `converged`, `copy_relevant_fields`, and `initialize_materialinterpolation` (None, dict and list inputs).
- [ ] Delete `topology_optimization.py`, and remove `topology_opt/*`, `fem_solvers/*` and `problem_solver.py` from the coverage omit list.
- [ ] Keep only new-driver examples in `tests/test_examples.py`. If their coverage should count, run them in-process via `runpy`.

## How to measure

```bash
pip install pytest-cov
pytest --cov=topoptlab --cov-report=term-missing:skip-covered
```
