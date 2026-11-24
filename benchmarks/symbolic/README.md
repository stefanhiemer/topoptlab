### `integration.py`

Compare the element-wise symbolic integration (`integrate`) with the separated
integration (`integrate_separated`) of element matrices, each serial and
parallel. The separated integration splits all matrix entries into
coefficients and terms of the integration variables, integrates each distinct
term once and reassembles the matrix. The integrands are taken from the
symbolic elements via `return_integrand=True`: Poisson (generic symmetric
conductivity), linear elasticity (generic anisotropic stiffness tensor), mass
matrix of a vector field and buckling (stress stiffness) matrix. All elements
are run in 1D, 2D and 3D for element order 1 and 2 (order 2 only parallel)
with both integration backends, sympy and symintegral. Each integration is aborted after `TIMEOUT` seconds.
Simplification is not included in the timings. Prints a markdown table and
writes the timings to `integration.json` (times of `null` are timeouts).
Results already in `integration.json` are not recomputed, so an aborted run
can be continued; delete the file for a fresh run.

    python integration.py

### `plot_integration.py`

Plot the wall clock time vs. number of dimensions from `integration.json`, one
column per element and one row per element order. Color encodes the method, line style the backend. Saves
`integration.png`.

    python plot_integration.py
