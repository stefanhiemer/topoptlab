### `integration.py`

Compare the element-wise symbolic integration (`integrate`) with the separated
integration (`integrate_separated`) of element matrices, each serial and
parallel. The separated integration splits all matrix entries into
coefficients and terms of the integration variables, integrates each distinct
term once and reassembles the matrix. The integrands are taken from the
symbolic elements via `return_integrand=True`: Poisson (generic symmetric
conductivity), linear elasticity (generic anisotropic stiffness tensor), mass
matrix of a vector field and buckling (stress stiffness) matrix. By default
only Poisson and linear elasticity are run. Simplification is not included in
the timings. Prints a markdown table.

    python integration.py
    python integration.py --elements elasticity --dims 3
    python integration.py --elements poisson elasticity mass buckling
