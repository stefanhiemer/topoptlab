# Passive and Active Elements

In many practical topology optimisation problems some regions of the design domain 
must be kept fully void or fully solid regardless of what the optimiser would 
choose. The `main()` function in `topology_optimization` handles these through 
flags for each element and a policy how to deal with these flags:

* **`el_flags`** — an integer array of length $n$ (one entry per element) that 
labels each element as free, passive, or active.
* **`el_flags_policy`** — a dictionary that controls how the pipeline
  enforces those labels.

## Element flag values

| Value | Meaning |
|-------|---------|
| `0` | Free — optimised normally. |
| `1` | **Passive** — density is fixed at **0** (void). |
| `2` | **Active** — density is fixed at **1** (solid). |
| `3` | **Non-design** — excluded from the optimiser but participates in the density filter; its physical density is determined by the filter from surrounding elements. |

`el_flags` can be built by hand or with the geometry helpers in
`topoptlab.geometries`.  Concrete usage examples can be found in
`examples/topology_optimization/compliance_minimization/cantilever_passive2d.py`
and the compliant mechanism examples under
`examples/topology_optimization/compliant_mechanisms/`.

## The policy dictionary

`el_flags_policy` has three boolean keys: `"correct_forward"`, `"correct_backward"` and `"neglect_in_filter"`.  

All three default to `True`.  Two helpers in {mod}`topoptlab.utils` manage the
policy without you having to remember the key names:
{func}`topoptlab.utils.default_el_flags_policy` returns a fresh dictionary
with all defaults filled in, and
{func}`topoptlab.utils.check_el_flags_policy` back-fills any keys missing
from an existing dict — useful when you only want to override one or two
options.

### `"correct_forward"` (default: `True`)

After every forward filter pass the physical densities of prescribed elements
are snapped back to their target values:

```
xPhys[passive_elements] = 0.
xPhys[active_elements]  = 1.
```

Without this correction the density filter would smooth the sharp
boundary between a prescribed element and its free neighbours, causing
prescribed elements to drift away from their target values.

### `"correct_backward"` (default: `True`)

After the sensitivity filter pass the objective and constraint sensitivities
at prescribed elements are zeroed:

```
dobj[prescribed_elements]     = 0.
dconstrs[prescribed_elements] = 0.
```

This prevents the fixed-density elements from generating gradient signals
that would contaminate their free neighbours and bias the optimiser update.

### `"neglect_in_filter"` (default: `True`)

When the density-filter matrix $\mathbf{H}$ is assembled, any stencil entry
that involves a prescribed element — either as a source or as a target — is
dropped entirely.  Concretely, a stencil entry $(i,\,j)$ is kept only if
both element $i$ and element $j$ are free:

```{math}
H_{ij} = \begin{cases}
  \max\!\bigl(r_{\min} - d_{ij},\,0\bigr) & \text{if } \mathrm{flag}(i)=0
    \text{ and } \mathrm{flag}(j)=0, \\
  0 & \text{otherwise.}
\end{cases}
```

The row sums $H_s = \mathbf{H}\mathbf{1}$ are set to 1 for prescribed rows
to avoid division by zero during normalisation.

> **Note**: `"neglect_in_filter"` is only supported for `filter_mode="matrix"`.
> Using it with the Helmholtz PDE filter raises a `NotImplementedError`.

## Non-design elements (flag 3)

Non-design elements (flag `3`) are a special case: they are never updated by
the optimiser (`correct_backward` zeros their gradient before the optimizer
step), but they **always** stay in the filter stencil regardless of
`"neglect_in_filter"`.  Their physical density `xPhys` is therefore
determined by the filter from surrounding free elements, making them useful
for border regions that should receive a smooth, physically consistent density
without being design variables themselves.

## Putting it together

```python
from topoptlab.utils import default_el_flags_policy
from topoptlab.topology_optimization import main

el_flags = np.zeros(nelx * nely, dtype=int)
el_flags[passive_ids]    = 1
el_flags[active_ids]     = 2
el_flags[border_ids]     = 3   # non-design: filter-filled, not optimised

policy = default_el_flags_policy()
# example: let prescribed elements participate in the filter stencil
policy["neglect_in_filter"] = False

x, xTilde, xPhys, obj = main(...,
                             el_flags=el_flags,
                             el_flags_policy=policy,)
```

The three policy options are independent and can be combined freely — for
instance, one might want `"neglect_in_filter": True` to keep the filter
stencil clean while setting `"correct_backward": False` to allow gradients
to flow through prescribed elements when debugging sensitivities.