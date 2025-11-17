These directories contain example cases for compliance minimization. Unless
explicitly stated, the examples make use of the `main` function in 
[`topology_optimization.py`](https://github.com/stefanhiemer/topoptlab/blob/main/topoptlab/topology_optimization.py).

heatexpansion_bilayer-aniso.py

Displacement maximization problem for an anisotropic bilayer under uniform 
heat expansion. This is a from scratch implementation. This will later be 
generalized to non-uniform temperature distribution, i. e. a weakly coupled 
thermo-mechanical problem and wrapped in the `main` routine. 