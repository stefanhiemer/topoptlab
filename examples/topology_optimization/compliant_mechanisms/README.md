These directories contain example cases for compliance minimization. Unless
explicitly stated, the examples make use of the `main` function in 
[`topology_optimization.py`](https://github.com/stefanhiemer/topoptlab/blob/main/topoptlab/topology_optimization.py).

### `force_inverter2d.py`

Demonstrates the use of TO for the displacement maximization problem otherwise 
known as compliant mechanism with the special case of the force inverter best
described by Fig. 5.5 on page 269 of the famous textbook "Topology Optimization 
- Theory, Methods and Applications." . It should be close to results in Fig. 5.5 
on page 271 with minor changes as when checked with Octave (I have no Matlab) 
the stiffness matrix was not symmetric leading to slightly different results. 

### `force_inverter2d_control.py`

Generalization from displacement maximization to a displacement control, i. e. 
the optimized design should show a preset displacement under the given boundary
conditions.  

### `force_inverter3d.py`

Generalization of the displacement maximization case to 2D.