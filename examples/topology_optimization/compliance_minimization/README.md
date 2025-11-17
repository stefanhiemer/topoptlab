These directories contain example cases for compliance minimization. Unless
explicitly stated, the examples make use of the `main` function in 
[`topology_optimization.py`](https://github.com/stefanhiemer/topoptlab/blob/main/topoptlab/topology_optimization.py).

cantilever_multiload.py

Demonstrates the use of TO for more then one set of boundary conditions. This 
is the same as Fig. 6 in "Efficient topology optimization in MATLAB using 88 
lines of code".

cantilever_passive2d.py

Demonstrates the use of TO with elements, that are permanently set as empty, 
i. e. "passive". This is the same as Fig. 6 in "Efficient topology optimization 
in MATLAB using 88 lines of code".

cantilever2d.py

Demonstrates the use of TO for the cantilever problem. This is the same as 
Fig. 5 in "Efficient topology optimization in MATLAB using 88 lines of code".

cantilever3d.py

Generalization of the previous example to 3D. It should be identical to 
Fig. 9 in "On multigrid-CG for efficient topology optimization ". However, as 
the paper is sparse in details regarding implementations, deviations might 
occur.

heatplate2d.py 

Maximize cooling for an evenly heated plate with one cooling site of T=0. This
is the equivalent problem to stiffness maximization just for the case of heat 
conduction. It should be identical to Fig. 5.6 on page 271 of the famous 
textbook "Topology Optimization - Theory, Methods and Applications."

heatplate3d.py 

Generalization to 3d of the previous case.

mbb_beam2d.py

**The** benchmark problem of TO for stiffness maximization with volume ineq. 
constraint. Best described by Fig. 1 and 2 in "Efficient topology optimization 
in MATLAB using 88 lines of code" and results should be identical to Fig. 3-5 
in the same paper. 

mbb_beam3d.py

Generalization of the previous case. 

slab_uniform_strain.py

maximize stiffness for a slab under uniform strain.

TO_from_scratch.py

This is a TO implementation for stiffness maximization from scratch using 
`topoptlab` as library. It does not use the `main()` routine mentioned before.

