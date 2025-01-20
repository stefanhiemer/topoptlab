These are various self contained codes. The big ones will disappear as I 
modularize/refactor topoptlab, so I will list the only ones that will remain
in the future:

topopt.py

Basic implementation more or less identical to the famous 88 line code paper. 
This code is just a modfication of the already existing code by Niels Aage and 
Villads Egede Johansen (JANUARY 2013) taken from https://www.topopt.mek.dtu.dk. 
Changes are 
- syntax changes from Python 2 to 3
- deletion of from __future__ import division
- vectorization of the filter and stiffness matrix construction, 
- insertion of plt.pause() as otherwise the figure appears only at the end
- deletion of unnecessary uses of tuples, indices, etc. pp.
- construct fixed bc without use of union to make it a little more efficient

topopt_cholmod.py

More efficient version of topopt.py and also taken from the same source with 
identical changes/modifications. 

topopth_time.py

Transient heat conduction to minimize f^{T} u at every timestep and serves as
demonstration code for transient topology optimization.