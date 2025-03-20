These are various self contained codes. The big ones will disappear as I 
modularize/refactor topoptlab, so I will list the only ones that will remain
in the future:

topopt88.py

Basic implementation more or less identical to the famous 88 line code paper. 
This code is just a modfication of the already existing code by Niels Aage and 
Villads Egede Johansen (JANUARY 2013) taken from https://www.topopt.mek.dtu.dk. 
Changes are 
- syntax changes from Python 2 to 3
- deletion of from __future__ import division
- extract filter construction into individual function
- switch change calculation to "np.abs(x-xold).max()" as it is better readable 
  and likely faster then the previous expression although probably only 
  marginally
- vectorization of the filter (could be done better I think) 
- vectorization of the stiffness matrix construction 
- insertion of plt.pause() as otherwise the figure appears only at the end
- deletion of unnecessary uses of tuples, indices, etc. pp.
- construct fixed bc without use of union to make it a little more efficient
- for constructing indices of stiffness matrix exchange Kronecker product with 
  repeat() and tile(). I guess it is faster (did not check), but more 
  importantly more clear in its meaning.

topopt_cholmod88.py

More efficient version of topopt.py and also taken from the same source with 
identical changes/modifications. 

topopt88h.py

Same as topopt88.py just for heat conduction. Originally taken from page 271 of 
the Bendsoe & Sigmund textbook and transcribed from the old 99 line code to the 
88 line code.

topopt_rank2.py

topology optimization using rank 2 laminates. Not yet finished.

topopt88m.py

Similar to topopt88.py just for compliant mechanism synthesis (maximized 
displacement). Originally taken from page 269 of the Bendsoe & Sigmund textbook 
and transcribed from the old 99 line code to the 88 line code. The results are 
close to the old 99 line code version but do not 100 % agree. I believe 
rounding errors in the old code to be the reason as the stiffness matrix there
seems to be asymmetric as well due to rounding errors. 

topopt88mh.py

Thermal bimaterial compliant mechanism and is just a proof of concept right 
now. The optimization "works" (objective function is minimized), but the 
thermal boundary condition is a bit useless right now. Also the material 
interpolation is pure SIMP by now and should be changed to i) thermal expansion
by explicit formula ii) conductivities, elast. modulus etc. pp. via 
Hashin-Shtrikman bound based interpolations. 

topopth_time.py

Transient heat conduction to minimize f^{T} u at every timestep and serves as
demonstration code for transient topology optimization.


homogenization.py

This is a conversion of the Matlab code from

Andreassen, Erik, and Casper Schousboe Andreasen. "How to determine composite material properties using numerical homogenization." Computational Materials Science 83 (2014): 488-495."

to Python- Currently only linear elasticity is available and still needs 
cleaning up.