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
- vectorization of the filter and stiffness matrix construction, 
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

topopt88m.py

Similar to topopt88.py just for compliant mechanism synthesis (maximized 
displacement). Originally taken from page 269 of the Bendsoe & Sigmund textbook 
and transcribed from the old 99 line code to the 88 line code. The results are 
close to the old 99 line code version but do not 100 % agree. I believe 
rounding errors in the old code to be the reason as the stiffness matrix there
seems to be asymmetric as well due to rounding errors. 

topopth_time.py

Transient heat conduction to minimize f^{T} u at every timestep and serves as
demonstration code for transient topology optimization.


homogenization.py

This is a conversion of the Matlab code from

Andreassen, Erik, and Casper Schousboe Andreasen. "How to determine composite material properties using numerical homogenization." Computational Materials Science 83 (2014): 488-495."

to Python- Currently only linear elasticity is available.