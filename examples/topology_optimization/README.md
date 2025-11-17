These directories contain application cases of the topoptlab for topology 
optimization. 

### `compliance_minimization`

Maximization of stiffness or equivalent problems (e. g. cooling maximization 
for heat conduction) typically combined with a volume inequality constraint.
This is the standard problem of TO and should be the starting point when 
starting TO or testing new methods. However this problem is known to be 
relatively simple, so this is just the first "simple" step.

#### `compliant_mechanism`

Maximization or control of the state variable (i. e. displacement in mechanics)
to produce compliant mechanisms. This is a much harder problem compared to 
the compliance minimization and also more expensive, as the adjoint problem 
has to be solved instead of relying on analytical solutions.

### `uniform_temp_response`

Maximization of displacement under uniform temperature expansion. Will be 
changed to a one sided coupled problem where first the heat conduction and then
the mechanical problem is solved.  
