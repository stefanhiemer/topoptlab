# General Filter Implementation
Suppose we have $n$ filter $f$ that we apply in sequence: 
```{math}
\begin{aligned}
& \tilde{x}_1 = f_{1}(x)\\
& \tilde{x}_{2} = f_{2}(\tilde{x}_1) \\
& ... \\
& x_{p} = f_{n}(\tilde{x}_{n-1}) \\
\end{aligned}
```
$\tilde{x}$  are the intermediary variables and $x_{p}$ is the final densitity 
used to scale the material properties e. g. via the modified SIMP relationship 
```{math}
A(x_p) = A_{\min} + (A_0 - A_{\min}) x_p^k
```
Adjoint analysis returns to us $\frac{\partial C}{\partial x_p}$, but the 
design is parameterized in $x$, so we need in the 
sensitivity with regard to the original design variables $\frac{\partial C}{\partial x}$
to update the design. We recover $\frac{\partial C}{\partial x}$ via the chain rule  
```{math}
\frac{\partial C}{\partial x} = \frac{\partial C}{\partial x_p} \frac{\partial x_p}{\partial \tilde{x}_{n-1}} \frac{\partial \tilde{x}_{n-1}}{\partial \tilde{x}_{n-2}} ... \frac{\partial \tilde{x}_{i-1}}{\partial \tilde{x}_{i-2}} ... \frac{\partial \tilde{x}_{1}}{\partial x}.
```

Therefor in an abstract manner the general implementation for a sequence of 
$n$ filters in pseudo Python code for the filtering of the design variables is
```
# first intermediary field
xTilde[0,:] = filters[0].apply(x)
# intermediate steps
for i in range(1,n-1):
   xTilde[i,:] = filters[i-1].apply(x)
#
xPhys = filters[n].apply(xTilde[:,n-1])
```
and for the recovery of the sensitivity via the chain rule:
```
# calculate sensitivities with respect to physical densities 
sens = solve_adjoint_prob_obj(state_variables)
# apply chain rule for design variable sensitivities
for i in range(n-1,-1,-1):
    sens = filters[i].apply_dx(sens)
```
`apply` and `apply_dx` represent the filtering and the recovery via the chain rule
$\frac{\partial \tilde{x}_{i-1}}{\partial \tilde{x}_{i-2}}$.