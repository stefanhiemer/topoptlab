# Filter
Suppose we have $k$ filter $f$ that we apply in sequence: 
```{math}
\begin{aligned}
& \tilde{x}_1 = f_{1}(x)\\
& \tilde{x}_{2} = f_{2}(\tilde{x}_I) \\
& ... \\
& x_{p} = f_{k}(\tilde{x}_{k-1}) \\
\end{aligned}
```
$\tilde{x}$  are the intermediary variables and $x_{p}$ is the final densitity 
used to scale the material properties e. g. via the SIMP relationship 
```{math}
A(x_e) = A_{\min} + (A_0 - A_{\min}) x_p^k
```
Adjoint analysis returns to us $\frac{\partial C}{\partial x_p}$, but we are 
actually interested in the sensitivity with regard to the original design 
variables $\frac{\partial C}{\partial x}$ not the filtered counterpart. We 
remember the chain rule  
```{math}
\frac{\partial C}{\partial x} = \frac{\partial C}{\partial x_p} \frac{\partial x_p}{\partial x_{k-1}}
```