# Calculating Gradients/Sensitivities for Optimization

In adjoint analysis we set out to calculate design sensitivities for optimizing
the objective function $C$ and for fulfilling the constraints $\boldsymbol{g},\boldsymbol{h}$.
As the procedure for $C$ and the constraints is exactly the same, the 
calculation will only be demonstrated for the objective function.

## Notation
Before starting a short note on notation: $\frac{d y}{d x}$ denotes the total 
derivative and $\frac{\partial y}{\partial x}$ the partial derivative which 
only accounts for the **direct (explicit)** dependence of $y$ on $x$, treating 
all other variables as fixed/constants.

Take the function $g(x)=x^4$ and rewrite it as $q(u)=u^2$ with $u=x^2$, then
the partial derivative is 
```{math}
\frac{\partial q}{\partial x}=0 
``` 
and the total derivative is
```{math}
\begin{align}
\frac{d q}{d x} &=\frac{\partial q}{\partial x} + \frac{\partial q}{\partial u} \cdot \frac{\partial u}{\partial x} \\
\frac{d q}{d x} &= 0 + 2u \cdot 2x = 4x^3
\end{align}
``` 
(direct-method)=
## Direct Method
We differentiate $C$ with regards to a design variable $x$ which 
yields 
```{math}
\frac{d C}{d x} = \frac{\partial C}{\partial x} + \nabla_{\boldsymbol{u}}C^T \frac{\partial \boldsymbol{u}}{\partial x}
```
While $\frac{\partial C}{\partial x}$ and $\nabla_{\boldsymbol{u}}C$ are easy to 
evaluate, $\frac{\partial \boldsymbol{u}}{\partial x}$ is a problem: remembering the physical
problem
```{math}
\boldsymbol{K}\boldsymbol{u} = \boldsymbol{f},
```
its solution can be stated as 
```{math}
\boldsymbol{u} = \boldsymbol{K}^{-1}\boldsymbol{f},
``` 
where $\boldsymbol{K}^{-1}$ is the inverse matrix of $\boldsymbol{K}$. 
We assume for the moment the right hand side to be independent of $x$, therefor
we can rewrite
```{math}
\frac{\partial \boldsymbol{u}}{\partial x} = \frac{\partial \boldsymbol{K}^{-1}}{\partial x}\boldsymbol{f},
```
and after looking up the derivative of a matrix with respect to a scalar 
(https://en.wikipedia.org/wiki/Matrix_calculus#Matrix-by-scalar_identities) 
this becomes
```{math}
\frac{\partial \boldsymbol{u}}{\partial x} = -\boldsymbol{K}^{-1} \frac{\partial \boldsymbol{K}}{\partial x}\boldsymbol{K}^{-1} \boldsymbol{f},
```
This solution for all practical purposes is impractical as $\boldsymbol{K}$ is 
a very large which makes the matrix products and inversion computationally too 
expensive.

(adjoint-analysis)=
## Adjoint Analysis

In adjoint analysis, one rewrites the objective function as 
```{math}
\tilde{C} = C + \boldsymbol{\lambda}^T \left( \boldsymbol{K}\boldsymbol{u} - \boldsymbol{f} \right)
```
where $\boldsymbol{\lambda}$ is an arbitrary vector which we call the adjoint 
vector or in general adjoint variables. It is arbitrary as we have written 
$\boldsymbol{\lambda } \cdot \boldsymbol{0}$, therefor the values of $\boldsymbol{\lambda}$ 
are arbitrary. After differentiation
```{math}
\frac{d \tilde{C}}{d x} = \frac{\partial C}{\partial x} + \nabla_{\boldsymbol{u}}C \frac{\partial \boldsymbol{u}}{\partial x} +  \boldsymbol{\lambda}^T \left( \frac{\partial \boldsymbol{K}}{\partial x} \boldsymbol{u} + \boldsymbol{K} \frac{\partial \boldsymbol{u}}{\partial x}  - \frac{\partial \boldsymbol{f}}{\partial x} \right)
```
we again assume for sake of clarity that the right hand side of the phys. 
problem $\boldsymbol{f}$ is independent of $x$, therefor $\frac{\partial \boldsymbol{f}}{\partial x}=\boldsymbol{0}$
and we re-group the terms: 
```{math}
\frac{d \tilde{C}}{d x} = \frac{\partial C}{\partial x} + \left( \nabla_{\boldsymbol{u}}C^T + \boldsymbol{\lambda}^T \boldsymbol{K}\right)\frac{\partial \boldsymbol{u}}{\partial x} +  \boldsymbol{\lambda}^T \frac{\partial \boldsymbol{K}}{\partial x} \boldsymbol{u}
```
We now notice if 
```{math}
\nabla_{\boldsymbol{u}}C^T + \boldsymbol{\lambda}^T \boldsymbol{K} = \boldsymbol{0}
```
the troublesome derivative $\frac{\partial \boldsymbol{u}}{\partial x}$ drops 
out of the expression for $\frac{d \tilde{C}}{d x}$. As $\boldsymbol{\lambda}$ 
is arbitrary, after re-arranging the terms one can state the adjoint problem
```{math}
\boldsymbol{K}^T \boldsymbol{\lambda} = -\nabla_{\boldsymbol{u}}C
```
which yields the final expression for the sensitivities as 
```{math}
\frac{d \tilde{C}}{d x} = \frac{\partial C}{\partial x} + \boldsymbol{\lambda}^T \frac{\partial \boldsymbol{K}}{\partial x} \boldsymbol{u}.
```
In adjoint analysis the calculation of gradients therefor amounts to just 
solving a linear problem which is much cheaper as compared to the direct method.