# Nonlinear FEM
## Nonlinear Systems
Nonlinear problems in computational modeling are typically formulated by discretizing a field $\phi$ 
into a vector $\boldsymbol{\Phi}_n$ and defining a nonlinear residual $\boldsymbol{r}\left(\boldsymbol{\Phi}_n\right)$
which for a viable solution returns
```{math}
\boldsymbol{r}\left(\boldsymbol{\Phi}_n\right) = \boldsymbol{0}
```
Let's consider an introductory example
```{math}
\boldsymbol{K}\left(\boldsymbol{\Phi}_n\right) \boldsymbol{\Phi}_n = \boldsymbol{f}
```
where $\boldsymbol{f}$ is a constant right hand side. There is no general 
method to solve the above problem in closed form, meaning iterative methods are 
necessary. 
### Newton method and Picard iteration for General Nonlinear Systems
Start from the previous iteration $i$ with values $\boldsymbol{\Phi}_{i,n}$ and 
find updated solution $\boldsymbol{\Phi}_{i+1,n}$ by approximating the nonlinear system with 
a Taylor expansion $\boldsymbol{r}$ around $\boldsymbol{\Phi}_{i,n}$ of first 
order and solve the local approximation
```{math}
\boldsymbol{K}\left(\boldsymbol{\Phi}_{i,n}\right) \boldsymbol{\Phi}_{i,n} - \boldsymbol{f} + \boldsymbol{J} \left(\boldsymbol{\Phi}_{i+1,n}-\boldsymbol{\Phi}_{i,n}\right) = \boldsymbol{0}
```
where $\boldsymbol{J}$ is the jacobian matrix of the nonlinear residual 
$\boldsymbol{r}$ with respect to $\boldsymbol{\Phi}_{n}$:
```{math}
\boldsymbol{J}_{k,l} = \frac{\partial \boldsymbol{r}_k}{\partial \boldsymbol{\Phi}_l}
```
$\boldsymbol{J}$ can be found in terms by using the chain rule 
```{math}
\boldsymbol{J} = \boldsymbol{K}\left(\boldsymbol{\Phi}_{i,n}\right) + \left(\nabla_{\boldsymbol{\Phi}_n} \boldsymbol{K}\left(\boldsymbol{\Phi}_{i,n}\right)\right) \cdot \boldsymbol{\Phi}_{i,n}
```
where $\nabla_{\boldsymbol{\Phi}_n} \boldsymbol{K}\left(\boldsymbol{\Phi}_{i,n}\right)$ 
is a third order tensor denoting entry-wise gradient of the matrix $\boldsymbol{K}$ 
with respect to $\boldsymbol{\Phi}_n$ which also acts on said vector via the single 
contraction $\cdot$. We will see later that in many numerical schemes this simplifies 
greatly. Write residual at current iteration 
```{math}
\boldsymbol{r}_i = \boldsymbol{K}\left(\boldsymbol{\Phi}_{i,n} \right)\boldsymbol{\Phi}_{i,n} - \boldsymbol{f} 
```
and rearrange to  
```{math}
\boldsymbol{J} \boldsymbol{\Phi}_{i+1,n} = -\boldsymbol{r}_i + \boldsymbol{J} \boldsymbol{\Phi}_{i,n}.
```
By repeatedly solving this linear problem, we will converge to the solution 
where $\boldsymbol{r} \approx \boldsymbol{0}$. In practice this form is rarely
chosen and further simplifications are done. We define the incremental update 
$\boldsymbol{\Phi}_{incr,n}$
```{math}
\boldsymbol{\Phi}_{incr,n} = \left(\boldsymbol{\Phi}_{i+1,n}-\boldsymbol{\Phi}_{i,n}\right)
```
after which the update of our solution changes to
```{math}
\boldsymbol{\Phi}_{i+1,n}=\boldsymbol{\Phi}_{i,n}+\boldsymbol{\Phi}_{incr,n}
```
and the linear problem simplifies to 
```{math}
\boldsymbol{J} \boldsymbol{\Phi}_{incr,n} = -\boldsymbol{r}_i
```
which gets rid of the matrix-vector-product $\boldsymbol{J} \boldsymbol{\Phi}_{i,n}$
and saves time. The procedure so far is the classical **Newton method**. An 
easier method often used in multiphysics is the **Picard iteration** where $\boldsymbol{J}$
is further simplified to 
```{math}
\boldsymbol{J} \approx \boldsymbol{K}\left(\boldsymbol{\Phi}_{i,n}\right).
```
It is often done as the element-wise gradient does not have to be calculated 
$\nabla_{\boldsymbol{\Phi}_n} \boldsymbol{K}\left(\boldsymbol{\Phi}_{i,n}\right)$ 
and it often preserves the symmetry of $\boldsymbol{K}$. It can basically be considered 
freezing $\boldsymbol{K}$ at the current intermediate $\boldsymbol{\Phi}_{i,n}$. 
The second part of the Jacobian is also often called the **Newton correction**
```{math}
\boldsymbol{K}^{corr} = \nabla_{\boldsymbol{\Phi}_n} \boldsymbol{K}\left(\boldsymbol{\Phi}_{i,n}\right) \cdot \boldsymbol{\Phi}_{i,n}
```
### Newton method and Picard iteration in Numerical Schemes like FEM
In many common numerical schemes, $\boldsymbol{K}$ as a function of the field $\phi$ which in
turn is a function of the discretized values $\boldsymbol{K}\left(\phi \left(\boldsymbol{\Phi}_n\right)\right)$.
Since in many common methods, $\phi\left(\boldsymbol{\Phi}_n\right)$ is linear with
respect to $\boldsymbol{\Phi}_n$. Thus by applying the chain rule to the Newton correction
(where the right hand side represents the already contracted form.)
```{math}
\left(\nabla_{\boldsymbol{\Phi}_n} \boldsymbol{K}\left(\boldsymbol{\Phi}_{i,n}\right) \right) \cdot \boldsymbol{\Phi}_{i,n} = \frac{\partial \boldsymbol{K}\left(\phi\left(\boldsymbol{\Phi}_{i,n}\right)\right)}{\partial \phi} \boldsymbol{\Phi}_{i,n} \left(\nabla_{\boldsymbol{\Phi}_n} \phi \right)^T
```
we may rewrite the Newton correction 
```{math}
\boldsymbol{K}^{corr} = \frac{\partial \boldsymbol{K}\left(\phi\left(\boldsymbol{\Phi}_{i,n}\right)\right)}{\partial \phi} \boldsymbol{\Phi}_{i,n} \left(\nabla_{\boldsymbol{\Phi}_n} \phi \right)^T
```
with $\left(\nabla_{\boldsymbol{\Phi}_n} \phi \right)^T$ given by the 
discretization technique at hand and 
$\frac{\partial \boldsymbol{K}\left(\phi\left(\boldsymbol{\Phi}_{i,n}\right)\right)}{\partial \phi}$
which e.g. in nonlinear heat conduction this would be the derivative
of the heat conductivity tensor with respect to temperature. This is just a 
material property that has to be known/measured. For FE we remember the 
interpolation with shape functions $\boldsymbol{N}$ from the previous chapter
```{math}
\phi = \boldsymbol{N}^T \boldsymbol{\Phi}_n 
``` 
therefore
```{math}
\left(\nabla_{\boldsymbol{\Phi}_n} \phi \right)^T = \boldsymbol{N}^T
```
Substituting this into Newton correction, we arrive at the final expression 
for FE:
```{math}
\boldsymbol{K}^{corr} = \frac{\partial \boldsymbol{K}\left(\phi\left(\boldsymbol{\Phi}_{i,n}\right)\right)}{\partial \phi} \boldsymbol{\Phi}_{i,n} \boldsymbol{N}^T
```
## Nonlin. Elasticity, stress and strain measures

In nonlinear mechanics, we distinguish between the (undeformed) reference 
configuration and the (deformed) current configuration. The motion 
$\varphi$ maps material points from the reference position 
$\boldsymbol{x} \in \Omega_0$ to the current position 
$\varphi(\boldsymbol{x}) \in \Omega$. Depending on which configuration the 
balance laws are expressed in, different stress measures and their
thermodynamic conjugate strain measures are used. 

| Stress measure | Work-conjugate strain measure | 
|----------------|-------------------------------|
| First Piola–Kirchhoff $ \boldsymbol{P} $ | deformation gradient $\boldsymbol{F}$ |
| Second Piola–Kirchhoff $ \boldsymbol{S} $ | Green–Lagrange strain $\boldsymbol{E}$ |
| Cauchy stress $ \boldsymbol{\sigma} $ |  |

The stress measures are related by 
```{math}
\boldsymbol{P} = \boldsymbol{F} \boldsymbol{S}
```
and 
```{math}
\boldsymbol{\sigma} = \det( \boldsymbol{F})^{-1} \boldsymbol{P} \boldsymbol{F}^T
```

**(a) Reference configuration with First Piola–Kirchhoff stress $\boldsymbol{P}$**
```{math}
\int_{\Omega_0} \nabla_0 \boldsymbol{w} : \boldsymbol{P} dV_0
= \int_{\Omega_0} \boldsymbol{w}\cdot \boldsymbol{b}_0 dV_0
+ \int_{\Gamma_{0,N}} \boldsymbol{w}\cdot \bar{\boldsymbol{t}}_0\, dA_0.
```
Here $\nabla_0$ is the gradient w.r.t. the reference coordinate $\boldsymbol{x}$, 
$\boldsymbol{b}_0$ is the body force per reference volume $V_0$, and 
$\bar{\boldsymbol{t}}_0$ is the nominal traction (per reference area).

**(b) Reference configuration with Second Piola–Kirchhoff stress $\boldsymbol{S}$**

We can rewrite the previous weak form also in terms of the 2nd Piola-Kirchhoff 
stress tensor $\boldsymbol{S}$ using $\boldsymbol{P}=\boldsymbol{F}\boldsymbol{S}$
```{math}
\int_{\Omega_0} (\nabla_0 \boldsymbol{w} \boldsymbol{F}) : \boldsymbol{S} dV_0
= \int_{\Omega_0} \boldsymbol{w}\cdot \boldsymbol{b}_0 dV_0 + 
\int_{\Gamma_{0,N}} \boldsymbol{w}\cdot \bar{\boldsymbol{t}}_0 dA_0.
```
Equivalently, using the Green–Lagrange strain 
$\boldsymbol{E}=\tfrac{1}{2}(\boldsymbol{F}^\mathrm{T}\boldsymbol{F}-\boldsymbol{I})$ 
and its variation
$\delta\boldsymbol{E}=\operatorname{sym}(\boldsymbol{F}^\mathrm{T}\nabla_0 \boldsymbol{w})$:
```{math}
\int_{\Omega_0} \boldsymbol{S} : \delta\boldsymbol{E} dV_0
= \int_{\Omega_0} \boldsymbol{w}\cdot \boldsymbol{b}_0 dV_0 + 
\int_{\Gamma_{0,N}} \boldsymbol{w}\cdot \bar{\boldsymbol{t}}_0 dA_0.
```
**(c) Current configuration with Cauchy stress $\boldsymbol{\sigma}$**

We can write weak form also in terms of the Cauchy stress tensor 
$\boldsymbol{\sigma}$ using $\boldsymbol{P}=\boldsymbol{F}\boldsymbol{S}$
```{math}
\int_{\Omega} \nabla \boldsymbol{w} : \boldsymbol{\sigma} dV
= \int_{\Omega} \boldsymbol{w}\cdot \boldsymbol{b} dV + 
\int_{\Gamma_{N}} \boldsymbol{w}\cdot \bar{\boldsymbol{t}} dA,
```
where $\nabla$ is the spatial gradient w.r.t. $\phi(x)$, $\boldsymbol{b}$ is body 
force per current volume $V$, and 
$\bar{\boldsymbol{t}}=\boldsymbol{\sigma}\boldsymbol{n}$ is the Cauchy 
traction on the Neumann boundary $\Gamma_N$ (with outward normal $\boldsymbol{n}$). 
On the Dirichlet boundary $\Gamma_D$, $\boldsymbol{w}=\boldsymbol{0}$.