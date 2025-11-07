# Nonlinear FEM

## Newton iteration

## Picard iteration

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