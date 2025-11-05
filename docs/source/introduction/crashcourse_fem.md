# A Crashcourse to FEM
Given the limitations in space and time, this is a brief introduction to the 
Finite Element Method (FEM). Interested readers are referred to the classic 
textbook by Belytschko and Fish {cite}`fish2007first`, the excellent lecture 
notes by Dennis Kochmann (https://mm.ethz.ch/education/lecture-notes.html), and 
the textbook by Peter Wriggers {cite}`wriggers2008nonlinear` for advanced 
topics. This section outlines the basic "recipe" to discretize a physical 
problem described by a partial differential equation (PDE) using FEM which is 
commonly done the following the steps: 

- derive weak form 
- if PDE nonlinear: linearize
- approximate weak form 

In this section we will restrict ourselves to examples of linear problems and
only hint or show final results for nonlinear problems.

## From Strong to Weak Form
Most physical problems can be written as PDEs based on a conservation law or 
other physical insights. Classical examples are the **Poisson equation**
```{math}
\nabla \cdot (\boldsymbol{K} \nabla \phi) = f
```
governing heat conduction or diffusion with scalar variable $\phi$ 
(temperature, concentration, etc.) and property tensor $\boldsymbol{K}$, 
the continuity equation
```{math}
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \boldsymbol{v}) = 0
```
describing conservation of mass (density $\rho$, time $t$ and velocity field 
$\boldsymbol{v}$ and the **Cauchy momentum balance**
```{math}
\rho \ddot{\boldsymbol{u}} = \nabla \cdot \boldsymbol{\sigma} + \boldsymbol{b}
```
representing conservation of linear momentum with the Cauchy stress tensor 
$\boldsymbol{\sigma}$, body force $\boldsymbol{b}$ and displacement 
$\boldsymbol{u}$ . Combined with a constitutive law (e.g. Hooke’s law 
$\boldsymbol{\sigma} = \boldsymbol{C} : \boldsymbol{\epsilon}$ with stiffness 
tensor $\boldsymbol{C}$ and engineering/incremental strain tensor
$\boldsymbol{\epsilon}=\frac{1}{2}\left(\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T \right)$), 
this gives the mechanical response of a solid. For small deformations, 
inserting Hooke’s law yields the **Navier–Lamé equations**
```{math}
\rho \ddot{\boldsymbol{u}} = \nabla \cdot \left( \boldsymbol{C} : \boldsymbol{\epsilon} \right)  + \boldsymbol{b},
```
and for Newtonian fluids the **Navier–Stokes equations**
```{math}
\rho \left( \frac{\partial \boldsymbol{v}}{\partial t} + \boldsymbol{v}\cdot\nabla\boldsymbol{v} \right)
= -\nabla p + \mu \nabla^2 \boldsymbol{v} + \rho \boldsymbol{b}.
```
All of these equations are expressed in terms of their derivatives i. e. 
**differential form** which is also called the **strong form**. We introduce a 
generic PDE for brevity as
```{math}
\mathcal{L}(u) = f \quad \text{in } \Omega,
```
where the linear differential operator $\mathcal{L}$ represents all terms with 
derivatives involving the unknown field $u$ (e. g. for linear elasticity 
$\nabla \cdot \left( \boldsymbol{C} : \boldsymbol{\epsilon} \right)$), 
$f$ represents any inhomogeneous terms often called source/sink terms and 
$\Omega$ the simulation domain. We multiply the strong form by a 
**weight function** $w$ and integrate over the domain $\Omega$, so we end up 
with
```{math}
\int_\Omega w \, (\mathcal{L}(u) - f) \, dV = 0.
```
This is the **weak form** which is fully equivalent to the strong 
form if $u$ and $w$ satisfy some conditions. Among these conditions are
$w!=0$ and that the weak form must hold for abitrary $w$ as then the residual 
$r$
```{math}
r=\int_\Omega w \, (\mathcal{L}(u) - f) \, dV 
```
vanishes everywhere, which corresponds to the solution of the strong form. 
The weight function $w$ is also useful from a numerical perspective as we will 
later see, it allows to use less smooth, i. e. cheaper approximations and it 
naturally introduces boundary terms for Neumann boundary conditions. 
Furthermore, from a numerical perspective, it is often convenient to work with 
integrals as they can be approximated cheaply with low-order splines or 
polynomials.

### Example 1: Poisson equation / Heat conduction / Diffusion
We write down weak form via standard procedure 
i) multiply by $w$ ii) integrate over domain $\Omega$:
```{math}
\int_\Omega w \nabla \cdot (\boldsymbol{K} \nabla \phi) dV = \int_\Omega w f dV
```
Technically one can stop now as this is correct, but we would like to reduce 
the highest order derivative as much as we can as in FEM this means we can 
approximate it cheaper. We split weak form into left hand side and right hand 
side. Now consider the "problematic" left hand side
```{math}
\int_\Omega w \nabla \cdot (\boldsymbol{K} \nabla \phi) dV
```
and try to simplify it. We write down the chain rule for a general vector $\boldsymbol{v}$ 
and the scalar function $w$
```{math}
\nabla \cdot (w\boldsymbol{v}) = w\nabla \cdot \boldsymbol{v} + \nabla w \cdot \boldsymbol{v}
```
which we rewrite to 
```{math}
w\nabla \boldsymbol{v} = \nabla w \cdot \boldsymbol{v} - \nabla \cdot (w\boldsymbol{v})
```
We then insert $\boldsymbol{v}=\nabla \phi$ and rewrite the left hand side to
```{math}
\int_\Omega w \nabla \cdot (\boldsymbol{K} \nabla \phi) dV = \int_\Omega \nabla w \cdot \boldsymbol{K} \nabla \phi dV - \int_\Omega \nabla \cdot \left(w \boldsymbol{K} \nabla \phi\right) dV
```
If we inspect this closer, we recognize that the second term on the right hand 
side is the volume integral of the divergence of the flow $\boldsymbol{K} \nabla \phi$ 
scaled by $w$. 
divergence theorem
```{math}
\int_{\Omega} \nabla \cdot \boldsymbol{v} dV = \int_{\Gamma} \boldsymbol{v} \cdot \boldsymbol{n} dA 
```
which we can use to simplify the second part 
```{math}
\int_\Omega \nabla \cdot \left(w \boldsymbol{K} \nabla \phi\right) dV = \int_{\Gamma} \left(w \boldsymbol{K} \nabla \phi\right) \cdot \boldsymbol{n} dA 
```
### Example 2: Linear Elasticity
derive weak form for 3D lin. elast.
### Common Weak Forms: Nonlin. Elasticity and stress measures

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

## Discretization of Weak Form

### Shape Functions and Interpolation

### Assembly of Global (Non-)Linear Problem

### Numerical Integration 