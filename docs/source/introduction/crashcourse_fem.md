# A Crashcourse to FEM
Given the limitations in space and time, this is a brief introduction to the 
Finite Element Method (FEM). Interested readers are referred to the classic 
textbook by Belytschko and Fish {cite}`fish2007first`, the excellent lecture 
notes by Dennis Kochmann (https://mm.ethz.ch/education/lecture-notes.html), and 
the textbook by Peter Wriggers {cite}`wriggers2008nonlinear` for advanced 
topics. This section outlines the basic "recipe" to discretize a physical 
problem described by a partial differential equation (PDE) using FEM.

## From Strong to Weak Form
Most physical problems can be written as PDEs based on a conservation law or 
other physical insights. Classical examples are the **Poisson equation**

```{math}
-\nabla \cdot (\boldsymbol{K} \nabla \phi) = f
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
\nabla \cdot \boldsymbol{\sigma} + \boldsymbol{b} = \rho \ddot{\boldsymbol{u}}
```
representing conservation of linear momentum with the Cauchy stress tensor 
$\boldsymbol{\sigma}$, body force $\boldsymbol{b}$ and displacement 
$\boldsymbol{u}$ . Combined with a constitutive law (e.g. Hooke’s law 
$\boldsymbol{\sigma} = \boldsymbol{C} : \boldsymbol{\epsilon}$ with stiffness 
tensor $\boldsymbol{C}$ and engineering/incremental strain tensor
$\boldsymbol{\epsilon}=\frac{1}{2}\left(\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T \right)$), this gives the mechanical response of a solid. For 
small deformations, inserting Hooke’s law yields the **Navier–Lamé equations**
```{math}
\nabla \cdot \left( \boldsymbol{C} : \boldsymbol{\epsilon} \right)  + \boldsymbol{b} = \rho \ddot{\boldsymbol{u}},
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
$\Omega$ our (simulation) domain. For numerical treatment, it is often 
convenient to work with integrals as they can be approximated cheaply with 
low-order splines or polynomials. As additional step, we multiply the strong 
form by an arbitrary **weight function** $w$ (also called *test function*) and 
integrate over the domain $\Omega$
```{math}
\int_\Omega w \, (\mathcal{L}(u) - f) \, dV = 0.
```
This is the **weak form**. The weak form is fully equivalent to the strong 
form if $u$ and $w$ are sufficiently smooth. The reason is the arbitrariness of 
$w$: if the integral holds for all admissible $w$, then the residual 
$(L(u) − f)$ must vanish everywhere, which is the same in the strong form.
The weight function $w$ is also useful from a numerical perspective as we will 
later see it allows to use less smooth, i. e. cheaper approximations and it 
naturally introduces boundary terms (Neumann conditions).  

### Example 1: Poisson equation / Heat conduction / Diffusion
derive weak form for 3D heat eq.
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

| Stress measure | Work-conjugate strain measure | Configuration |
|----------------|-------------------------------|----------------|
| First Piola–Kirchhoff $ \boldsymbol{P} $ | $\boldsymbol{F}$ | Reference |
| Second Piola–Kirchhoff $ \boldsymbol{S} $ | Green–Lagrange strain $\boldsymbol{E}$ | Reference |
| Cauchy stress $ \boldsymbol{\sigma} $ | $\boldsymbol{\epsilon}$ | Current |

The stress measures are related by  
$\boldsymbol{P} = \boldsymbol{F} \boldsymbol{S}$ and 
$\boldsymbol{\sigma} = det(F)^{-1}\,P\,\boldsymbol{F}^T$.

**(a) Reference configuration with First Piola–Kirchhoff stress $\boldsymbol{P}$**
```{math}
\int_{\Omega_0} \nabla_0 \boldsymbol{w} : \boldsymbol{P}\, \mathrm{d}V_0
= \int_{\Omega_0} \boldsymbol{w}\cdot \boldsymbol{b}_0\, \mathrm{d}V_0
+ \int_{\Gamma_{0,N}} \boldsymbol{w}\cdot \bar{\boldsymbol{t}}_0\, \mathrm{d}A_0.
```
Here $\nabla_0$ is the gradient w.r.t. the reference coordinate $\boldsymbol{x}$, 
$\boldsymbol{b}_0$ is the body force per reference volume, and 
$\bar{\boldsymbol{t}}_0$ is the nominal traction (per reference area).

**(b) Reference configuration with Second Piola–Kirchhoff stress $\boldsymbol{S}$**

We can rewrite the previous weak form also in terms of the 2nd Piola-Kirchhoff 
stress tensor $\boldsymbol{S}$ using $\boldsymbol{P}=\boldsymbol{F}\boldsymbol{S}$
```{math}
\int_{\Omega_0} (\nabla_0 \boldsymbol{w}, \boldsymbol{F}) : \boldsymbol{S}, \mathrm{d}V_0
= \int_{\Omega_0} \boldsymbol{w}\cdot \boldsymbol{b}_0, \mathrm{d}V_0
    \int_{\Gamma_{0,t}} \boldsymbol{w}\cdot \bar{\boldsymbol{t}}_0, \mathrm{d}A_0.
```
Equivalently, using the Green–Lagrange strain 
$\boldsymbol{E}=\tfrac{1}{2}(\boldsymbol{F}^\mathrm{T}\boldsymbol{F}-\boldsymbol{I})$ 
and its variation
$\delta\boldsymbol{E}=\operatorname{sym}(\boldsymbol{F}^\mathrm{T}\nabla_0 \boldsymbol{w})$:
```{math}
\int_{\Omega_0} \boldsymbol{S} : \delta\boldsymbol{E}, \mathrm{d}V_0
= \int_{\Omega_0} \boldsymbol{w}\cdot \boldsymbol{b}_0, \mathrm{d}V_0
    \int_{\Gamma_{0,t}} \boldsymbol{w}\cdot \bar{\boldsymbol{t}}_0, \mathrm{d}A_0.
```
**(c) Current configuration with Cauchy stress $\boldsymbol{\sigma}$**

We can write weak form also in terms of the Cauchy stress tensor 
$\boldsymbol{sigma}$ using $\boldsymbol{P}=\boldsymbol{F}\boldsymbol{S}$
```{math}
\int_{\Omega} \nabla \boldsymbol{w} : \boldsymbol{\sigma}, \mathrm{d}v
= \int_{\Omega} \boldsymbol{w}\cdot \boldsymbol{b}, \mathrm{d}v

\int_{\Gamma_{N}} \boldsymbol{w}\cdot \bar{\boldsymbol{t}}, \mathrm{d}a,
```
where $\nabla$ is the spatial gradient w.r.t. $x$, $\boldsymbol{b}$ is body 
force per current volume, and 
$\bar{\boldsymbol{t}}=\boldsymbol{\sigma}, \boldsymbol{n}$ is the Cauchy 
traction on the Neumann boundary $\Gamma_N$ (with outward normal $\boldsymbol{n}$). On 
the Dirichlet boundary $\Gamma_D$, $\boldsymbol{w}=\boldsymbol{0}$.

## Discretization of Weak Form

### Shape Functions and Interpolation

### Assembly of Global (Non-)Linear Problem

### Numerical Integration 