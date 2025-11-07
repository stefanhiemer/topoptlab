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

### Example: Weak form of  Poisson equation 
In this paragraph we will write down the weak form for the Poisson equation
which guides an abundant number of physical phenomena like temperature 
conduction, diffusion, gravity, electrostatics, etc. pp. For sake of brevity, 
we focus on time-independent (stationary) heat conduction. We start with the 
weak form via standard procedure i) multiply by $w$ ii) integrate over domain 
$\Omega$:
```{math}
\int_\Omega w \nabla \cdot (\boldsymbol{K} \nabla \phi) dV = \int_\Omega w f dV
```
Technically one can stop now as this is a correct weak from, but we want to 
reduce the highest order derivative as much as we can as in FEM this means we 
can approximate it cheaper. We split weak form into left hand side and right 
hand side and consider the "problematic" left hand side
```{math}
\int_\Omega w \nabla \cdot (\boldsymbol{K} \nabla \phi) dV
```
to simplify it further. We remember the chain rule for a general vector $\boldsymbol{v}$ 
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
side is the volume integral of the divergence of the flow 
$\boldsymbol{K} \nabla \phi$ scaled by $w$. In simple words, this integral 
describes how much of $w \boldsymbol{K} \nabla \phi $ is being produced or lost 
within the volume. Instead of measuring what happens inside, we can 
equivalently measure how much of $w \boldsymbol{K} \nabla \phi $ flows in or 
out through the surface enclosing it which is what the divergence theorem 
expresses
```{math}
\int_{\Omega} \nabla \cdot \boldsymbol{v} dV = \int_{\Gamma} \boldsymbol{v} \cdot \boldsymbol{n} dA 
```
which we can use to simplify the second part to
```{math}
\int_\Omega \nabla \cdot \left(w \boldsymbol{K} \nabla \phi\right) dV = \int_{\Gamma_{N}} \left(w \boldsymbol{K} \nabla \phi\right) \cdot \boldsymbol{n} dA 
```
This boundary flux conveniently represent the von Neumann boundary conditions, 
so they appear naturally in the weak form which is why they are often referred 
to as natural boundary conditions in FEM terminology. We write down the full 
weak form in its simplified form
```{math}
\int_\Omega \nabla w \cdot \boldsymbol{K} \nabla \phi dV - \int_{\Gamma_{N}} \left(w \boldsymbol{K} \nabla \phi\right) \cdot \boldsymbol{n} dA   = \int_\Omega w f dV.
```
Since we want to solve for $phi$ while both the von Neumann boundary conditions
and the function $f$ are given, we move the von Neumann terms to the right hand
side such that we have cleanly divided the weak form in the terms we seek to 
solve/invert and the terms that are part of the problem statement:
```{math}
\int_\Omega \nabla w \cdot \boldsymbol{K} \nabla \phi dV   = \int_\Omega w f dV + \int_{\Gamma_{N}} \left(w \boldsymbol{K} \nabla \phi\right) \cdot \boldsymbol{n} dA.
```

## Discretization of Weak Form

### Shape Functions and Interpolation
- approximate the field variable $u$ and $w$ by low order splines 
- in standard Galerkin FE $u,w$ described same function form 
- linear combination of basis functions 
```{math}
u(x) = \sum_{i=1} N\left(x;x_i\right)u_{i}
```
```{math}
w(x) = \sum_{i=1} N\left(x;x_i\right)w_{i}
```
rewrite to vector format
```{math}
u(x) = \boldsymbol{N}^T \boldsymbol{u}_n
```
```{math}
w(x) = \boldsymbol{N}^T \boldsymbol{w}_n
```



### Assembly of Global Linear Problem

### Numerical Integration 

### Isoparametric Map