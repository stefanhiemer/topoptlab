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
$\Omega$ the simulation domain. In terms of boundary conditions, we only 
concern ourselves in this article with Dirichlet boundary conditions that fix
the state variable to specific values on a part of the boundary $\Gamma_D$
```{math}
u = u_{D} \quad \text{on } \Gamma_{D}
```
and von Neumann boundary conditions that fixes the value first derivative(s) to
on a part of the boundary $\Gamma_N$ 
 
```{math}
\nabla u = \boldsymbol{\partial u}_{N} \quad \text{on } \Gamma_{N}
```
In solid mechanics, Dirichlet boundary conditions correspond to displacement 
boundary conditions while in heat conduction or diffusion they correspond to 
temperature/concentration boundary condtions. Von Neumann boundary conditions 
are often formulated differently, as in real cases rarely the gradient of the 
state variable $u$ is available, but instead the **flux** is measured, so the
above equation is simply rescaled by some material constants 
```{math}
\boldsymbol{K} \nabla u = q_{N} \quad \text{on } \Gamma_{N}
```
which however does not change the nature of the boundary condition: it is still
a boundary condition in terms of the first order derivatives.

To arrive at the **weak form**, we multiply the strong form and its boundary 
conditions by a **weight function** $w$ and integrate over the domain $\Omega$, 
so we end up with
```{math}
\int_\Omega w \, (\mathcal{L}(u) - f) \, dV = 0. \\ 
\int_{\Gamma_{N}} w ( \boldsymbol{K} \nabla u)^T \boldsymbol{n} dA  = u_{N} 
```
The weak form is fully equivalent to the strong form only if $u$ and $w$ 
satisfy some conditions. Among these conditions are $w!=0$ and that the weak 
form must hold for abitrary (!) $w$ as then the residual $r$
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

### Example: Weak form of Poisson equation 
In this paragraph we will write down the weak form for the Poisson equation
which guides an abundant number of physical phenomena like temperature 
conduction, diffusion, gravity, electrostatics, etc. pp. For sake of brevity, 
we focus on time-independent (stationary) heat conduction. We start with the 
weak form via standard procedure i) multiply PDE and boundary conditions by $w$ 
ii) integrate over domain $\Omega$:
```{math}
\int_\Omega w \nabla \cdot (\boldsymbol{K} \nabla \phi) dV = \int_\Omega w f dV \\
\int_{\Gamma_{N}} w\left(\boldsymbol{K} \nabla \phi\right)^T \boldsymbol{n} dA
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
\int_\Omega w \nabla \cdot (\boldsymbol{K} \nabla \phi) dV = \int_\Omega \nabla w \cdot \boldsymbol{K} \nabla \phi dV - \int_\Omega \nabla \cdot w\left(\boldsymbol{K} \nabla \phi\right) dV
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
\int_\Omega \nabla \cdot w\left(\boldsymbol{K} \nabla \phi\right) dV = \int_{\Gamma_{N}} w\left(\boldsymbol{K} \nabla \phi\right)^T \boldsymbol{n} dA 
```
This boundary flux conveniently represent the von Neumann boundary conditions, 
so they appear naturally in the weak form which is why they are often referred 
to as natural boundary conditions in FEM terminology. We write down the full 
weak form in its simplified form
```{math}
\int_\Omega \nabla w \cdot \boldsymbol{K} \nabla \phi dV - \int_{\Gamma_{N}} w\left(\boldsymbol{K} \nabla \phi\right)^T \boldsymbol{n} dA   = \int_\Omega w f dV.
```
Since we want to solve for $phi$ while both the von Neumann boundary conditions
and the function $f$ are given, we move the von Neumann terms to the right hand
side such that we have cleanly divided the weak form in the terms we seek to 
solve/invert and the terms that are part of the problem statement, i. e. the 
boundary conditions:
```{math}
\int_\Omega \nabla w \cdot \boldsymbol{K} \nabla \phi dV = \int_\Omega w f dV + \int_{\Gamma_{N}} w\left(\boldsymbol{K} \nabla \phi\right)^T \boldsymbol{n} dA.
```

## Discretization of Weak Form
Since the weak form is an integral, we can now slowly see how elements as 
subdivisions of the simulation domain emerge: integrals are additive, meaning 
an integral over interval $a$ to $c$ can be split into sub-intervals $a$ to 
$b$ and $b$ to $c$
```{math}
\int_a^c ... dx = \int_a^b ... dx + \int_b^c ... dx
```
so we can re-write the weak form to 
```{math}
\sum_{e=1}^{N_{e}}\int_{\Omega_e} \left(\nabla w\right)^T \boldsymbol{K} \nabla \phi dV_{\Omega_e} = \sum_{e=1}^{N_{e}} \int_{\Omega_e} w f dV_e + \int_{\Gamma_{N,e}} w\left(\boldsymbol{K} \nabla \phi\right)^T \boldsymbol{n} dA_e.
```
where the index $e$ is the element index. So in other words an element is just 
a sub-interval/area/volume of the entire domain. This step is purely geometric 
and does not yet introduce any approximation as it just rewrites the weak form 
as a sum of element-wise contributions that via simple summation assemble the 
global problem. We now move on, how to approximate these integrals to arrive at a continuous, smooth solution.
 
### Shape Functions and Interpolation
After splitting the domain into elements, a few things need to be kept in mind: 
i) we need to approximate the unknown field $u$ inside each element to solve 
the local element integrals. ii) the values of $u$ at the element borders 
should match for the smoothness and continuity required by the PDE. The natural 
choice for these requirements are splines as they only require nodal values at 
the border between elements for interpolation, due to their local polynomial 
nature are easy to integrate (even high school students could do it) and offer 
flexibel smoothness/continuity via the choice of order. As a rule of thumb, it 
is preferable to choose the lowest order of spline that is sufficient to solve
the PDE at hand. Having chosen splines an interpolation, we can now express 
both the unknown field $u$ and the test function $w$ in terms of these local basis 
functions. This turns the continuous fields into finite sets of nodal values, 
which can then be used directly in the weak form on each element.

We restrict ourselves to the standard Galerkin finite element method where both 
$w$ and $u$ are approximated by the same functions, i. e. are interpolated in 
the same function space. In general, $u$ and $w$ inside element $e$ are 
approximated as 
```{math}
u_e(x) = \sum_{j=1}^{n} N_{j}\left(x;x_j\right)u_{j}
```
```{math}
w_e(x) = \sum_{j=1}^{n} N_{j}\left(x;x_j\right)w_{j}
```
or more commonly in vector format as
```{math}
u_e(x) = \boldsymbol{N}^T \boldsymbol{u}_n
```
```{math}
w_e(x) = \boldsymbol{N}^T \boldsymbol{w}_n.
```
where $n$ is the number of nodes in the element, $u_{i},w_{i}$ are the value 
of $u,w$ at the nodes and $N_{i}$ the **shape function** associated with node 
$i$. Generally these shape functions constitute of polynomial functions in 
terms of space and take the value $1$ at $x_i$ and the value $0$ at nodes 
$i \neq j$. The latter is referred as the **delta-property** of shape 
functions which constructs a smooth interpolation within the element and that
connects smoothly to the other elements thus fulfilling our requirement for
a smooth solution. 

We now take the weak form for a single element  
```{math}
\int_{\Omega_e} \left(\nabla w_e\right)^T \boldsymbol{K} \nabla \phi_e dV_{\Omega_e} = \int_{\Omega_e} w_e f_e dV_e + \int_{\Gamma_{N,e}} \left(w_e \boldsymbol{K} \nabla \phi_e\right)^T \boldsymbol{n} dA_e.
```
and insert our approximation 
```{math}
\boldsymbol{w}_{n}^T \int_{\Omega_e} \left(\nabla \boldsymbol{N} \right)^T \boldsymbol{K} \nabla \boldsymbol{N}^T  dV_{\Omega_i} \boldsymbol{\phi}_n = \boldsymbol{w}_e^T \left[ \int_{\Omega_e} \boldsymbol{N} f dV_e + \int_{\Gamma_{N,e}} \boldsymbol{N} \left( \boldsymbol{K} \nabla \phi\right)^T \boldsymbol{n} dA_e \right].
```
After some careful consideration, we recognize the left hand side as a linear
system 
```{math}
\boldsymbol{w}_n^T \int_{\Omega_e} \boldsymbol{A} \boldsymbol{\phi}_n  = \boldsymbol{w}_n^T \int_{\Omega_e}  \left(\nabla \boldsymbol{N} \right)^T \boldsymbol{K} \nabla \boldsymbol{N}^T  dV_{\Omega_i} \boldsymbol{\phi}_n.
```
$\boldsymbol{w}_n$ must not matter as the requirement for the weak form
to deliver a solution to the original PDE was that $w$ must be arbitrary. This
reduces the expression further to
```{math}
\boldsymbol{A} \boldsymbol{\phi}_n  = \int_{\Omega_e} \left(\nabla \boldsymbol{N} \right)^T \boldsymbol{K} \nabla \boldsymbol{N}^T dV_{\Omega_i} \boldsymbol{\phi}_n
```


### Assembly of Global Linear Problem

```{math}
\boldsymbol{K}= \sum_{e=1}^{N_e} \boldsymbol{K}_e
```

### Numerical Integration 

### Isoparametric Map