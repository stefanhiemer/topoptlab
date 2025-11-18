---
title: 'topoptlab: An Open and Modular Framework for Benchmarking and Research in Topology Optimization'
tags:
  - topology optimization
  - finite element method
  - materials science
authors:
  - name: Stefan Hiemer
    orcid: 0000-0001-6562-4883
    affiliation: "1, 2" 
affiliations:
 - name: Center for Complexity and Biosystems, Department of Physics ”Aldo Pontremoli”, University of Milan, Via Celoria 16, 20133 Milano, Italy
   index: 1
 - name: CNR – Consiglio Nazionale delle Ricerche, Istituto di Chimica della Materia Condensata e di Tecnologie per l’Energia, Via R. Cozzi 53, 20125 Milano, Italy
   index: 2 
date: 25 August 2025
bibliography: paper.bib
---

# Summary

Topology optimization (TO) is becoming increasingly popular across physics, 
engineering, and materials science, as it provides a systematic way to discover 
efficient designs. Given a set of bounded design variables 
$x_{\min} \leq x_e \leq x_{\max}$ and a cost or objective function $C(x)$, 
subject to a discretized physical problem $K(x) u = f$ as well as a set of 
$m$ equality and $n$ inequality constraints, the general TO problem can be 
formulated as  

$$
\begin{aligned}
& \min_{x} \; C(x) \\
& \text{subject to:} \\
& \quad K(x) u = f, \\
& \quad g_i(x) = 0, \quad i = 1, \ldots, m, \\
& \quad h_j(x) \leq 0, \quad j = 1, \ldots, n, \\
& \quad x_{\min} \leq x_e \leq x_{\max}, \quad \forall e \in \text{elements}.
\end{aligned}
$$  

The common TO choice of design representation is density-based material 
interpolation scheme, where the abstract design variables $x_e$ become relative 
element-wise densities which scale the physical properties $A$ of each element 
via simple relationships as in the popular modified SIMP approach [@sigmund2007morphology]:

$A(x_e) = A_{min} + (A_{0}-A_{min})x_{e}^{k}$ with $0\leq x_e\leq 1$

where $k$ is a penalization factor ensuring densities close to 0/1, $A_0$ 
the property of the full material and $A_{min}$ a small value to prevent 
singularities in the physical problem. The final design then emerges from the 
optimal density distribution.  

*Topoptlab* is a modular and transparent framework for research and 
benchmarking in topology optimization with a focus on clarity, reproducibility, 
and accessibility as a tool for both research and advanced education.  

# Statement of need


In TO, it has become longstanding practice to demonstrate new methods with short 
Matlab scripts[@sigmund200199; @andreassen2011efficient; @ferrari2020new; @wang2021comprehensive]. 
While these codes have played an important role in the spread 
and development of ideas, they also come with notable limitations: First, Matlab 
requires a commercial license, and is only partially compatible to its free 
alternative Octave. Second, extension and combination of state-of-the-art 
methods demands combining multiple monolithic scripts, some of 
which are outdated or mutually incompatible. Third, while modern finite element 
frameworks such as **FEniCS**[@alnaes2014unified; @scroggs2022construction; @scroggs2022basix; @baratta2023dolfinx], 
**deal.II** [@arndt2021deal], and **ElmerFEM**[@malinen2013elmer] provide powerful 
high-performance environments, their abstraction layers tend to complicate 
access to low-level implementation details which is necessary for research in 
TO. Examples of standard TO tasks with the need to access low-level data 
structures are the update of the element stiffness matrices $K_{e}(x)$ 
based on the design variables $x$, calculation of the sensitivity of the element 
matrices with respect to the design variables $\frac{\partial K_e}{\partial x}$ 
and access to the global stiffness matrix for solving the adjoint problem to 
derive the gradients. Also common use cases in TO allow shortcuts such as 
regular meshes as ideally the geometry emerges during the optimization process 
or partial negligence of close to empty elements as preconditioning. 

*Topoptlab* was developed to address these challenges by providing a stable 
and extensible environment tailored to the needs of the TO community. It serves 
as a library for writing complete problems from scratch in spirit of the 
already conventional Matlab scripts, and offers a high-level driver 
routine (`topology_optimization`) in which users can exchange components 
(filter, objective function, etc.) by passing custom callables or objects as 
arguments. It may also serve as a reference implementation which can be used as 
test case for existing HPC codes that want to incorporate TO in their software.
An example demonstrating the topology_optimization routine, as well as links to 
from-scratch implementations, are available in the [`README.md`](https://github.com/stefanhiemer/topoptlab/blob/main/README.md) 
while tutorials, derivations and explanations of the background of TO are in 
the documentation. A more extensive list of examples can be found in the 
[`examples`](https://github.com/stefanhiemer/topoptlab/tree/main/examples) 
section of the repository.

*Topoptlab* offers the components needed for TO such as different material 
interpolation schemes (SIMP, RAMP, and bound-based interpolation), filters for 
regularization [@sigmund1997design; @bruns2001topology], 
projections [@guest2004achieving, @sigmund2007morphology; @xu2010volume] and 
manufacturability [@langelaar2017additive], and finite element implementations 
for different physical problems (linear elasticity, heat conduction, etc.) with 
both standard numerical integration and analytically integrated elements 
generated through *Symfem*[@scroggs2021symfem]. Constrained optimization is 
supported through the Method of Moving Asymptotes (MMA) [@svanberg1987], the 
Globally Convergent Method of Moving Asymptotes (GCMMA) [@svanberg2002class] 
as implemented in [@deetman2024gcmma] as well as the Optimality Criteria 
[@andreassen2011efficient ;@bendsoe2003topology], while the solution of 
the system of equations is done via routines offered by *scipy* [@2020SciPy-NMeth], 
*cvxopt* [@andersen2020cvxopt], and also custom implementations of 
preconditioners like algebraic multigrid or block-preconditioners. Finally, 
*Topoptlab* offers a number of introductory articles with comments on 
implementation and contains monolithic scripts that serve as teaching tools as 
well as an archive for Python translations of important Matlab teaching codes 
(e.g. [@andreassen2011efficient; @andreassen2014determine]). 

# Acknowledgments

We acknowledge the support by the Humboldt foundation through the Feodor-Lynen 
fellowship. We acknowledge partial support from the ARCHIBIOFOAM project which 
received funding from the European Union’s Horizon Europe research and 
innovation programme under grant agreement No 101161052. Views and opinions 
expressed are however those of the author(s) only and do not necessarily 
reflect those of the European Union or European Innovation Council and SMEs 
Executive Agency (EISMEA). Neither the European Union nor the granting 
authority can be held responsible for them.

Special thanks are extended to Christof Schulze for his valuable suggestions on 
code improvements, to Stefano Zapperi for hosting at the University of Milan 
and for his continuous support, and to Peter Råback for sharing his vast 
knowledge on FEM related implementation details.

# References
