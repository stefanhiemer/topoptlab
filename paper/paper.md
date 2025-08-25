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

Topology optimization (TO) is becoming increasingly popular across physics, engineering, and materials science, as it provides a systematic way to discover efficient designs. Given a set of bounded design variables $\rho_{\min} \leq \rho_e \leq \rho_{\max}$ and a cost or objective function $C(\rho)$, subject to a discretized physical problem $K(\rho) u = f$ as well as a set of $m$ equality and $n$ inequality constraints, the general TO problem can be formulated as  

$$
\begin{aligned}
& \min_{\rho} \; C(\rho) \\
& \text{subject to:} \\
& \quad K(\rho) u = f, \\
& \quad g_i(\rho) = 0, \quad i = 1, \ldots, m, \\
& \quad h_j(\rho) \leq 0, \quad j = 1, \ldots, n, \\
& \quad \rho_{\min} \leq \rho_e \leq \rho_{\max}, \quad \forall e \in \text{elements}.
\end{aligned}
$$  

The common TO choice of design representation is material interpolation scheme, where the abstract design variables $\rho$ become relative element-wise densities which scale the physical properties $A$ of each element via simple relationships as in the popular SIMP approach:

$A(\rho_e) = A_{min} + (A_{0}-A_min)*rho_{e}^{k}$ with $0\leq \rho_e\leq 1$

where $k$ is a penalization factor ensuring densities close to 0/1 and $A_0$ the property of the full material. The final design then emerges from the optimal density distribution.  

*Topoptlab* is a modular and transparent framework for research and benchmarking in topology optimization with a focus on clarity, reproducibility, and accessibility as a tool for both research and advanced education.  

# Statement of need


In TO, it has become longstanding practice to demonstrate new methods with short 
Matlab scripts. While these codes have played an important role in the spread 
and development of ideas, they also come with notable limitations: First, Matlab 
requires a commercial license, and is only partially compatible to its free 
alternative, Octave. Second, extension and combination of state-of-the-art 
methods demands puzzling together multiple independent scripts, some of 
which are outdated or mutually incompatible. Third, while modern finite element 
frameworks such as **FEniCS**, **deal.II**, and **ElmerFEM** provide powerful 
high-performance environments, their abstraction layers tend to complicate 
access to low-level implementation details which is necessary for research in 
TO. Also common use cases in TO allow shortcuts such as regular meshes as 
ideally the geometry emerges during the optimization process or partial 
negligence of close to empty elements as preconditioning.   

*Topoptlab* was developed to address these challenges by providing a stable and 
extensible environment tailored to the needs of the TO community. It serves 
as a library for writing complete problems from scratch in spirit of the 
already conventional Matlab scripts, and offers a offers a high-level driver 
routine (topology_optimization) in which users can exchange components 
(filter, objective function, etc.) by passing custom callables as arguments. 
It may also serve as a reference implementation which can be used as test case 
for existing HPC codes that want to incorporate TO in their software.

*Topoptlab* offers the components needed for TO such as different material 
interpolation schemes (SIMP, RAMP, and bound-based methods), filters for 
regularization and manufacturability, and finite element solvers for different 
physical problems (linear elasticity, heat conduction, etc.) with 
both standard numerical integration and analytically integrated elements 
generated through *Symfem*. Constrained optimization is supported through MMA, 
GCMMA, and multiple optimality criteria methods, while the solution of the 
system of equations is done via routines offered by *scipy*, *cvxopt*, and also 
custom implementations of preconditioners (algebraic multigrid, geometric 
multigrid). Finally, *TopOptLab* also contains self-contained scripts 
that serve as transparent teaching tools and as an “archive” of canonical 
implementations. 

# Acknowledgments

Special thanks are extended to Christof Schulze for his valuable suggestions on code improvements, to Stefano Zapperi for hosting at the University of Milan and for his continuous support, and to Peter Råback for sharing hsi vast knowledge on FEM related implementation details.

# References