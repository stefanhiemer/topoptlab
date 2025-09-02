Introduction to Topology Optimization
==============================================

In this article we introduce the basic concepts behind topology optimization (TO). We will implicitly assume that the finite element method (FEM) is the underlying discretization technique, but TO is a general technique that is not  tied to any specific numerical technique. Loosely speaking in TO we set out to  find the optimal design parameters :math:`\boldsymbol{x}` for a given objective function :math:`C(\boldsymbol{x})`, a set of equality as well as inequality  constraints on the design  :math:`\boldsymbol{g}(\boldsymbol{x}),\boldsymbol{h}(\boldsymbol{x})` and a set of discretized physical problems :math:`\boldsymbol{K}(\boldsymbol{x}) \boldsymbol{u} = \boldsymbol{f}` with associated boundary conditions:

.. math::

   \begin{aligned}
   & \min_{\boldsymbol{x}} \; C(\boldsymbol{x}) \\
   & \text{subject to:} \\
   & \quad K(\boldsymbol{x}) \boldsymbol{u} = \boldsymbol{f}, \\
   & \quad g_i(\boldsymbol{x}) = 0, \quad i = 1, \ldots, m, \\
   & \quad h_j(\boldsymbol{x}) \leq 0, \quad j = 1, \ldots, n, \\
   & \quad \boldsymbol{x}_{\min} \leq \boldsymbol{x} \leq \boldsymbol{x}_{\max}, \quad \forall e \in \text{elements}.
   \end{aligned}

Common choices here are material volume constraints reflecting economic/monetary 
constraints or stress. The question arises now, what are the design parameters 
:math:`\boldsymbol{x}` ? A straightforward, intuitive approach would be to switch each finite element in the design between two states: either solid or void which is the same as deleting the element from the FEM. This results in a discrete optimization problem where :math:`x_e \in \{0,1\}` for each element :math:`e` which leads to a combinatorial explosion in the number of design possibilities and is also known to suffer from severe numerical problems (convergence, stability, efficiency, ...).

TO on the other hand employs a continuous relaxation of the concept of solid/void by introducing artificial relative material densities :math:`x_e \in [0, 1]` for each element, allowing for a continuous interpolation between a void state (:math:`x_e = 0`) and a fully solid state (:math:`x_e = 1`). This enables the use of constrained optimizers like the Method of Movin Asymptotes (MMA) and its globally convergent modification (GCMMA) which are known to converge fast and reliably to optimal designs. A common interpolation scheme is the Solid Isotropic Material with Penalization (SIMP) method in which the interpolation of material property :math:`A(x_e)` takes the form:

.. math::

   A(x_e) = A_{\min} + (A_0 - A_{\min}) x_e^k

Here, :math:`A_0` denotes the property of the full material, :math:`A_{\min}` 
is a small positive number used to avoid singularity in the stiffness matrix 
(typically :math:`10^{-9} A_0`), and :math:`k > 1` is a penalization exponent. 
It is important to emphasize that this interpolation is not arbitrary. It must 
obey physical bounds—such as the Hashin–Shtrikman limits—to ensure that the 
resulting material behavior remains physically meaningful. From a 
practical standpoint, the interpolation should be constructed such that 
intermediate densities (i.e., gray areas) do not influence the physical problem 
significantly, since this forces the optimizer to avoid choosing gray values, as 
they require significant "investment" in density to meaningfully alter the 
objective function and constraints. This not only improves numerical stability 
but also yields designs that are more amenable to manufacturing. After the 
optimization, the continuous densities are thresholded toa "black-and-white" 
designs of discrete densities 0/1. If volume constraints have been applied, the 
thresholding must obviously take the constraint into account.

We now proceed naively and try to find the classical result for the stiffness 
maximization of the MBB beam: we calculate the gradients (in TO called 
sensitivities) necessary for the optimization :math:`\Delta_{\boldsymbol{x}} C = 0` via adjoint analysis and use a 
constrained optimizer (e. g. MMA) which yields a strange design with artifacts 
commonly known as checkerboard patterns. These patterns arise because the 
optimizer abuses short comings of low-order elements to model the mechanical 
problem. An expensive alternative would be to employ higher order elements, but 
after some experimentation you will find designs with a mesh-dependent length 
scale rendering convergence impossible as your design must be independent of 
the underlying numerical discretization (mesh). The first ad-hoc countermeasure 
was to apply a smoothing filter to the sensitivities as this prevents the 
concentration of gradients while later the filtering of the densities was found
to guarantee well-posedness and mesh-independence.

Take home messages:
- filtering is always necessary in TO for well-posedness of the problem and 
mesh indepence
- be aware of the artifacts of FEM or the numerical method of your choice. if 
there are artifacts in your numerical physical solution, the constrained 
optimizer will (ab)use them.

For further reading:
Bendsoe, M. P. & Sigmund, O. Topology optimization: theory, methods, and applications (Springer Science & Business Media, 2003).
Sigmund, Ole, and Kurt Maute. "Topology optimization approaches: A comparative review." Structural and multidisciplinary optimization 48.6 (2013): 1031-1055.
Bayat, Mohamad, et al. "Holistic computational design within additive manufacturing through topology optimization combined with multiphysics multi-scale materials and process modelling." Progress in Materials Science 138 (2023): 101129.