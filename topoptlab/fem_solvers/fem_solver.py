# SPDX-License-Identifier: GPL-3.0-or-later
from abc import abstractmethod
from typing import Any
from collections.abc import Callable

import numpy as np

from topoptlab.problem_solver import ProblemSolver
from topoptlab.log_utils import BaseLogger, EmptyLogger, log_material_properties  # noqa: F401


class FEMSolver(ProblemSolver):
    """
    FEM infrastructure layer.

    Orchestrates the standard FEM pipeline:
      1. collect_terms          – assemble all weak-form contributions
      2. assemble_system        – scatter into the global algebraic system
      3. apply_constraints      – enforce BCs / MPCs / periodicity
      4. solve_discrete_system  – drive the actual linear/nonlinear/eigen solve

    Contribution hooks follow semantic categories rather than geometric ones:
      core_terms       – central PDE operator (stiffness, mass, damping, …)
      source_terms     – body forces, heat sources, volume loads
      boundary_terms   – Neumann, Robin, tractions, fluxes, pressure loads
      constraint_terms – Dirichlet, MPC, periodicity, hanging nodes
      auxiliary_terms  – post-processing quantities (stresses, fluxes, …)

    All hooks return empty dicts by default; concrete solvers override only what
    they need.  The assembled system uses named block keys rather than assuming
    K and f:

      Linear elasticity : {"K_uu": K, "f_u": f}
      Heat conduction   : {"K_TT": K, "M_TT": M, "f_T": f}
      Stokes            : {"A_vv": A, "B_pv": B, "B_vp": B.T, "f_v": f, "g_p": g}
      Buckling          : {"K_uu": K, "G_uu": G}
    """

    def apply_constraints(self,
                          system: dict,
                          state: dict = {},
                          parameters: dict = {},
                          lin_solver_kw: dict = {},
                          logger: BaseLogger = None,
                          ) -> dict:
        """Enforce BCs, MPCs, periodicity, hanging nodes.  Returns modified system."""
        return system

    def assemble(self,
                 state: dict = {},
                 parameters: dict = {},
                 solver_kw: dict = {},
                 lin_solver_kw: dict = {},
                 logger: BaseLogger = None,
                 ) -> dict:
        """Collect all terms, assemble to global algebraic system, enforce BC and constraints."""
        self._terms = self.collect_terms(state, parameters, solver_kw, logger)
        system = self.assemble_system(self._terms, state, parameters, lin_solver_kw, logger)
        system = self.apply_constraints(system, state, parameters, lin_solver_kw, logger)
        return system

    def assemble_blocks(self,
                        state: dict = {},
                        parameters: dict = {},
                        solver_kw: dict = {},
                        logger: BaseLogger = None,
                        ) -> dict:
        """
        Return local blocks for monolithic assembly.

        Returns ``{"blocks": {name: matrix, ...}, "adj_rhs": {name: rhs, ...}}``.
        Used by ``adjoint_loop`` for the ``"monolithic"`` coupling mode.
        """
        raise NotImplementedError

    def assemble_system(self,
                        terms: dict,
                        state: dict = {},
                        parameters: dict = {},
                        lin_solver_kw: dict = {},
                        logger: BaseLogger = None,
                        ) -> dict:
        """Assemble local terms into the global algebraic system."""
        return {}

    def auxiliary_terms(self,
                        state: dict = {},
                        parameters: dict = {},
                        solver_kw: dict = {},
                        logger: BaseLogger = None,
                        ) -> dict:
        return {}

    def auxiliary_terms_dx(self,
                           state: dict = {},
                           parameters: dict = {},
                           adjoint: Any = None,
                           solver_kw: dict = {},
                           logger: BaseLogger = None,
                           ) -> dict:
        return {}

    def boundary_terms(self,
                       state: dict = {},
                       parameters: dict = {},
                       solver_kw: dict = {},
                       logger: BaseLogger = None,
                       ) -> dict:
        return {}

    def boundary_terms_dx(self,
                          state: dict = {},
                          parameters: dict = {},
                          adjoint: Any = None,
                          solver_kw: dict = {},
                          logger: BaseLogger = None,
                          ) -> dict:
        return {}

    def collect_terms(self,
                      state: dict = {},
                      parameters: dict = {},
                      solver_kw: dict = {},
                      logger: BaseLogger = None,
                      ) -> dict:
        terms = {}
        terms.update(self.core_terms(state, parameters, solver_kw, logger))
        terms.update(self.source_terms(state, parameters, solver_kw, logger))
        terms.update(self.boundary_terms(state, parameters, solver_kw, logger))
        terms.update(self.constraint_terms(state, parameters, solver_kw, logger))
        terms.update(self.auxiliary_terms(state, parameters, solver_kw, logger))
        return terms

    def collect_terms_dx(self,
                         state: dict = {},
                         parameters: dict = {},
                         adjoint: Any = None,
                         solver_kw: dict = {},
                         logger: BaseLogger = None,
                         ) -> dict:
        """Collect element-wise sensitivity contributions from all term hooks."""
        terms_dx = {}
        terms_dx.update(self.core_terms_dx(state, parameters, adjoint, solver_kw, logger))
        terms_dx.update(self.source_terms_dx(state, parameters, adjoint, solver_kw, logger))
        terms_dx.update(self.boundary_terms_dx(state, parameters, adjoint, solver_kw, logger))
        terms_dx.update(self.constraint_terms_dx(state, parameters, adjoint, solver_kw, logger))
        terms_dx.update(self.auxiliary_terms_dx(state, parameters, adjoint, solver_kw, logger))
        return terms_dx

    def constraint_terms(self,
                         state: dict = {},
                         parameters: dict = {},
                         solver_kw: dict = {},
                         logger: BaseLogger = None,
                         ) -> dict:
        return {}

    def constraint_terms_dx(self,
                            state: dict = {},
                            parameters: dict = {},
                            adjoint: Any = None,
                            solver_kw: dict = {},
                            logger: BaseLogger = None,
                            ) -> dict:
        return {}

    def core_terms(self,
                   state: dict = {},
                   parameters: dict = {},
                   solver_kw: dict = {},
                   logger: BaseLogger = None,
                   ) -> dict:
        return {}

    def core_terms_dx(self,
                      state: dict = {},
                      parameters: dict = {},
                      adjoint: Any = None,
                      solver_kw: dict = {},
                      logger: BaseLogger = None,
                      ) -> dict:
        return {}

    def derived_quantities(self,
                           state: dict = {},
                           parameters: dict = {},
                           solver_kw: dict = {},
                           logger: BaseLogger = None,
                           ) -> dict:
        """Calculate quantities derived from the solution, e.g. stress."""
        return {}

    def log_material_property(self,
                              solver_name: str,
                              field_name: str,
                              symmetries: list[str],
                              prop_names: list[list[str]],
                              arg_names: list[list[str]],
                              converter_functions: list[list[Callable]],
                              logger: BaseLogger = None) -> tuple[list, str]:
        """
        Generic dispatcher: validate, log, and convert ``self.material_kw`` to a
        uniform tensor representation.

        Delegates to ``log_material_properties`` in ``log_utils``, which detects
        the symmetry class of each material by matching key names from ``prop_names``
        against the material dict, then applies ``converter_functions`` to cast all
        properties to the most general symmetry class present.

        Parameters
        ----------
        solver_name : str
            Human-readable name of the solver, used only for the log header.
        field_name : str
            Name of the physical field (e.g. ``"T"``), used only for the log header.
        symmetries : list of str
            Symmetry class names in increasing generality, e.g.
            ``["isotropic", "orthotropic", "triclinic"]``.
        prop_names : list of list of str
            For each symmetry class, the expected keys in ``material_kw``.
        arg_names : list of list of str
            For each symmetry class, the corresponding keyword argument names
            passed to the converter functions.
        converter_functions : list of list of callable
            ``converter_functions[mat_sym_idx][effective_sym_idx](**prop)``
            converts a material of symmetry ``mat_sym_idx`` to the effective
            symmetry ``effective_sym_idx``.  Entries where
            ``mat_sym_idx > effective_sym_idx`` are never reached and may be
            ``None``.
        logger : BaseLogger or None

        Returns
        -------
        props : list
            Converted per-material property objects (one per entry in
            ``self.material_kw``), all in the effective symmetry representation.
        symmetry : str
            The effective (most general) symmetry class name.
        """
        logger = logger or EmptyLogger()
        logger.info(f"Material Properties for FEM {solver_name} solver for field {field_name}")
        props, symmetry = log_material_properties(
            material_kw=self.material_kw,
            symmetries=symmetries,
            names=prop_names,
            arg_names=arg_names,
            converter_functions=converter_functions,
            logger=logger)
        return props, symmetry

    def material_properties(self,
                            logger: BaseLogger = None,
                            ) -> dict:
        """Define the material model and log its properties."""
        raise NotImplementedError

    @property
    def default_interpol_kw(self) -> dict:
        """Per-property call-time kwargs derived from construction choices.

        Strips ``mode``, ``func``, and ``func_dx`` from each entry in
        ``self.interpol_kw`` and returns only the kwargs that are passed to
        the interpolation function at call time (e.g. ``eps``, ``penal``).
        Used by ``initialize_materialinterpolation`` as the baseline before
        applying user overrides.
        """
        skip = frozenset({"mode", "func", "func_dx"})
        return {p: {k: v for k, v in spec.items() if k not in skip}
                for p, spec in self.interpol_kw.items()}

    def resolve_interpol_kw(self, prop: str, parameters: dict) -> dict:
        """Return the call-time kwargs for ``prop`` to pass to the interpolation function.

        Starts from the construction-time defaults stored in
        ``self.interpol_kw[prop]`` and overrides them with any runtime values
        in ``parameters["interpol_kw"][prop]`` (e.g. updated ``penal`` during
        continuation).
        """
        skip = frozenset({"mode", "func", "func_dx"})
        kw = {k: v for k, v in self.interpol_kw[prop].items() if k not in skip}
        if "interpol_kw" in parameters and prop in parameters["interpol_kw"]:
            kw.update(parameters["interpol_kw"][prop])
        return kw

    @abstractmethod
    def setup_discretization(self,
                             *,
                             element_type: str,
                             order: int | tuple[int],
                             formulation: str,
                             ndim: int,
                             coordinate_system: str = "cartesian",
                             regular_mesh: bool = True,
                             ) -> None:
        """Creates all element-level routines and global assembly indices."""
        ...

    def solve(self,
              system: dict = {},
              state: dict = {},
              parameters: dict = {},
              solver_kw: dict = {},
              lin_solver_kw: dict = {},
              preconditioner_kw: dict = {},
              logger: BaseLogger = None,
              ) -> dict[str, Any]:
        return self.solve_discrete_system(system, state, parameters,
                                          lin_solver_kw, preconditioner_kw, logger)

    @abstractmethod
    def solve_discrete_system(self,
                              system: dict,
                              state: dict = {},
                              parameters: dict = {},
                              lin_solver_kw: dict = {},
                              preconditioner_kw: dict = {},
                              logger: BaseLogger = None,
                              ) -> dict[str, Any]:
        """Drive the actual linear / nonlinear / eigenvalue solve.

        lin_solver_kw must contain at least ``"name"``; additional keys are
        forwarded as solver-specific options.  preconditioner_kw must contain
        at least ``"name"`` (may be None to disable).
        """
        ...

    def sensitivity(self,
                    state: dict = {},
                    parameters: dict = {},
                    adjoint: Any = None,
                    solver_kw: dict = {},
                    logger: BaseLogger = None,
                    ) -> dict:
        """Compute dL/dxPhys by summing element-wise contributions from all term hooks."""
        terms_dx = self.collect_terms_dx(state, parameters, adjoint, solver_kw, logger)
        dL = np.zeros((parameters["xPhys"].shape[0], 1), order="F")
        for val in terms_dx.values():
            dL += val
        return {"dL_dxPhys": dL}

    def source_terms(self,
                     state: dict = {},
                     parameters: dict = {},
                     solver_kw: dict = {},
                     logger: BaseLogger = None,
                     ) -> dict:
        return {}

    def source_terms_dx(self,
                        state: dict = {},
                        parameters: dict = {},
                        adjoint: Any = None,
                        solver_kw: dict = {},
                        logger: BaseLogger = None,
                        ) -> dict:
        return {}

    def transient(self,
                  state: dict = {},
                  parameters: dict = {},
                  solver_kw: dict = {},
                  logger: BaseLogger = None,
                  ) -> dict[str, Any]:
        """Advance the solution by one time step."""
        raise NotImplementedError
