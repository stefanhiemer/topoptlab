# SPDX-License-Identifier: GPL-3.0-or-later
from abc import abstractmethod
from typing import Any, Dict, Tuple, Union

from topoptlab.problem_solver import ProblemSolver
from topoptlab.log_utils import BaseLogger  # noqa: F401 – re-exported for subclasses


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

    def assemble(self,
                 state: Dict = {},
                 parameters: Dict = {},
                 solver_kw: Dict = {},
                 logger: BaseLogger = None,
                 ) -> Dict:
        """Collect all terms, assemble to global algebraic system, enforce BC and constraints."""
        self._terms = self.collect_terms(state, parameters, solver_kw, logger)
        system = self.assemble_system(self._terms, state, parameters, solver_kw, logger)
        system = self.apply_constraints(system, state, parameters, solver_kw, logger)
        return system

    def assemble_system(self,
                        terms: Dict,
                        state: Dict = {},
                        parameters: Dict = {},
                        solver_kw: Dict = {},
                        logger: BaseLogger = None,
                        ) -> Dict:
        """Assemble local terms into the global algebraic system."""
        return {}

    def apply_constraints(self,
                          system: Dict,
                          state: Dict = {},
                          parameters: Dict = {},
                          solver_kw: Dict = {},
                          logger: BaseLogger = None,
                          ) -> Dict:
        """Enforce BCs, MPCs, periodicity, hanging nodes.  Returns modified system."""
        return system

    def collect_terms(self,
                      state: Dict = {},
                      parameters: Dict = {},
                      solver_kw: Dict = {},
                      logger: BaseLogger = None,
                      ) -> Dict:
        terms = {}
        terms.update(self.core_terms(state, parameters, solver_kw, logger))
        terms.update(self.source_terms(state, parameters, solver_kw, logger))
        terms.update(self.boundary_terms(state, parameters, solver_kw, logger))
        terms.update(self.constraint_terms(state, parameters, solver_kw, logger))
        terms.update(self.auxiliary_terms(state, parameters, solver_kw, logger))
        return terms

    def core_terms(self,
                   state: Dict = {},
                   parameters: Dict = {},
                   solver_kw: Dict = {},
                   logger: BaseLogger = None,
                   ) -> Dict:
        return {}

    def source_terms(self,
                     state: Dict = {},
                     parameters: Dict = {},
                     solver_kw: Dict = {},
                     logger: BaseLogger = None,
                     ) -> Dict:
        return {}

    def boundary_terms(self,
                       state: Dict = {},
                       parameters: Dict = {},
                       solver_kw: Dict = {},
                       logger: BaseLogger = None,
                       ) -> Dict:
        return {}

    def constraint_terms(self,
                         state: Dict = {},
                         parameters: Dict = {},
                         solver_kw: Dict = {},
                         logger: BaseLogger = None,
                         ) -> Dict:
        return {}

    def auxiliary_terms(self,
                        state: Dict = {},
                        parameters: Dict = {},
                        solver_kw: Dict = {},
                        logger: BaseLogger = None,
                        ) -> Dict:
        return {}

    def derived_quantities(self,
                           state: Dict = {},
                           parameters: Dict = {},
                           solver_kw: Dict = {},
                           logger: BaseLogger = None,
                           ) -> Dict:
        """Calculate quantities derived from the solution, e.g. stress."""
        return {}

    def material_properties(self,
                            logger: BaseLogger = None,
                            ) -> Dict:
        """Define the material model and log its properties."""
        raise NotImplementedError

    @abstractmethod
    def setup_discretization(self,
                             *,
                             element_type: str,
                             order: Union[int, Tuple[int]],
                             formulation: str,
                             ndim: int,
                             coordinate_system: str = "cartesian",
                             regular_mesh: bool = True,
                             ) -> None:
        """Creates all element-level routines and global assembly indices."""
        ...

    def solve(self,
              system: Dict = {},
              state: Dict = {},
              parameters: Dict = {},
              solver_kw: Dict = {},
              logger: BaseLogger = None,
              ) -> Dict[str, Any]:
        return self.solve_discrete_system(system, state, parameters, solver_kw, logger)

    @abstractmethod
    def solve_discrete_system(self,
                              system: Dict,
                              state: Dict = {},
                              parameters: Dict = {},
                              solver_kw: Dict = {},
                              logger: BaseLogger = None,
                              ) -> Dict[str, Any]:
        """Drive the actual linear / nonlinear / eigenvalue solve."""
        ...

    def assemble_blocks(self,
                        state: Dict = {},
                        parameters: Dict = {},
                        solver_kw: Dict = {},
                        logger: BaseLogger = None,
                        ) -> Dict:
        """
        Return local blocks for monolithic assembly.

        Returns ``{"blocks": {name: matrix, ...}, "adj_rhs": {name: rhs, ...}}``.
        Used by ``adjoint_loop`` for the ``"monolithic"`` coupling mode.
        """
        raise NotImplementedError

    def transient(self,
                  state: Dict = {},
                  parameters: Dict = {},
                  solver_kw: Dict = {},
                  logger: BaseLogger = None,
                  ) -> Dict[str, Any]:
        """Advance the solution by one time step."""
        raise NotImplementedError
