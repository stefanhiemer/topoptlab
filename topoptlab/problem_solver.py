# SPDX-License-Identifier: GPL-3.0-or-later
from abc import ABC, abstractmethod
from typing import Any, Dict

from topoptlab.log_utils import BaseLogger


class ProblemSolver(ABC):
    """
    Generic solver protocol for topology optimisation and multiphysics problems.

    Assembly and solution are separate steps so that solvers with an unchanged
    system (same operators, changed RHS only) can skip reassembly:

      system = solver.assemble(state, parameters, solver_kw, logger)        # once
      out1   = solver.solve(system, state1, parameters, solver_kw, logger)  # many times
      out2   = solver.solve(system, state2, parameters, solver_kw, logger)

    __call__ combines both steps for convenience.

    Only solve() is required.  All other methods (assemble, adjoint, sensitivity,
    residual, linearize, derivative) are optional and raise NotImplementedError
    by default so that discrete, graph, black-box, or non-differentiable solvers
    can be used without dummy implementations.

    Communication between solvers uses plain dicts:
      state      – field values produced and consumed by solvers
      parameters – design variables or fixed problem parameters
      solver_kw  – linear solver settings: name, tolerances, preconditioner, …
    """

    def __call__(self,
                 state: Dict = {},
                 parameters: Dict = {},
                 solver_kw: Dict = {},
                 logger: BaseLogger = None,
                 ) -> dict[str, Any]:
        """Assemble and solve in one step."""
        system = self.assemble(state, parameters, solver_kw, logger)
        return self.solve(system, state, parameters, solver_kw, logger)

    def adjoint(self,
                rhs: Any = None,
                state: Dict = {},
                parameters: Dict = {},
                solver_kw: Dict = {},
                logger: BaseLogger = None,
                ) -> Any:
        """Solve the adjoint system for the given right-hand side."""
        raise NotImplementedError

    def assemble(self,
                 state: Dict = {},
                 parameters: Dict = {},
                 solver_kw: Dict = {},
                 logger: BaseLogger = None,
                 ) -> dict:
        """Assemble discrete operators (matrices, RHS, constraints). Returns a system dict."""
        raise NotImplementedError

    def derivative(self,
                   wrt: str = "",
                   state: Dict = {},
                   parameters: Dict = {},
                   solver_kw: Dict = {},
                   logger: BaseLogger = None,
                   ) -> Any:
        """Return the derivative of the output w.r.t. the named quantity."""
        raise NotImplementedError

    @property
    def implicit(self) -> bool:
        """True if the solver requires assembling and solving a linear system
        (implicit); False for explicit update rules that only evaluate the RHS."""
        return True

    @property
    def input_keys(self) -> tuple[str, ...]:
        """Keys this solver reads from state / parameters."""
        return ()

    def linearize(self,
                  state: Dict = {},
                  parameters: Dict = {},
                  solver_kw: Dict = {},
                  logger: BaseLogger = None,
                  ) -> Any:
        """Return the linearization (Jacobian / tangent) of the problem at the current state."""
        raise NotImplementedError

    def material_properties(self,
                            logger: BaseLogger = None,
                            ) -> Dict:
        """Define the material model and log its properties."""
        raise NotImplementedError

    @property
    def output_keys(self) -> tuple[str, ...]:
        """Keys this solver writes into the returned dict."""
        return ()

    def residual(self,
                 state: Dict = {},
                 parameters: Dict = {},
                 solver_kw: Dict = {},
                 logger: BaseLogger = None,
                 ) -> Any:
        """Evaluate the residual R(state, parameters). Needed for nonlinear problems."""
        raise NotImplementedError

    def sensitivity(self,
                    state: Dict = {},
                    parameters: Dict = {},
                    adjoint: Any = None,
                    solver_kw: Dict = {},
                    logger: BaseLogger = None,
                    ) -> Any:
        """Compute dL/d(parameters) given the adjoint field."""
        raise NotImplementedError

    @abstractmethod
    def solve(self,
              system: Dict = {},
              state: Dict = {},
              parameters: Dict = {},
              solver_kw: Dict = {},
              logger: BaseLogger = None,
              ) -> dict[str, Any]:
        """Solve the assembled system. Returns new fields to merge into state."""
        ...

    def transient(self,
                  state: Dict = {},
                  parameters: Dict = {},
                  solver_kw: Dict = {},
                  logger: BaseLogger = None,
                  ) -> Any:
        """Advance the solution by one time step."""
        raise NotImplementedError
