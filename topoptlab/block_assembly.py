# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

from topoptlab.problem_solver import ProblemSolver
from topoptlab.log_utils import BaseLogger


def assemble_global_block_system(system: dict,
                                 state: dict,
                                 parameters: dict,
                                 solver_kw: dict = {},
                                 logger: BaseLogger = None,
                                 blocks: dict = None,
                                 rhs: dict = None,
                                 ) -> dict:
    """
    Assemble a monolithic block system from individual solver blocks.

    In the forward pass ``system`` holds the already-assembled block dicts
    (one entry per solver, e.g. ``{"K_uu": ..., "K_TT": ..., "f_u": ...,
    "f_T": ...}``).  The function arranges them into a single block matrix
    and RHS, enforcing inter-field coupling terms.

    In the adjoint pass ``blocks`` and ``rhs`` carry the transposed blocks
    and adjoint RHS seeds respectively.
    """
    raise NotImplementedError("Monolithic block assembly not yet implemented.")


def solve_global_block_system(global_system: dict) -> Any:
    """Solve the assembled monolithic block system; returns a flat solution array."""
    raise NotImplementedError("Monolithic block solve not yet implemented.")


def split_global_solution_by_fields(solution: Any,
                                    problem: list[ProblemSolver],
                                    ) -> dict:
    """
    Slice the global solution vector back into per-solver field dicts.

    The returned dict can be passed directly to ``state.update(...)``.
    """
    raise NotImplementedError("Global solution splitting not yet implemented.")


def transpose_blocks(blocks: dict) -> dict:
    """Transpose each matrix block for the adjoint monolithic system."""
    return {k: v.T for k, v in blocks.items()}
