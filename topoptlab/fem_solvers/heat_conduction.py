# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from topoptlab.fem_solvers.fem_solver import FEMSolver
from topoptlab.fem import assemble_matrix, apply_bc, create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.elements.poisson_2d import lk_poisson_2d, _lk_poisson_2d
from topoptlab.elements.poisson_3d import lk_poisson_3d, _lk_poisson_3d
from topoptlab.material_interpolation import simp, simp_dx
from topoptlab.solve_linsystem import solve_lin
from topoptlab.log_utils import BaseLogger, EmptyLogger


class HeatConduction(FEMSolver):
    """
    Stationary heat conduction (Poisson) solver.

    Solves  -div( k(x) grad T ) = f  on Omega,  T = T_D  on Gamma_D.

    Fits the FEMSolver pipeline:
      setup_discretization  – element routines and assembly indices (called from __init__)
      core_terms            – scaled element conductivity matrices K_e
      assemble_system       – scatter K_e into sparse K_TT
      apply_constraints     – enforce Dirichlet BCs → {K_TT, f_T}
      solve_discrete_system – K_TT T = f_T, caches factorization

    Parameters
    ----------
    nelx, nely : int
        Number of elements in x and y directions.
    bc : callable
        Returns ``(T0, f, fixed, free, springs)`` given
        ``(nelx, nely, nelz, ndof)``.
    ndim : int
        Spatial dimension (2 or 3).
    nelz : int or None
        Number of elements in z direction.  None for 2D.
    l : float or np.ndarray
        Element side length(s).
    formulation : str or dict
        Built-in string ``"galerkin"`` selects the standard Poisson element
        for the given ``ndim`` and polynomial order.  Pass a dict with at
        least ``{"lk": callable}`` to supply custom element functions.
    order : int or tuple of int
        Polynomial order of the element (passed to ``setup_discretization``).
    assembly_mode : str
        ``"full"`` or ``"lower"`` (exploits symmetry).
    regular_mesh : bool
        True for structured grids; irregular mesh support not yet implemented.
    interpolation_mode : str
        ``"scaling"`` – any ``scale(x_e) * K_ref`` interpolation (SIMP, RAMP, Voigt).
        ``"hashin"``  – callable returns full K_e per element (no K_ref factoring).
    matinterpol : callable
        Scaling mode: ``scale(xPhys, **matinterpol_kw) -> (nel, 1)``.
        Hashin mode:  ``Kes(xPhys, **matinterpol_kw) -> (nel, dof, dof)``.
    matinterpol_dx : callable
        Derivative of ``matinterpol`` w.r.t. ``xPhys``.
    matinterpol_kw : dict
        Extra keyword arguments forwarded to ``matinterpol`` / ``matinterpol_dx``.
    fieldname : str
        Key used for the temperature field in state / output dicts.
    """

    def __init__(self,
                 nelx: int,
                 nely: int,
                 bc: Callable,
                 ndim: int,
                 nelz: Optional[int] = None,
                 l: Union[float, np.ndarray] = 1.,
                 formulation: Union[str, Dict] = "galerkin",
                 order: Union[int, Tuple[int]] = 1,
                 assembly_mode: str = "lower",
                 regular_mesh: bool = True,
                 element_type: Union[None, str] = None,
                 interpolation_mode: str = "scaling",
                 matinterpol: Callable = simp,
                 matinterpol_dx: Callable = simp_dx,
                 matinterpol_kw: Dict = {"eps": 1e-9, "penal": 3.0},
                 fieldname: str = "T",
                 **kwargs: Any) -> None:
        self.ndim = ndim
        if regular_mesh:
            if isinstance(l, float):
                l = np.full(ndim, l)
            self.nelx = nelx
            self.nely = nely
            self.nelz = nelz
            n = np.array([nelx, nely, nelz][:ndim])
            self.nel = int(np.prod(n))
        else:
            if not isinstance(l, float):
                raise ValueError(
                    "For irregular meshes, l is the length unit of the mesh. Must be float.")
            # here i either need to have a read function based on gmsh or that needs to be done
            # already in topology_optimization.py
            raise NotImplementedError
        self.l = l
        self.assembly_mode = assembly_mode
        if element_type is None:
            if ndim == 1:
                raise NotImplementedError("1D not implemented.")
            elif ndim == 2:
                element_type = "quadrilateral"
            else:
                element_type = "hexahedron"
        self.setup_discretization(element_type=element_type,
                                  order=order,
                                  formulation=formulation,
                                  ndim=ndim,
                                  regular_mesh=regular_mesh)
        self.T, self.f, self.fixed, self.free, self._springs = bc(
            nelx=nelx, nely=nely, nelz=nelz, ndof=self.ndof)
        self.interpolation_mode = interpolation_mode
        self.matinterpol = matinterpol
        self.matinterpol_dx = matinterpol_dx
        self.matinterpol_kw = matinterpol_kw
        self.fieldname = fieldname
        self._fact = None
        self._precond = None

    def setup_discretization(self,
                             *,
                             element_type: str,
                             order: Union[int, Tuple[int]],
                             formulation: Union[str, Dict],
                             ndim: int,
                             coordinate_system: str = "cartesian",
                             regular_mesh: bool = True,
                             ) -> None:
        """Select element lk callable, compute K_E0 and all assembly indices."""
        if isinstance(formulation, dict):
            lk = formulation["lk"]
        elif formulation == "galerkin" and regular_mesh:
            _lk_map = {("quadrilateral", 1): lk_poisson_2d,
                       ("hexahedron", 1): lk_poisson_3d}
            lk = _lk_map.get((element_type, order))
            if lk is None:
                raise ValueError(
                    f"No built-in lk for element_type={element_type!r}, order={order}.")
        elif formulation == "galerkin" and not regular_mesh:
            raise NotImplementedError(
                "Irregular mesh not yet supported for galerkin formulation.")
        else:
            raise ValueError(f"Unknown formulation: {formulation!r}")
        self._KE0 = lk(l=self.l)
        nd_ndof = self._KE0.shape[0] // (2 ** ndim)
        n = np.array([self.nelx, self.nely, self.nelz][:ndim])
        self.ndof = int(np.prod(n + 1)) * nd_ndof
        _create_edofMat = create_edofMat2d if ndim == 2 else create_edofMat3d
        self.edofMat, *_ = _create_edofMat(nelx=self.nelx,
                                           nely=self.nely,
                                           nelz=self.nelz,
                                           nnode_dof=nd_ndof)
        self.iK, self.jK = create_matrixinds(self.edofMat, mode=self.assembly_mode)
        if self.assembly_mode == "lower":
            assm = np.column_stack(np.tril_indices_from(self._KE0))
            self._assm_indcs = assm[np.lexsort((assm[:, 0], assm[:, 1]))]
        else:
            self._assm_indcs = None

    def core_terms(self,
                   state: Dict = {},
                   parameters: Dict = {},
                   solver_kw: Dict = {},
                   logger: BaseLogger = None,
                   ) -> Dict:
        """Scaled element conductivity matrices."""
        xPhys = parameters["xPhys"]
        if self.interpolation_mode == "scaling":
            scale = self.matinterpol(xPhys=xPhys, **self.matinterpol_kw)
            Kes = self._KE0[None, :, :] * scale[:, :, None]
        elif self.interpolation_mode == "hashin":
            Kes = self.matinterpol(xPhys=xPhys, **self.matinterpol_kw)
        else:
            raise ValueError(f"Unknown interpolation_mode: {self.interpolation_mode!r}")
        return {"Kes": Kes}

    def assemble_system(self,
                        terms: Dict,
                        state: Dict = {},
                        parameters: Dict = {},
                        solver_kw: Dict = {},
                        logger: BaseLogger = None,
                        ) -> Dict:
        """Scatter element matrices into the global conductivity matrix K_TT."""
        Kes = terms["Kes"]
        nel = Kes.shape[0]
        dof_el = Kes.shape[1]
        lin_solver = solver_kw.get("lin_solver", "cvxopt-cholmod")
        if self.assembly_mode == "full":
            sK = Kes.reshape(nel * dof_el * dof_el)
        else:
            sK = Kes[:, self._assm_indcs[:, 0],
                        self._assm_indcs[:, 1]].reshape(
                 nel * (dof_el * (dof_el + 1) // 2))
        K = assemble_matrix(sK=sK,
                            iK=self.iK,
                            jK=self.jK,
                            ndof=self.ndof,
                            solver=lin_solver,
                            springs=self._springs)
        return {"K_TT": K}

    def apply_constraints(self,
                          system: Dict,
                          state: Dict = {},
                          parameters: Dict = {},
                          solver_kw: Dict = {},
                          logger: BaseLogger = None,
                          ) -> Dict:
        """Enforce Dirichlet BCs; returns reduced {K_TT, f_T}."""
        lin_solver = solver_kw.get("lin_solver", "cvxopt-cholmod")
        K_bc = apply_bc(K=system["K_TT"],
                        solver=lin_solver,
                        free=self.free,
                        fixed=self.fixed)
        return {"K_TT": K_bc, "f_T": self.f}

    def solve_discrete_system(self,
                              system: Dict,
                              state: Dict = {},
                              parameters: Dict = {},
                              solver_kw: Dict = {},
                              logger: BaseLogger = None,
                              ) -> Dict[str, Any]:
        """Solve K_TT T = f_T; caches factorization for adjoint reuse."""
        logger = logger or EmptyLogger()
        self.T[self.free, :], self._fact, self._precond = solve_lin(
            K=system["K_TT"],
            rhs=system["f_T"][self.free],
            rhs0=self.T[self.free, :],
            solver=solver_kw.get("lin_solver", "cvxopt-cholmod"),
            solver_kw={k: v for k, v in solver_kw.items() if k != "lin_solver"},
            factorization=self._fact,
            preconditioner=solver_kw.get("preconditioner", None),
            preconditioner_kw=solver_kw.get("preconditioner_kw", {}),
            P=self._precond,
            logger=logger)
        self._system = system
        return {self.fieldname: self.T}

    @property
    def output_keys(self) -> tuple:
        return (self.fieldname,)

    def material_properties(self,
                            logger: BaseLogger = None,
                            ) -> Dict:
        """Log and return the material model parameters."""
        logger = logger or EmptyLogger()
        props = {"interpolation_mode": self.interpolation_mode,
                 "matinterpol": getattr(self.matinterpol,
                                        "__name__", str(self.matinterpol)),
                 "matinterpol_dx": getattr(self.matinterpol_dx,
                                           "__name__", str(self.matinterpol_dx)),
                 **self.matinterpol_kw}
        lines = ["Material model (heat conduction):"]
        for k, v in props.items():
            lines.append(f"  {k}: {v}")
        logger.info("\n".join(lines))
        return props

    def adjoint(self,
                rhs: Any = None,
                state: Dict = {},
                parameters: Dict = {},
                solver_kw: Dict = {},
                logger: BaseLogger = None,
                ) -> Dict:
        """Solve K_TT^T lambda = rhs, reusing the cached factorization."""
        logger = logger or EmptyLogger()
        scalar = rhs.ndim == 1
        if scalar:
            rhs = rhs[:, None]
        adj = np.zeros_like(rhs)
        adj[self.free, :], _, _ = solve_lin(
            K=self._system["K_TT"],
            rhs=rhs[self.free, :],
            rhs0=None,
            solver=solver_kw.get("lin_solver", "cvxopt-cholmod"),
            solver_kw={k: v for k, v in solver_kw.items() if k != "lin_solver"},
            factorization=self._fact,
            preconditioner=solver_kw.get("preconditioner", None),
            preconditioner_kw=solver_kw.get("preconditioner_kw", {}),
            P=self._precond,
            logger=logger)
        adj = adj[:, 0] if scalar else adj
        return {f"adj_{self.fieldname}": adj}

    def sensitivity(self,
                    state: Dict = {},
                    parameters: Dict = {},
                    adjoint: Any = None,
                    solver_kw: Dict = {},
                    logger: BaseLogger = None,
                    ) -> Dict:
        """Compute dL/dxPhys via the material interpolation chain rule."""
        xPhys = parameters["xPhys"]
        T = state[self.fieldname]
        if adjoint.ndim == 1:
            adjoint = adjoint[:, None]
        if T.ndim == 1:
            T = T[:, None]
        n_lc = T.shape[1]
        dL = np.zeros((xPhys.shape[0], 1))
        if self.interpolation_mode == "scaling":
            scale_dx = self.matinterpol_dx(xPhys=xPhys, **self.matinterpol_kw)
            for i in range(n_lc):
                dL[:, 0] += (scale_dx *
                             (adjoint[self.edofMat, i] *
                              (self._KE0 @ T[self.edofMat, i].T).T)
                             ).sum(axis=1)
        elif self.interpolation_mode == "hashin":
            raise NotImplementedError("Hashin–Shtrikman sensitivity not yet implemented.")
        return {"dL_dxPhys": dL}
