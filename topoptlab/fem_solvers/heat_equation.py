# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from topoptlab.fem_solvers.fem_solver import FEMSolver
from topoptlab.fem import assemble_matrix, apply_bc, create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.elements.poisson_2d import lk_poisson_2d, _lk_poisson_2d
from topoptlab.elements.poisson_3d import lk_poisson_3d, _lk_poisson_3d
from topoptlab.material_interpolation import simp, simp_dx
from topoptlab.solve_linsystem import solve_lin
from topoptlab.log_utils import BaseLogger, EmptyLogger, log_material_properties
from topoptlab.material_tensors import resolve_property


class HeatEquation(FEMSolver):
    """
    Solver for the (compressible) heat equation:

       ρc_v ( ∂T ∂t + (u · ∇)T) − ∇ · (k∇T ) = -p ∇·u  τ : ε + ρh,

    The incompressible heat equation 

        ρc_p ( ∂T ∂t + (u · ∇)T) − ∇ · (k∇T ) = τ : ε + ρh,

    So far we are restricted to heat conduction:

       -∇ ·( K(x) ∇ T ) = f  on Omega,  T = T_D  on Gamma_D.

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
                 transient: bool = False, 
                 convection: bool = False,
                 viscous_heating: bool = False,
                 volumetric_source: Union[None,Callable] = None,
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
                 material_kw: Union[Dict, List[Dict]] = {"heat conductivity": 1.0},
                 fieldname: str = "T",
                 **kwargs: Any) -> None:
        #
        self.fieldname = fieldname
        #
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
        # for regular sets side lengths, for irregular sets length scale of mesh
        self.l = l
        # different assembly modes to use symmetry to advantage
        self.assembly_mode = assembly_mode
        if element_type is None:
            if ndim == 1:
                raise NotImplementedError("1D not implemented.")
            elif ndim == 2:
                element_type = "quadrilateral"
            else:
                element_type = "hexahedron"
        # normalize material_kw to list and detect symmetry before setup_discretization
        if isinstance(material_kw, dict):
            material_kw = [material_kw]
        self.material_kw = material_kw
        self.log_material_properties()
        #
        self.setup_discretization(element_type=element_type,
                                  order=order,
                                  formulation=formulation,
                                  ndim=ndim,
                                  regular_mesh=regular_mesh)
        # unpack BC
        self.T, self.f, self.fixed, self.free, self._springs = bc(
            nelx=nelx, nely=nely, nelz=nelz, ndof=self.ndof)
        #
        
        # set up material interpolation
        self.interpolation_mode = interpolation_mode
        self.matinterpol = matinterpol
        self.matinterpol_dx = matinterpol_dx
        self.matinterpol_kw = matinterpol_kw
        #
        self._fact = None
        self._precond = None
        return 

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
            #
            self.ndof = np.prod(np.array([self.nelx, self.nely, self.nelz][:ndim]) + 1)
            # element conductivity matrix
            _lk_map = {("quadrilateral", 1): lk_poisson_2d,
                       ("hexahedron", 1): lk_poisson_3d}
            lk = _lk_map.get((element_type, order))
            if lk is None:
                raise ValueError(f"No built-in lk for element_type={element_type!r}, order={order}.")
            # needed for some objective functions
            self._KE0 = lk(l=self.l)
            # create element degree-of-freedom matrix
            if ndim == 1:
                raise NotImplementedError("1D not implemented.")
            elif ndim == 2:
                create_edofMat = create_edofMat2d 
            else: 
                create_edofMat = create_edofMat3d
            self.edofMat = create_edofMat(nelx=self.nelx,
                                          nely=self.nely,
                                          nelz=self.nelz,
                                          nnode_dof=1)[0]
        elif formulation == "galerkin" and not regular_mesh:
            _lk_map = {("quadrilateral", 1): _lk_poisson_2d,
                       ("hexahedron", 1): _lk_poisson_3d}
            lk = _lk_map.get((element_type, order))
            raise NotImplementedError(
                "Irregular mesh not yet supported for galerkin formulation.")
        else:
            raise ValueError(f"Unknown formulation: {formulation!r}")
        # create matrix indices
        self.iK, self.jK = create_matrixinds(self.edofMat, 
                                             mode=self.assembly_mode)
        #
        if self.assembly_mode == "lower":
            assm = np.column_stack(np.tril_indices_from(self._KE0))
            self._assm_indcs = assm[np.lexsort((assm[:, 0], assm[:, 1]))]
        else:
            self._assm_indcs = None
        return 

    def core_terms(self,
                   state: Dict = {},
                   parameters: Dict = {},
                   solver_kw: Dict = {},
                   logger: BaseLogger = None,
                   ) -> Dict:
        """Scaled element conductivity matrices."""
        xPhys = parameters["xPhys"]
        if self.interpolation_mode == "scaling":
            scale = self.matinterpol(xPhys=xPhys, 
                                     **self.matinterpol_kw)
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

    def log_material_properties(self, 
                                logger: BaseLogger = None) -> Dict:
        """
        Parse, validate, log, and cache material properties from ``self.material_kw``.

        Sets ``self.symmetry`` to the most general symmetry class across all
        materials (``"isotropic"`` < ``"orthotropic"`` < ``"anisotropic"``).
        Must be called before ``setup_discretization`` so the correct element
        ``lk`` can be selected.

        Parameters
        ----------
        logger : BaseLogger or None

        Returns
        -------
        props : dict
            ``{"heat conductivity": [...], "heat conductivity symmetry": str}``
        """
        # heat conductivity
        symmetries = ["isotropic", 
                     "orthotropic", 
                     "anisotropic"]
        prop_names = [["heat conductivity"], 
                      ["heat conductivity x", 
                       "heat conductivity y", 
                       "heat conductivity z"][:self.ndim], 
                      ["heat conductivity tensor"]]
        arg_names = [["k"],
                     ["kx","ky","kz"], 
                     ["k"]]
        #
        props = log_material_properties(material_kw=self.material_kw,
                                        prop_name="heat conductivity",
                                        kind="tensor",
                                        ndim=self.ndim,
                                        fieldname=f"heat equation, field: {self.fieldname}",
                                        logger=logger)
        # function to achieve uniform representation
        normalizing = {"isotropic": False, 
                       "orthotropic": [isotropic, orthotropic, False],
                       "anisotropic": [isotropic, orthotropic, False]}
        #

        #
        return props

    def log_material_interpolation(self, logger: BaseLogger = None) -> Dict:
        """Log and return the material interpolation parameters."""
        logger = logger or EmptyLogger()
        props = {"interpolation_mode": self.interpolation_mode,
                 "matinterpol": getattr(self.matinterpol,
                                        "__name__", str(self.matinterpol)),
                 "matinterpol_dx": getattr(self.matinterpol_dx,
                                           "__name__", str(self.matinterpol_dx)),
                 **self.matinterpol_kw}
        lines = [f"Material interpolation (heat equation) for field: {self.fieldname}"]
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
