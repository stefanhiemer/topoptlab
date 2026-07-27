# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from topoptlab.fem_solvers.fem_solver import FEMSolver
from topoptlab.fem import assemble_matrix, apply_bc, create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d,\
                                                    lk_linear_elast_aniso_2d,\
                                                    _lk_linear_elast_2d,\
                                                    lf_strain_2d,\
                                                    lf_strain_aniso_2d,\
                                                    _lf_strain_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d,\
                                                    lk_linear_elast_aniso_3d,\
                                                    _lk_linear_elast_3d,\
                                                    lf_strain_3d,\
                                                    lf_strain_aniso_3d,\
                                                    _lf_strain_3d
from topoptlab.elements.bodyforce_2d import lf_bodyforce_2d,\
                                            _lf_bodyforce_2d
from topoptlab.elements.bodyforce_3d import lf_bodyforce_3d,\
                                            _lf_bodyforce_3d
from topoptlab.elements.heatexpansion_2d import fk_heatexp_aniso_2d, _fk_heatexp_2d
from topoptlab.elements.heatexpansion_3d import fk_heatexp_aniso_3d, _fk_heatexp_3d
from topoptlab.elements.mass_vector_2d import lm_mass_2d
from topoptlab.elements.mass_vector_3d import lm_mass_3d
from topoptlab.stiffness_tensors import isotropic_2d, isotropic_3d, orthotropic_2d, orthotropic_3d
from topoptlab.material_interpolation import simp, simp_dx, \
    heatexpcoeff_binary_iso, heatexpcoeff_binary_iso_dx
from topoptlab.solve_linsystem import solve_lin
from topoptlab.log_utils import BaseLogger, EmptyLogger
from topoptlab.utils import identity


class LinearElasticity(FEMSolver):
    _default_interpol_prop = "Young's modulus"
    """
    Solver for linear elasticity:

        ρ ∂**2 / d ∂t**2  = ∇ · τ + f_b

        τ = C : ε

    where f_b describes body forces. So far we are restricted to stationary
    linear elasticity

       -div( C(x) : eps(u) ) = f  on Omega,   u = u_D  on Gamma_D.

    Optionally coupled to a thermal field (HeatEquation) via thermal expansion:

       eps_th = a * DeltaT,   f_T = integral B^T C a N_T dOmega * DeltaT.

    Fits the FEMSolver pipeline:
      setup_discretization  – element routines and assembly indices
      core_terms            – scaled element stiffness matrices K_e
      source_terms          – thermal expansion load vector (optional)
      assemble_system       – scatter K_e into sparse K_uu and assemble f_u
      apply_constraints     – enforce Dirichlet BCs → {K_uu, f_u}
      solve_discrete_system – K_uu u = f_u, caches factorization

    Parameters
    ----------
    nelx, nely : int
        Number of elements in x and y directions.
    bc : callable
        Returns ``(u0, f, fixed, free, springs)`` given
        ``(nelx, nely, nelz, ndof)``.
    ndim : int
        Spatial dimension (2 or 3).
    nelz : int or None
        Number of elements in z direction.  None for 2D.
    l : float or np.ndarray
        Element side length(s).
    plane_stress : bool
        If True (default), use plane-stress formulation (2D only).
    formulation : str or dict
        ``"galerkin"`` selects the built-in element for ``ndim`` and ``order``.
    order : int
        Polynomial order.
    assembly_mode : str
        ``"full"`` or ``"lower"``.
    interpolation_mode : str
        ``"scaling"`` – element-wise ``scale(x_e) * K_ref`` (SIMP/RAMP/Voigt).
        ``"hashin"``  – callable returns full K_e per element.
    matinterpol : callable
        Scaling-mode interpolation function.
    matinterpol_dx : callable
        Derivative of ``matinterpol``.
    interpol_kw : dict
        Per-property overrides for the interpolation spec.  Key is the
        property name (``"Young's modulus"``); value is
        ``{"mode": ..., "func": ..., "func_dx": ...}``.
    material_kw : dict or list of dict
        Material property specification.  Recognised key sets:
          - isotropic   : ``{"Young's modulus": E, "Poisson's ratio": nu}``
          - orthotropic : ``{"Young's modulus x": ..., "Young's modulus y": ...,
                             "Poisson's ratio xy": ..., "shear modulus xy": ...}``
                         (3D adds z-components)
          - anisotropic : ``{"stiffness tensor": C}``
        For transient problems, ``"density"`` must additionally be provided.
    thermal_expansion_kw : dict or None
        Activates thermal expansion.  Required keys:
          - ``"thermal expansion tensor"``: np.ndarray, shape ``(ndim, ndim)``
            or ``(nel, ndim, ndim)`` for per-element expansion.
        Mutually exclusive optional keys — choose one:
          - ``"delta_T"``: float.  Uniform temperature change applied to the whole
            domain.  ΔT is a fixed external parameter; no coupled heat solver is
            needed and no coupling adjoint is produced.
          - ``"T_fieldname"`` (default ``"T"``): state key of the temperature field
            supplied by a coupled ``HeatEquation`` solver.  In the adjoint pass a
            ``dL/dT`` coupling term is produced and must be fed to the heat adjoint
            RHS by the driver.
    fieldname : str
        Key used for the displacement field in state / output dicts.
    """

    def __init__(self,
                 nelx: int,
                 nely: int,
                 bc: Callable,
                 ndim: int,
                 nelz: Optional[int] = None,
                 transient: bool = False,
                 thermal_expansion: bool = False,
                 l: Union[float, np.ndarray] = 1.,
                 plane_stress: bool = True,
                 formulation: Union[str, Dict] = "galerkin",
                 order: Union[int, Tuple[int]] = 1,
                 assembly_mode: str = "full",
                 regular_mesh: bool = True,
                 element_type: Union[None, str] = None,
                 interpolation_mode: str = "scaling",
                 matinterpol: Callable = simp,
                 matinterpol_dx: Callable = simp_dx,
                 interpol_kw: Dict = {},
                 material_kw: Union[Dict, List[Dict]] = {"Young's modulus": 1.0,
                                                         "Poisson's ratio": 0.3,
                                                         "density": 1.0},
                 thermal_expansion_kw: Union[None, Dict] = None,
                 body_forces_kw: Dict = {},
                 fieldname: str = "u",
                 logger: Union[None,BaseLogger] = None,
                 **kwargs: Any) -> None:
        self.fieldname = fieldname
        self.ndim = ndim
        self.transient = transient
        self.thermal_expansion = thermal_expansion
        self.plane_stress = plane_stress
        if regular_mesh:
            if isinstance(l, float):
                l = np.full(ndim, l)
            self.nelx = nelx
            self.nely = nely
            self.nelz = nelz
            n = np.array([nelx, nely, nelz][:ndim])
            self.nel = int(np.prod(n))
        else:
            raise NotImplementedError("Irregular mesh not yet supported.")
        self.l = l
        self.assembly_mode = assembly_mode
        if element_type is None:
            element_type = "quadrilateral" if ndim == 2 else "hexahedron"
        # normalize material_kw to list and parse material properties
        if isinstance(material_kw, dict):
            material_kw = [material_kw]
        self.material_kw = material_kw
        self.log_material_properties(logger=logger)
        # discretize
        self.setup_discretization(element_type=element_type,
                                  order=order,
                                  formulation=formulation,
                                  ndim=ndim,
                                  regular_mesh=regular_mesh)
        # unpack BC
        self.u, self.f, self.fixed, self.free, self._springs = bc(
            nelx=nelx, nely=nely, nelz=nelz, ndof=self.ndof)
        # validate multi-material interpolation
        if len(self.material_kw) > 1:
            if self.symmetry != "isotropic":
                raise ValueError(
                    "Multi-material anisotropic interpolation has no sensible default; "
                    "provide interpol_kw with an explicit \"Young's modulus\" spec.")
            if "Young's modulus" not in interpol_kw:
                raise ValueError(
                    "Multi-material isotropic requires an explicit \"Young's modulus\" "
                    "interpolation spec in interpol_kw; "
                    "the recommended default is Hashin-Shtrikman bounds.")
        # build per-property interpolation spec
        self.interpol_kw = {}
        default_spec = {"mode": interpolation_mode,
                        "func": matinterpol,
                        "func_dx": matinterpol_dx}
        if interpolation_mode == "scaling":
            ym_call_defaults = {"eps": 1e-9, "penal": 3.0}
        else:
            ym_call_defaults = {}
        ym_user = interpol_kw["Young's modulus"] if "Young's modulus" in interpol_kw else {}
        self.interpol_kw["Young's modulus"] = {**default_spec, **ym_call_defaults, **ym_user}
        dens_user = interpol_kw["density"] if "density" in interpol_kw else {}
        self.interpol_kw["density"] = {**default_spec, **ym_call_defaults, **dens_user}
        bf_user = interpol_kw["body force"] if "body force" in interpol_kw else {}
        self.interpol_kw["body force"] = {**default_spec, "eps": 0., "penal": 1., **bf_user}
        # thermal expansion coupling
        self._fTe = None
        self._th_rh_kw = None  # set below for isotropic materials
        if self.thermal_expansion:
            if thermal_expansion_kw is None:
                raise ValueError("thermal_expansion=True requires thermal_expansion_kw to be set.")
            self.thermal_expansion_kw = thermal_expansion_kw
            self._setup_thermal_coupling(thermal_expansion_kw)
        # body forces
        self._setup_body_forces(body_forces_kw)
        #
        self._fact = None
        self._precond = None

    def adjoint(self,
                rhs: Any = None,
                state: Dict = {},
                parameters: Dict = {},
                solver_kw: Dict = {},
                lin_solver_kw: Dict = {},
                preconditioner_kw: Dict = {},
                logger: BaseLogger = None,
                ) -> Dict:
        """Solve K_uu^T lambda = rhs, reusing the cached factorization."""
        logger = logger or EmptyLogger()
        adj = np.zeros_like(rhs)
        adj[self.free, :], _, _ = solve_lin(
            K=self._system["K_uu"],
            rhs=rhs[self.free, :],
            rhs0=None,
            solver=lin_solver_kw["name"],
            solver_kw={k: v for k, v in lin_solver_kw.items() if k != "name"},
            factorization=self._fact,
            preconditioner=preconditioner_kw["name"],
            preconditioner_kw={k: v for k, v in preconditioner_kw.items() \
                               if k != "name"},
            P=self._precond,
            logger=logger)
        return {f"adj_{self.fieldname}": adj}

    def apply_constraints(self,
                          system: Dict,
                          state: Dict = {},
                          parameters: Dict = {},
                          lin_solver_kw: Dict = {},
                          logger: BaseLogger = None,
                          ) -> Dict:
        """Enforce Dirichlet BCs; returns reduced {K_uu, f_u}."""
        K_bc = apply_bc(K=system["K_uu"],
                        solver=lin_solver_kw["name"],
                        free=self.free,
                        fixed=self.fixed)
        return {"K_uu": K_bc, "f_u": system["f_u"]}

    def assemble_system(self,
                        terms: Dict,
                        state: Dict = {},
                        parameters: Dict = {},
                        lin_solver_kw: Dict = {},
                        logger: BaseLogger = None,
                        ) -> Dict:
        """Scatter element contributions into the global algebraic system."""
        nel, dof_el = terms["Kes"].shape[:2]
        # stiffness matrix entries
        if self.assembly_mode == "full":
            sK = terms["Kes"].reshape(nel * dof_el**2)
        elif self.assembly_mode == "lower":
            sK = (terms["Kes"][:, self._assm_indcs[:, 0], self._assm_indcs[:, 1]]
                  .reshape(nel * self._assm_indcs.shape[0]))
        else:
            raise ValueError("Unknown assembly_mode: ", self.assembly_mode)
        K = assemble_matrix(sK=sK,
                            iK=self.iK,
                            jK=self.jK,
                            ndof=self.ndof,
                            solver=lin_solver_kw["name"],
                            springs=self._springs)
        # force contributions
        f = self.f.copy()
        for key in ("f_thermal", "f_strain", "f_gravity"):
            if key in terms:
                f += terms[key]
        system = {"K_uu": K, "f_u": f}
        # mass matrix
        if self.transient and "Mes" in terms:
            if self.assembly_mode == "full":
                sM = terms["Mes"].reshape(nel * dof_el * dof_el)
            elif self.assembly_mode == "lower":
                sM = (terms["Mes"][:, self._assm_indcs[:, 0], self._assm_indcs[:, 1]]
                      .reshape(nel * self._assm_indcs.shape[0]))
            else:
                raise ValueError("Unknown assembly_mode: ", self.assembly_mode)
            system["M_uu"] = assemble_matrix(sK=sM,
                                             iK=self.iK,
                                             jK=self.jK,
                                             ndof=self.ndof,
                                             solver=lin_solver_kw["name"],
                                             springs=None)
        return system

    def core_terms(self,
                   state: Dict = {},
                   parameters: Dict = {},
                   solver_kw: Dict = {},
                   logger: BaseLogger = None,
                   ) -> Dict:
        """Scaled element stiffness and (if transient) mass matrices."""
        elast = self.interpol_kw["Young's modulus"]
        kw = self.resolve_interpol_kw("Young's modulus", parameters)
        if elast["mode"] == "scaling":
            if elast["func"] is not None:
                scale = elast["func"](xPhys=parameters["xPhys"], 
                                      **self.resolve_interpol_kw("Young's modulus", parameters))
            else:
                scale = np.ones((parameters["xPhys"].shape[0], 1))
            terms = {"Kes": self._KE0 * scale[:, :, None]}
            if self.transient:
                terms["Mes"] = self.rho[0] * self._ME0 * scale[:, :, None]
        elif elast["mode"] == "hashin":
            if elast["func"] is None:
                raise ValueError("hashin mode requires a func; set func=None only for scaling mode.")
            terms = {"Kes": self._KE0 * elast["func"](xPhys=parameters["xPhys"], 
                     **self.resolve_interpol_kw("Young's modulus", parameters))}
            if self.transient:
                raise NotImplementedError("Mass matrix not supported for hashin mode.")
        else:
            raise ValueError(f"Unknown interpolation mode: {self.interpol_kw["Young's modulus"]['mode']!r}")
        return terms

    def log_material_properties(self,
                                logger: BaseLogger = None) -> Dict:
        """
        Parse, validate, log, and cache the stiffness tensors from
        ``self.material_kw``.  Sets ``self.symmetry`` and ``self.C``
        (list of (n_voigt, n_voigt) stiffness tensors, one per material).
        """
        #
        logger = logger or EmptyLogger()
        # stiffness related properties
        n_shear = int(((self.ndim + 1) * self.ndim) / 2) - self.ndim
        symmetries = ["isotropic", "orthotropic", "anisotropic"]
        prop_names = [["Young's modulus",
                       "Poisson's ratio",
                       "shear modulus",
                       "bulk modulus",
                       "Lamé's first parameter",
                       "P-wave modulus"],
                      ["Young's modulus x", "Young's modulus y", "Young's modulus z"][:self.ndim] +
                      ["Poisson's ratio xy", "Poisson's ratio xz", "Poisson's ratio yz"][:n_shear] +
                      ["shear modulus xy", "shear modulus xz", "shear modulus yz"][:n_shear],
                      ["stiffness tensor"]]
        arg_names = [["E", "nu"],
                     ["Ex", "Ey", "Ez"][:self.ndim] +
                     ["nu_xy", "nu_xz", "nu_yz"][:n_shear] +
                     ["G_xy", "G_xz", "G_yz"][:n_shear],
                     ["C"]]
        if self.ndim == 2:
            converter_functions = [[partial(isotropic_2d, 
                                            plane_stress=self.plane_stress),
                                    partial(isotropic_2d, 
                                            plane_stress=self.plane_stress),
                                    partial(isotropic_2d, 
                                            plane_stress=self.plane_stress)],
                                   [None,
                                   partial(orthotropic_2d, 
                                           plane_stress=self.plane_stress),
                                   partial(orthotropic_2d, 
                                           plane_stress=self.plane_stress)],
                                   [None, 
                                   None, 
                                   identity],]
        else:
            converter_functions = [[isotropic_3d, 
                                    isotropic_3d, 
                                    isotropic_3d],
                                   [None, 
                                    orthotropic_3d, 
                                    orthotropic_3d],
                                   [None, 
                                    None, 
                                    identity],]
        #
        props, symmetry = self.log_material_property(
            solver_name="LinearElasticity",
            field_name=self.fieldname,
            symmetries=symmetries,
            prop_names=prop_names,
            arg_names=arg_names,
            converter_functions=converter_functions,
            logger=logger)
        #
        self.symmetry = symmetry
        self.C = props
        result = {"stiffness": {"symmetry": self.symmetry, 
                                "materials": props}}
        if self.transient:
            density_props, _ = self.log_material_property(
                solver_name="LinearElasticity",
                field_name=self.fieldname,
                symmetries=["isotropic"],
                prop_names=[["density"]],
                arg_names=[["rho"]],
                converter_functions=[[identity]],
                logger=logger)
            result["density"] = {"materials": density_props}
            self.rho = density_props
        return result

    @property
    def output_keys(self) -> tuple:
        return (self.fieldname,)

    def core_terms_dx(self,
                      state: Dict = {},
                      parameters: Dict = {},
                      adjoint: Any = None,
                      solver_kw: Dict = {},
                      logger: BaseLogger = None,
                      ) -> Dict:
        """Element-wise sensitivity of the stiffness contribution."""
        #
        xPhys = parameters["xPhys"]
        u = state[self.fieldname]
        if u.ndim != 2:
            raise ValueError(f"state['{self.fieldname}'] must be 2D (ndof, n_rhs), got shape {u.shape}")
        if adjoint.ndim != 2:
            raise ValueError(f"adjoint must be 2D (ndof, n_rhs), got shape {adjoint.shape}")
        elast = self.interpol_kw["Young's modulus"]
        kw = self.resolve_interpol_kw("Young's modulus", parameters)
        dL_core = np.zeros((xPhys.shape[0], 1), order="F")
        if elast["func_dx"] is not None:
            scale_dx = elast["func_dx"](xPhys=xPhys, **kw)
            for i in range(u.shape[1]):
                u_el = u[self.edofMat, i]
                adj_el = adjoint[self.edofMat, i]
                Ku_el = (self._KE0 @ u_el[:, :, None]).squeeze(-1)
                dL_core[:, 0] += (scale_dx[:, 0] * (adj_el * Ku_el).sum(axis=1))
        return {"dL_core": dL_core}

    def setup_discretization(self,
                             *,
                             element_type: str,
                             order: Union[int, Tuple[int]],
                             formulation: Union[str, Dict],
                             ndim: int,
                             coordinate_system: str = "cartesian",
                             regular_mesh: bool = True,
                             refcell_needed: bool = False,
                             ) -> None:
        """Select element lk callable, compute _KE0 and assembly indices."""
        #
        if refcell_needed:
            pass
        #
        if isinstance(formulation, dict):
            raise NotImplementedError("Custom formulation dict not yet supported.")
        elif formulation == "galerkin" and regular_mesh:
            self.ndof = int(np.prod(np.array([self.nelx, self.nely, self.nelz][:ndim]) + 1)) * ndim
            if len(self.material_kw) != 1:
                raise NotImplementedError("Per-element material assignment not yet supported.")
            if ndim == 2:
                create_edofMat = create_edofMat2d
                if (element_type, order) == ("quadrilateral", 1):
                    if self.symmetry == "isotropic":
                        E = self.material_kw[0]["Young's modulus"]
                        nu = self.material_kw[0]["Poisson's ratio"]
                        self._KE0 = lk_linear_elast_2d(
                            E=E, nu=nu, plane_stress=self.plane_stress, l=self.l)[None]
                    else:
                        self._KE0 = lk_linear_elast_aniso_2d(c=self.C[0], l=self.l)[None]
                else:
                    raise ValueError(
                        f"No built-in lk for element_type={element_type!r}, order={order}.")
            else:
                create_edofMat = create_edofMat3d
                if (element_type, order) == ("hexahedron", 1):
                    if self.symmetry == "isotropic":
                        E = self.material_kw[0]["Young's modulus"]
                        nu = self.material_kw[0]["Poisson's ratio"]
                        self._KE0 = lk_linear_elast_3d(E=E, nu=nu, l=self.l)[None]
                    else:
                        self._KE0 = lk_linear_elast_aniso_3d(c=self.C[0], l=self.l)[None]
                else:
                    raise ValueError(
                        f"No built-in lk for element_type={element_type!r}, order={order}.")
            self.edofMat = create_edofMat(nelx=self.nelx, nely=self.nely, nelz=self.nelz,
                                          nnode_dof=ndim)[0]
            self.edofMatT = create_edofMat(nelx=self.nelx, nely=self.nely, nelz=self.nelz,
                                           nnode_dof=1)[0]
            self.iK, self.jK = create_matrixinds(self.edofMat, mode=self.assembly_mode)
            if self.assembly_mode == "lower":
                assm = np.column_stack(np.tril_indices_from(self._KE0[0]))
                self._assm_indcs = assm[np.lexsort((assm[:, 0], assm[:, 1]))]
            else:
                self._assm_indcs = None
            if self.transient:
                if ndim == 2:
                    self._ME0 = lm_mass_2d(p=1., l=self.l)[None]
                else:
                    self._ME0 = lm_mass_3d(p=1., l=self.l)[None]
        elif formulation == "galerkin" and not regular_mesh:
            raise NotImplementedError("Irregular mesh not yet supported for galerkin formulation.")
        else:
            raise ValueError(f"Unknown formulation: {formulation!r}")

    def _setup_body_forces(self, body_forces_kw: Dict) -> None:
        """Precompute reference element body-force vectors."""
        self._FE_strain0 = None
        self._FE_grav0 = None
        if "strain_uniform" in body_forces_kw:
            eps = body_forces_kw["strain_uniform"]
            if eps.ndim == 1:
                eps = eps[:, None]
            n_lc = eps.shape[1]
            if self.ndim == 2:
                if self.symmetry == "isotropic":
                    E = self.material_kw[0]["Young's modulus"]
                    nu = self.material_kw[0]["Poisson's ratio"]
                    fe_cols = [lf_strain_2d(eps=eps[:, i], E=E, nu=nu,
                                            plane_stress=self.plane_stress, l=self.l)
                               for i in range(n_lc)]
                else:
                    fe_cols = [lf_strain_aniso_2d(eps=eps[:, i], c=self.C[0], l=self.l)
                               for i in range(n_lc)]
            else:
                if self.symmetry == "isotropic":
                    E = self.material_kw[0]["Young's modulus"]
                    nu = self.material_kw[0]["Poisson's ratio"]
                    fe_cols = [lf_strain_3d(eps=eps[:, i], E=E, nu=nu, l=self.l)
                               for i in range(n_lc)]
                else:
                    fe_cols = [lf_strain_aniso_3d(eps=eps[:, i], c=self.C[0], l=self.l)
                               for i in range(n_lc)]
            self._FE_strain0 = np.concatenate(fe_cols, axis=1)[None]  # (1, dof_el, n_lc)
        if "density_coupled" in body_forces_kw:
            b = body_forces_kw["density_coupled"]
            if self.ndim == 2:
                self._FE_grav0 = lf_bodyforce_2d(b=b, l=self.l)[None]  # (1, dof_el, 1)
            else:
                self._FE_grav0 = lf_bodyforce_3d(b=b, l=self.l)[None]

    def _setup_thermal_coupling(self, thermal_expansion_kw: Dict) -> None:
        """Precompute thermal-expansion coupling matrices and determine temperature source."""
        # Rosen-Hashin parameters for the alpha interpolation (isotropic only)
        if self.symmetry == "isotropic":
            E = self.material_kw[0]["Young's modulus"]
            nu = self.material_kw[0]["Poisson's ratio"]
            K_s = E / (3.0 * (1.0 - 2.0 * nu))
            eps = self.interpol_kw["Young's modulus"].get("eps", 1e-9)
            self._th_rh_kw = {"K_s": K_s, "Kmin": eps * K_s, "Kmax": K_s}
        a = thermal_expansion_kw["thermal expansion tensor"]
        if a.ndim == 2:
            if self.ndim == 2:
                KeET_ref = fk_heatexp_aniso_2d(c=self.C[0], a=a, l=self.l)
            else:
                KeET_ref = fk_heatexp_aniso_3d(c=self.C[0], a=a, l=self.l)
            self._KeET = np.tile(KeET_ref[None], (self.nel, 1, 1))
        else:
            if self.ndim == 2:
                self._KeET = np.stack([
                    fk_heatexp_aniso_2d(c=self.C[0], a=a[e], l=self.l)
                    for e in range(self.nel)
                ])
            else:
                self._KeET = np.stack([
                    fk_heatexp_aniso_3d(c=self.C[0], a=a[e], l=self.l)
                    for e in range(self.nel)
                ])
        # scalar uniform ΔT: precompute constant element load via partition of unity
        # (summing KeET over T DOFs collapses N_T to 1 for a uniform temperature).
        # No coupling adjoint is produced — ΔT is a fixed external parameter.
        if "delta_T" in thermal_expansion_kw:
            self._T_fieldname = None
            self._delta_T = thermal_expansion_kw["delta_T"]
            self._fTe = self._KeET.sum(axis=-1, keepdims=True) * self._delta_T
        # field mode: temperature comes from a coupled HeatEquation solver each
        # forward pass.  The adjoint pass must produce a dL/dT coupling term and
        # feed it to the heat adjoint RHS (responsibility of the driver).
        else:
            if "T_fieldname" in thermal_expansion_kw:
                self._T_fieldname = thermal_expansion_kw["T_fieldname"]
            else:
                self._T_fieldname = "T"
            self._delta_T = None
            self._fTe = None  # set each forward pass in source_terms

    def solve_discrete_system(self,
                              system: Dict,
                              state: Dict = {},
                              parameters: Dict = {},
                              lin_solver_kw: Dict = {},
                              preconditioner_kw: Dict = {},
                              logger: BaseLogger = None,
                              ) -> Dict[str, Any]:
        """Solve K_uu u = f_u; caches factorization for adjoint reuse."""
        logger = logger or EmptyLogger()
        self.u[self.free, :], self._fact, self._precond = solve_lin(
            K=system["K_uu"],
            rhs=system["f_u"][self.free],
            rhs0=self.u[self.free, :],
            solver=lin_solver_kw["name"],
            solver_kw={k: v for k, v in lin_solver_kw.items() if k != "name"},
            factorization=self._fact,
            preconditioner=preconditioner_kw["name"],
            preconditioner_kw={k: v for k, v in preconditioner_kw.items() if k != "name"},
            P=self._precond,
            logger=logger)
        self._system = system
        return {self.fieldname: self.u}

    def source_terms(self,
                     state: Dict = {},
                     parameters: Dict = {},
                     solver_kw: Dict = {},
                     logger: BaseLogger = None,
                     ) -> Dict:
        """Thermal expansion, strain-induced, and gravity body-force load vectors."""
        terms = {}
        elast = self.interpol_kw["Young's modulus"]
        kw = self.resolve_interpol_kw("Young's modulus", parameters)
        # thermal expansion: scale_C (stiffness) * scale_alpha (Rosen-Hashin)
        if self.thermal_expansion:
            if elast["func"] is not None:
                scale_C = elast["func"](xPhys=parameters["xPhys"], **kw)
            else:
                scale_C = np.ones((parameters["xPhys"].shape[0], 1))
            if self._th_rh_kw is not None:
                scale_alpha = heatexpcoeff_binary_iso(
                    xPhys=parameters["xPhys"],
                    K=scale_C * self._th_rh_kw["K_s"],
                    Kmin=self._th_rh_kw["Kmin"],
                    Kmax=self._th_rh_kw["Kmax"],
                    amin=0., amax=1.)
                scale = scale_C * scale_alpha
            else:
                scale = scale_C
            fT = np.zeros(self.f.shape)
            if self._T_fieldname is None:
                # scalar uniform ΔT — fTe precomputed at setup, same for every load case
                for i in range(fT.shape[1]):
                    np.add.at(fT[:, i], self.edofMat.flatten(),
                              (scale * self._fTe[:, :, 0]).flatten())
            elif self._T_fieldname in state:
                # field mode: temperature field from coupled HeatEquation
                T = state[self._T_fieldname]
                if T.ndim == 1:
                    T = T[:, None]
                fTe = self._KeET @ T[self.edofMatT]
                self._fTe = fTe
                for i in range(T.shape[1]):
                    np.add.at(fT[:, i], self.edofMat.flatten(),
                              (scale * fTe[:, :, i]).flatten())
            terms["f_thermal"] = fT
        # strain-induced body forces: stiffness interpolation (prescribed eigenstrain)
        if self._FE_strain0 is not None:
            if elast["func"] is not None:
                scale = elast["func"](xPhys=parameters["xPhys"], **kw)
            else:
                scale = np.ones((parameters["xPhys"].shape[0], 1))
            n_lc = self._FE_strain0.shape[-1]
            fS = np.zeros((self.ndof, n_lc))
            for i in range(n_lc):
                np.add.at(fS[:, i], self.edofMat.flatten(),
                          (scale * self._FE_strain0[:, :, i]).flatten())
            terms["f_strain"] = fS
        # gravity body forces (own interpolation, linear by default)
        if self._FE_grav0 is not None:
            gravity = self.interpol_kw["body force"]
            grav_kw = self.resolve_interpol_kw("body force", parameters)
            if gravity["func"] is not None:
                scale_g = gravity["func"](xPhys=parameters["xPhys"], **grav_kw)
            else:
                scale_g = np.ones((parameters["xPhys"].shape[0], 1))
            fG = np.zeros((self.ndof, 1))
            np.add.at(fG[:, 0], self.edofMat.flatten(),
                      (scale_g * self._FE_grav0[:, :, 0]).flatten())
            terms["f_gravity"] = fG
        return terms

    def source_terms_dx(self,
                        state: Dict = {},
                        parameters: Dict = {},
                        adjoint: Any = None,
                        solver_kw: Dict = {},
                        logger: BaseLogger = None,
                        ) -> Dict:
        """Element-wise sensitivities of thermal, strain-induced, and gravity contributions."""
        has_thermal = self.thermal_expansion and self._fTe is not None
        has_strain = self._FE_strain0 is not None
        has_gravity = self._FE_grav0 is not None
        if not (has_thermal or has_strain or has_gravity):
            return {}
        xPhys = parameters["xPhys"]
        if adjoint.ndim == 1:
            adjoint = adjoint[:, None]
        elast = self.interpol_kw["Young's modulus"]
        kw = self.resolve_interpol_kw("Young's modulus", parameters)
        dL_source = np.zeros((xPhys.shape[0], 1), order="F")
        # thermal expansion sensitivity: product rule d/dx(scale_C * scale_alpha)
        if has_thermal and elast["func_dx"] is not None:
            scale_C = elast["func"](xPhys=xPhys, **kw)
            scale_C_dx = elast["func_dx"](xPhys=xPhys, **kw)
            if self._th_rh_kw is not None:
                K_eff = scale_C * self._th_rh_kw["K_s"]
                dK_eff_dx = scale_C_dx * self._th_rh_kw["K_s"]
                scale_alpha = heatexpcoeff_binary_iso(
                    xPhys=xPhys,
                    K=K_eff,
                    Kmin=self._th_rh_kw["Kmin"],
                    Kmax=self._th_rh_kw["Kmax"],
                    amin=0., amax=1.)
                dalpha_dx = heatexpcoeff_binary_iso_dx(
                    xPhys=xPhys,
                    K=K_eff,
                    dKdx=dK_eff_dx,
                    Kmin=self._th_rh_kw["Kmin"],
                    Kmax=self._th_rh_kw["Kmax"],
                    amin=0., amax=1.)
                combined_dx = scale_C_dx * scale_alpha + scale_C * dalpha_dx
            else:
                combined_dx = scale_C_dx
            if self._T_fieldname is None:
                fTe_el = self._fTe[:, :, 0]
                for i in range(adjoint.shape[1]):
                    adj_el = adjoint[self.edofMat, i]
                    dL_source[:, 0] -= combined_dx[:, 0] * (adj_el * fTe_el).sum(axis=1)
            else:
                for i in range(self._fTe.shape[-1]):
                    adj_el = adjoint[self.edofMat, i]
                    dL_source[:, 0] -= combined_dx[:, 0] * (adj_el * self._fTe[:, :, i]).sum(axis=1)
        # strain-induced sensitivity: stiffness interpolation (prescribed eigenstrain)
        if has_strain and elast["func_dx"] is not None:
            scale_dx = elast["func_dx"](xPhys=xPhys, **kw)
            for i in range(self._FE_strain0.shape[-1]):
                adj_el = adjoint[self.edofMat, i]
                dL_source[:, 0] -= (scale_dx[:, 0]
                                    * (adj_el * self._FE_strain0[0, :, i]).sum(axis=1))
        # gravity sensitivity
        if has_gravity:
            gravity = self.interpol_kw["body force"]
            grav_kw = self.resolve_interpol_kw("body force", parameters)
            if gravity["func_dx"] is not None:
                scale_g_dx = gravity["func_dx"](xPhys=xPhys, **grav_kw)
                adj_el = adjoint[self.edofMat, 0]
                dL_source[:, 0] -= (scale_g_dx[:, 0]
                                    * (adj_el * self._FE_grav0[0, :, 0]).sum(axis=1))
        return {"dL_source": dL_source}

def discretization_regular(nelx: int,
                           nely: Union[None,int],
                           nelz: Union[None,int],
                           ndim: int,  
                           material_kw: Dict, 
                           refcell_needed: bool = False,
                           scalar_needed: bool = False) -> int:

    ndof = int(np.prod(np.array([nelx, nely, nelz][:ndim]) + 1)) * ndim
    if len(material_kw) != 1:
        raise NotImplementedError("Per-element material assignment not yet supported.")
    if refcell_needed:
        xe_ref = ref_cell(ndim=ndim, 
                          l=l)
    else:
        xe_ref = None
    #
    if ndim == 2:
        create_edofMat = create_edofMat2d
        if (element_type, order) == ("quadrilateral", 1):
            if self.symmetry == "isotropic":
                E = material_kw[0]["Young's modulus"]
                nu = material_kw[0]["Poisson's ratio"]
                KE = lk_linear_elast_2d(
                    E=E, nu=nu, plane_stress=self.plane_stress, l=self.l)[None]
            else:
                KE = lk_linear_elast_aniso_2d(c=C[0], 
                                              l=self.l)[None]
        else:
            raise ValueError(
                f"No built-in lk for element_type={element_type!r}, order={order}.")
    else:
        create_edofMat = create_edofMat3d
        if (element_type, order) == ("hexahedron", 1):
            if self.symmetry == "isotropic":
                E = self.material_kw[0]["Young's modulus"]
                nu = self.material_kw[0]["Poisson's ratio"]
                self._KE0 = lk_linear_elast_3d(E=E, nu=nu, l=self.l)[None]
            else:
                self._KE0 = lk_linear_elast_aniso_3d(c=self.C[0], l=self.l)[None]
        else:
            raise ValueError(
                f"No built-in lk for element_type={element_type!r}, order={order}.")
    edofMat = create_edofMat(nelx=self.nelx, nely=self.nely, nelz=self.nelz,
                             nnode_dof=ndim)[0]
    edofMat_scalar = create_edofMat(nelx=self.nelx, nely=self.nely, nelz=self.nelz,
                                    nnode_dof=1)[0]
    self.iK, self.jK = create_matrixinds(self.edofMat, mode=self.assembly_mode)
    if self.assembly_mode == "lower":
        assm = np.column_stack(np.tril_indices_from(self._KE0[0]))
        self._assm_indcs = assm[np.lexsort((assm[:, 0], assm[:, 1]))]
    else:
        self._assm_indcs = None
    if self.transient:
        if ndim == 2:
            self._ME0 = lm_mass_2d(p=1., l=self.l)[None]
        else:
            self._ME0 = lm_mass_3d(p=1., l=self.l)[None]

    return ndof

def ref_cell(ndim,
             l) -> np.ndarray:
    if ndim == 1:
        xe = l * np.array([[-1.],[1.]])/2
    elif ndim == 2:
        xe = l * np.array([[[-1., -1.], [1., -1.],
                            [1., 1.], [-1., 1.]]]) / 2 
    elif ndim == 3:
        xe = l * np.array([[[-1, -1, -1], [1, -1, -1], [1, 1, -1],
                            [-1, 1, -1], [-1, -1, 1], [1, -1, 1],
                            [1, 1, 1], [-1, 1, 1]]]) / 2
    else:
        raise ValueError("ndim must be in [1,2,3]: ", ndim)
    return xe

def log_stiffness():

    return 

def log_density():

    return 

def log_bodyforce():

    return 

def log_strain():

    return 