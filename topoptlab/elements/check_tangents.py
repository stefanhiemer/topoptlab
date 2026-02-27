# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, Tuple
from itertools import product

import numpy as np

from topoptlab.voigt import from_voigt
from topoptlab.fem import get_integrpoints 
from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi
from topoptlab.elements.isoparam_mapping import invjacobian
from topoptlab.elements.strain_measures import dispgrad_matrix,\
                                               lagrangian_strainvar_matrix

def check_tangent_fd(Ke_fe : Callable, 
                     Ke_fe_args : Dict, 
                     u : np.ndarray, 
                     eps=1e-6, 
                     ntests=5, 
                     verbose=True):
    """
    Central finite-difference verification of tangent stiffness Ke with respect 
    to internal forces fe, where we assume: 
        
        Ke[i,j] = dfe[i]/du[j].

    Parameters
    ----------
    Ke_fe : callable
        function that returns Ke_fe(args_dict, u) -> (fe, KT)
    Ke_fe_args : dict
        Dictionary of element/material parameters needed for Ke_fe
    u : ndarray
        current field variable vector
    eps : float
        finite difference step size
    ntests : int
        Number of random directional tests
    """
    #
    Ke, fe = Ke_fe(ue=u,**Ke_fe_args)
    #
    Ke_fd = np.zeros_like(Ke)
    for j in range(Ke.shape[1]):
        # random perturbation direction
        du = np.zeros_like(u)
        du[:,j] = eps
        # central finite difference directional derivative
        _, fe_p = Ke_fe(ue=u + du,**Ke_fe_args)
        _, fe_m = Ke_fe(ue=u - du,**Ke_fe_args)
        # tangent directional derivative
        Ke_fd[:, :, j] = (fe_p - fe_m) / (2 * eps)
    #
    errs = Ke - Ke_fd
    abs_err = np.linalg.norm(Ke - Ke_fd)
    rel_err = abs_err / (np.linalg.norm(Ke_fd) + 1e-14)
    #
    if verbose:
        print("Ke errors:", errs)
        print("Ke abs. error:", abs_err)
        print("Ke rel. error:", rel_err)

    return errs

if __name__ == "__main__":
    
    from topoptlab.material_models.stvenant import stvenant_matmodel
    from topoptlab.material_models.neohooke import neohookean_matmodel
    from topoptlab.stiffness_tensors import isotropic_2d
    from topoptlab.elements.nonlinear_elasticity_2d import _lk_nonlinear_elast_2d
    #
    nel = 1 
    xe = np.array([[[-1,-1],[1,-1],[1,1],[-1,1]]])
    print(xe.shape)
    print(np.array([[0.,0.,0.,1.,
                   0.,0.,0.,1.],
                  ]).shape)
    u = np.random.rand(nel,np.prod(xe.shape[-2:]))
    check_tangent_fd(Ke_fe = _lk_nonlinear_elast_2d,
                     Ke_fe_args = {"xe": xe,
                                   "material_model": neohookean_matmodel,
                                   "material_constants": {"c": np.ones((3,3)), 
                                                          "h": np.ones((nel,1)),
                                                          "mu": np.ones((nel,1))}},
                     u = u)