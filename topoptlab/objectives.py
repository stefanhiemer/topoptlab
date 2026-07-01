# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, Tuple, Union
from warnings import warn

import numpy as np
from scipy.ndimage import correlate, convolve

from topoptlab.aggregation import softmax_pnorm, softmax_pnorm_dx,\
                                  softmax_ks, softmax_ks_dx
from topoptlab.design_analysis import baseplate_slice, baseplate_kernel

def compliance(xPhys: np.ndarray, 
               u: np.ndarray, 
               KE: np.ndarray, 
               edofMat: np.ndarray, 
               i: int,
               matinterpol: Callable, #matinterpol_dx : Callable,
               matinterpol_kw: Dict,
               obj: float, 
               **kwargs: Any) -> Tuple[float,np.ndarray,bool]:
    """
    Update objective and gradient for stiffness maximization / compliance
    minimization. 

    Parameters
    ----------
    xPhys : np.ndarray
        SIMP densities of shape (nel).
    u : np.ndarray
        state variable (displacement, temperature) of shape (ndof).
    KE : np.ndarray
        element stiffness matrix of shape (nedof).
    edofMat : np.ndarray shape (nel,nedof)
        element degree of freedom matrix
    i : int
        index of the problem. i-th problem is used to compute the objective
        function.
    matinterpol : callable 
        callable for material interpolation. Default is SIMP (simp).
    matinterpol_kw : callable 
        dictionary containing the arguments for the material interpolation.
    obj : float
        objective function.

    Returns
    -------
    obj : float
        updated objective function.
    rhs_adj : np.ndarray
        right hand side of the adjoint problem. if problem is self adjoint,
        this is already the solution to the self-adjoint problem.
    selfadjoint : bool, True
        obj. is selfadjoint, so no adjoint problem has to be solved

    """
    ce = (((KE @ u[edofMat,i][:,:,None])[:,:,0]) * u[edofMat,i]).sum(1) if KE.ndim == 3 else (np.dot(u[edofMat,i], KE) * u[edofMat,i]).sum(1)
    obj += (matinterpol(xPhys,**matinterpol_kw)[:,0]*ce).sum()
    #dc = (-1) * matinterpol_dx(xPhys,**matinterpol_kw)*ce
    #return obj, dc, True #
    return obj,-u[:,i:i+1], True

def compliance_squarederror(xPhys: np.ndarray, 
                            u: np.ndarray, 
                            c0: float,
                            KE: np.ndarray, 
                            edofMat: np.ndarray, 
                            i: int,
                            matinterpol: Callable, #matinterpol_dx : Callable,
                            matinterpol_kw: Dict,
                            obj: float, 
                            **kwargs: Any
                            ) -> Tuple[float,np.ndarray,bool]:
    """
    Update objective and gradient for stiffness/compliance control. 

    Parameters
    ----------
    xPhys : np.ndarray
        SIMP densities of shape (nel).
    u : np.ndarray
        state variable (displacement, temperature) of shape (ndof).
    c0 : float
        target compliance
    KE : np.ndarray
        element stiffness matrix of shape (nedof).
    edofMat : np.ndarray
        element degree of freedom matrix of shape (nel,nedof)
    i : int
        index of the problem. i-th problem is used to compute the objective
        function.
    matinterpol : callable 
        callable for material interpolation. Default is SIMP (simp).
    matinterpol_kw : callable 
        dictionary containing the arguments for the material interpolation.
    obj : float
        objective function.

    Returns
    -------
    obj : float
        updated objective function.
    rhs_adj : np.ndarray
        right hand side of the adjoint problem. if problem is self adjoint,
        this is already the solution to the self-adjoint problem.
    selfadjoint : bool, True
        obj. is selfadjoint, so no adjoint problem has to be solved

    """
    ce = (np.dot(u[edofMat,i], KE)
             * u[edofMat,i]).sum(1)
    c = (matinterpol(xPhys,**matinterpol_kw)[:,0]*ce).sum()
    if isinstance(c0,float):
        delta = c - c0
    else:
        delta = c - c0[i]
    obj += delta**2
    #dc = 2*delta * (-1) * penal*xPhys**(penal-1)*(Amax-Amin)*ce
    return obj, -u * (c-c0), True 

def vol_frac(xPhys: np.ndarray, 
             el_vols: Union[None,float,np.ndarray] = None,
             col_ind: int = 0,
             **kwargs: Any) -> Tuple[float,np.ndarray,bool]:
    """
    """
    if el_vols is None:
        return xPhys[:,col_ind].mean(axis=0), np.full(xPhys.shape[0],1/xPhys.shape[0]), None
    else:
        raise NotImplementedError("Volume fraction objective is implemented for irregular meshes, but not yet tested")
        vol = el_vols.sum()
        return (xPhys*el_vols)[:,col_ind].sum(axis=0) / vol, el_vols/vol, None 

def var_maximization(u: np.ndarray, 
                     l: np.ndarray, 
                     i: int,
                     obj: float, 
                     **kwargs: Any
                     ) -> Tuple[float,np.ndarray,bool]:
    """
    Update objective and gradient for maximization of state variable in
    specified points. The mechanic version of this is the compliant mechanism
    with maximized displacement.

    Parameters
    ----------
    u : np.ndarray
        state variable (displacement, temperature) of shape (ndof).
    l : np.ndarray
        indicator vector for state variable of shape (ndof). Is 1 or -1 at
        output nodes depending on which direction of the dof you want to
        maximize.
    i : int
        index of the problem. i-th problem is used to compute the objective
        function.
    obj : float
        objective function.

    Returns
    -------
    obj : float
        updated objective function.
    rhs_adj : np.ndarray
        right hand side for the adjoint problem
    selfadjoint : bool, False
        obj. is not selfadjoint, so adjoint problem has to be solved

    """
    obj += u[l[:,i]!=0].sum()
    return obj, l, False

def var_squarederror(u: np.ndarray, 
                     u0: np.ndarray, 
                     l: np.ndarray, 
                     i: int,
                     obj: float, 
                     **kwargs: Any
                     )-> Tuple[float,np.ndarray,bool]:
    """
    Update objective and gradient for forcing a state variable to a specific
    values at certain points. The mechanic version of this is the compliant
    mechanism with controlled displacement.

    Parameters
    ----------
    u : np.ndarray
        state variable (displacement, temperature) of shape (ndof).
    u0 : np.ndarray
        value that state variable is supposed to take. shape (ncontr).
    l : np.ndarray
        indicator vector for state variable of shape (ndof). Is 1 at output
        nodes.
    obj : float
        objective function.

    Returns
    -------
    obj : float
        updated objective function.
    rhs_adj : np.ndarray
        right hand side for the adjoint problem. if problem is self adjoint,
        this is already the solution to the self-adjoint problem.
    selfadjoint : bool, False
        obj. is not selfadjoint, so adjoint problem has to be solved

    """
    #
    if isinstance(u0, float):
        u0 = np.array([u0])
    mask = l[:,i]!=0
    obj += ((u[mask,i] - u0)**2).mean()
    rhs_adj = np.zeros(l.shape)
    #mask = l != 0
    rhs_adj[mask,0] = (-2)*(u[mask,i]-u0) / u0.shape[0]
    return obj, rhs_adj , False

def _inverse_homogenization(u: np.ndarray, 
                            u0: np.ndarray, 
                            edofMat: np.ndarray, 
                            i: int, 
                            j: int,
                            KE: np.ndarray,
                            cellVolume: float, 
                            xPhys: np.ndarray,
                            Amax, Amin, penal,
                                        results, obj,
                                        **kwargs):
    """
    Update objective and gradient for stiffness maximization / compliance
    minimization. 

    Parameters
    ----------
    xPhys : np.ndarray
        SIMP densities of shape (nel).
    u : np.ndarray
        state variable (displacement, temperature) of shape (ndof).
    KE : np.ndarray
        element stiffness matrix of shape (nedof).
    edofMat : np.ndarray shape (nel,nedof)
        element degree of freedom matrix
    i : int
        index of the problem. i-th problem is used to compute the objective
        function.
    j : int
        index of the problem. i-th problem is used to compute the objective
        function.
    matinterpol : callable 
        callable for material interpolation. Default is SIMP (simp).
    matinterpol_kw : callable 
        dictionary containing the arguments for the material interpolation.
    obj : float
        objective function.

    Returns
    -------
    obj : float
        updated objective function.
    rhs_adj : np.ndarray
        right hand side of the adjoint problem. if problem is self adjoint,
        this is already the solution to the self-adjoint problem.
    selfadjoint : bool, True
        obj. is selfadjoint, so no adjoint problem has to be solved

    """
    #
    warn("Untested and probably not correct.")
    #
    if "CH" not in results.keys():
        results["CH"] = np.zeros((u.shape[-1],u.shape[-1]))
    # calculate effective elastic tensor
    du = u0[None,:] - u[edofMat]
    # # Homogenized elasticity tensor
    dobj = np.zeros(xPhys.shape[0])
    for j in range(i,u.shape[-1]):
        # calculate elemental compliance deviation
        delta_ce = (np.dot(du[edofMat,i], KE) * du[edofMat,i]).sum(1)
        deltac = ((Amin+xPhys**penal*(Amax-Amin))*delta_ce).sum()
        results["CH"][i,j] = deltac / cellVolume
        #results["CH"][i,j] = np.einsum('nj,nij,ni->n', du[:,:,i], Kes, du[:,:,j]).sum()
        dobj += (-1) * penal*xPhys**(penal-1)*(Amax-Amin)*delta_ce
    # fill off-diagonal
    results["CH"][i:,i] = results["CH"][i,i:]
    #
    obj = results["CH"][i,:].sum()
    return obj, dobj, True

def inverse_homogenization_control(u, u0, edofMat, i, KE,
                                   cellVolume, CH0, xPhys,
                                   matinterpol: Callable, #matinterpol_dx : Callable,
                                   matinterpol_kw: Dict,
                                   results, obj,
                                   **kwargs):
    #
    warn("Untested and probably not correct.")
    #
    if "CH" not in results.keys():
        results["CH"] = np.zeros(CH0)
    # calculate effective elastic tensor
    du = u0[None,:] - u[edofMat]
    # # Homogenized elasticity tensor
    dobj = np.zeros(xPhys.shape[0])
    for j in range(i,CH0.shape[-1]):
        # calculate elemental compliance deviation
        delta_ce = (np.dot(du[edofMat,i], KE) * du[edofMat,i]).sum(1)
        deltac = (matinterpol(xPhys,**matinterpol_kw)[:,0]*delta_ce).sum()
        results["CH"][i,j] = deltac / cellVolume
        #results["CH"][i,j] = np.einsum('nj,nij,ni->n', du[:,:,i], Kes, du[:,:,j]).sum()
        dc = (-1) * matinterpol(xPhys,**matinterpol_kw)[:,0]*delta_ce
        dobj += 2*(deltac/cellVolume -CH0[i,j]) * dc
    # fill off-diagonal
    results["CH"][i:,i] = results["CH"][i,i:]
    #
    obj = (results["CH"][i,:] - CH0[i,:]).sum()**2 + \
          (results["CH"][i:,:] - CH0[i,:]).sum()**2
    return obj, dobj, True

def stress_pnorm(u: np.ndarray,
                 i: int,
                 edofMat: np.ndarray,
                 B: np.ndarray,
                 C_es: np.ndarray,
                 xPhys: np.ndarray,
                 stress_vm: np.ndarray,
                 dsvm: np.ndarray,
                 penal_sig: float,
                 Pnorm: float,
                 obj: float,
                 dscale: np.ndarray,
                 cs: np.ndarray,
                 strain: np.ndarray,
                 **kwargs: Any) -> Tuple[float, np.ndarray, np.ndarray, bool]:
    """
    Aggregated relaxed von Mises stress objective.
    The relaxed stress is defined as

        stress_re = stress_vm / (xPhys_s ** penal_sig)

    and the global p-norm objective is

        J = ( (1/n) * sum stress_re**Pnorm )**(1/Pnorm)

    Parameters
    ----------
    u : ndarray of shape (ndof, nload)
        Global displacement field.
    i : int
        index of the problem. i-th problem is used to compute the objective
        function.
    edofMat : ndarray of shape (ne, ndof_el)
        Element degree-of-freedom connectivity matrix.
    B : ndarray of shape (ne, nvoigt, ndof_el)
        Element strain-displacement matrix.
    C_es : ndarray of shape (ne, nvoigt, nvoigt)
        Interpolated constitutive matrix for each element.
    xPhys : np.ndarray, shape (ne, 1)
        Physical density field.
    stress_vm : np.ndarray, shape (ne, 1)
        Element-wise von Mises stress.
    dsvm : np.ndarray, shape (ne, nvoigt)
        Derivative of the elemental von Mises stress with respect to the
        elemental stress vector,dsvm = d(stress_vm) / d(stress)
    penal_sig : float
        Penalization exponent used in the relaxed stress measure.
    Pnorm : float
        Exponent used for p-norm aggregation.
    obj : float
        Accumulated objective value.
    dscale : ndarray of shape (ne, 1)
        Derivative of the interpolation factor with respect to the physical
        density field,
    cs : ndarray of shape (ne, nvoigt, nvoigt)
        Base constitutive tensor for each element before density interpolation.
    strain : ndarray of shape (ne, nvoigt)
        Element strain vector in Voigt notation for each element.
    **kwargs : dict
        Unused extra arguments for compatibility with the optimization driver.

    Returns
    -------
    obj : float
        Updated objective value.
    rhs_adj : ndarray of shape (ndof, 1)
        Adjoint right-hand side associated with the stress objective.
    selfadjoint : bool
        Always False. The stress p-norm objective is not treated as
        self-adjoint in this implementation.
    """
    n = xPhys.shape[0]
    xPhys_s = np.maximum(xPhys, 1e-8)  
    # xPhys_s = xPhys 
    stress_re = stress_vm / (xPhys_s ** penal_sig)  
    # p-norm aggregation of the relaxed stresses.
    sP = stress_re ** Pnorm      
    mean_sP = np.mean(sP)              
    stress_pnorm = mean_sP ** (1.0 / Pnorm)  

    # Derivative of the p-norm objective with respect to the relaxed stress.   
    coeff = (1. / n) * (mean_sP ** (1. / Pnorm - 1.))
    ds_p_ds_re = coeff * (stress_re**(Pnorm - 1.))
    
    # Explicit derivative of the p-norm stress with respect to xPhys.
    # A mask is used to suppress the contribution exactly at very small
    # densities where clipping is active.
    # ds_p_dxPhys = ds_p_ds_re * (-penal_sig * stress_vm / (xPhys_s**(penal_sig + 1)))
    mask = (xPhys > 1e-8).astype(xPhys.dtype)
    ds_p_dxPhys = ds_p_ds_re * (-penal_sig * stress_vm / (xPhys_s ** (penal_sig + 1))) * mask

    # Additional direct contribution through the density dependence of the
    # stress tensor:
    #   stress = C_es : strain
    # and thus
    #   d(stress_vm)/dxPhys = dsvm : (dC_es/dxPhys : strain)
    dC_esdx = dscale[:, 0][:, None, None] * cs
    dsvm_dx = np.einsum('ei,eij,ej->e', dsvm, dC_esdx, strain, optimize=True)[:, None]
    ds_p_dxPhys += (ds_p_ds_re / (xPhys_s ** penal_sig)) * dsvm_dx

    obj += stress_pnorm
    # Derivative of relaxed stress to the elemental stress vector.
    ds_re_d_s = dsvm / (xPhys_s ** penal_sig)  
    # Element-wise derivative of the objective with respect to the element
    # displacement vector:
    #   dJ/du_e = (dJ/ds_re) * (ds_re/dsigma) * (dsigma/dstrain) * (dstrain/du_e)
    #           = ds_p_ds_re * ds_re_d_s * C_es * B
    ds_p_du = np.einsum('ei,eij,ejk->ek', ds_re_d_s, C_es, B, optimize=True)   # (ne, ndof_el)
    ds_p_du *= ds_p_ds_re
    rhs_adj = np.zeros((u.shape[0], 1), dtype=u.dtype)
    np.add.at(rhs_adj[:, 0], edofMat,- ds_p_du)
    return obj, rhs_adj, ds_p_dxPhys, False

def am_angle(xPhys: np.ndarray, 
             p: float, 
             ks_exponent: float,
             baseplate: str,
             ndim: int,
             invmapping: Callable,
             mapping: Callable,
             **kwargs: Any) -> Tuple[float, np.ndarray, np.ndarray, bool]:
    """
    Additive-manufacturing overhang constraint based on local support.

    For each element, a supporting region is defined by a baseplate-dependent
    stencil. The support value is computed as a smooth maximum over the
    densities in this stencil,

        s_e = (sum_{j in S_e} x_j^p)^(1/p),

    and the local overhang constraint is

        c_e = x_e - s_e <= 0.

    Thus, an element is feasible if its density does not exceed the smooth
    maximum density of its supporting region. The baseplate layer is excluded
    from the constraint and is always treated as feasible.

    The element-wise constraints are aggregated into one scalar constraint by
    a Kreisselmeier-Steinhauser smooth maximum,

        g = KS(c_e).

    The returned derivative is the total derivative of the aggregated
    constraint with respect to the physical densities, including the adjoint
    action of the support stencil.

    Parameters
    ----------
    xPhys : np.ndarray
        Physical element densities of shape ``(nel, 1)``.
    p : float
        Exponent used for the smooth p-norm maximum over the supporting
        stencil.
    ks_exponent : float
        Exponent used in the KS aggregation of the element-wise constraints.
    baseplate : str
        Baseplate/build direction. In 2D: ``"N"``, ``"S"``, ``"E"``, ``"W"``.
        In 3D: ``"B"``, ``"U"``, ``"N"``, ``"S"``, ``"E"``, ``"W"``.
    ndim : int
        Spatial dimension, either 2 or 3.
    invmapping : Callable
        Function mapping image/voxel-shaped arrays back to element-vector
        layout.
    mapping : Callable
        Function mapping element-vector arrays to image/voxel layout.

    Returns
    -------
    softmax : float or np.ndarray
        Aggregated AM overhang constraint value.
    softmax_dx : np.ndarray
        Derivative of ``softmax`` with respect to ``xPhys``.
    None
        Always None as not self-adjoint and also requires no 
        adjoint problem.

    """
    # kernel for supporting region
    slicing = baseplate_slice(baseplate=baseplate, 
                              ndim=ndim)
    kernel = baseplate_kernel(baseplate=baseplate, 
                              ndim=ndim)
    # get soft maxmimum of supporting region
    placeholder = invmapping(correlate(mapping(xPhys**p),
                                       weights=kernel, 
                                       axes=(0,1,2)[:ndim],
                                       mode="constant",
                                       cval=0.))
    #
    el_softmax = placeholder**(1/p)
    # element-wise constraints 
    el_constr = xPhys - el_softmax
    # baseplate row always fulfills constraint.
    el_constr = mapping(el_constr)
    el_constr[slicing] = 0. 
    el_constr = invmapping(el_constr)
    # soft maximum of element-wise constraints 
    softmax = softmax_ks(values=el_constr, 
                         exponent=ks_exponent)
    #
    w = softmax_ks_dx(values=el_constr,
                      exponent=ks_exponent)
    # baseplate row always fulfills constraint.
    w = mapping(w)
    w[slicing] = 0.
    w = invmapping(w)
    #
    mask = placeholder != 0
    placeholder[mask] = placeholder[mask]**((1-p)/p)
    #
    softmax_dx = w - xPhys**(p-1) * \
                 invmapping(convolve(mapping(w * placeholder), 
                                      weights=kernel,
                                      axes=(0,1,2)[:ndim],
                                      mode="constant",
                                      cval=0.))
    #
    softmax_dx[np.isnan(softmax_dx)] = 0.
    return softmax, softmax_dx, None

def support(xPhys: np.ndarray, 
            baseplate: str,
            ndim: int,
            invmapping: Callable,
            mapping: Callable,
            **kwargs: Any) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Zhao-style unsupported-material measure.

    Zhao, Dengyang, Ming Li, and Yusheng Liu. "Self-supporting topology 
    optimization for additive manufacturing." arXiv preprint 
    arXiv:1708.07364 (2017).

    A hard active set of unsupported solid elements is detected by checking
    whether an element has nonzero density and no material in its supporting
    stencil. The objective/constraint value is then

        g = sum_{e in U} x_e^2,

    where U is the active set of unsupported elements. The active set is
    treated as fixed when differentiating.

    Parameters
    ----------
    xPhys : np.ndarray
        Physical densities of shape ``(nel, 1)``.
    baseplate : str
        Baseplate/build direction.
    ndim : int
        Spatial dimension, either 2 or 3.
    invmapping : Callable
        Map image/voxel arrays back to element layout.
    mapping : Callable
        Map element arrays to image/voxel layout.

    Returns
    -------
    g : float
        Sum of squared densities of unsupported elements.
    dg_dx : np.ndarray
        Derivative with respect to ``xPhys`` with the active set fixed.
    None
        No adjoint right-hand side.
    """
    #
    kernel = baseplate_kernel(baseplate=baseplate, ndim=ndim)
    slicing = baseplate_slice(baseplate=baseplate, ndim=ndim)
    #
    x_img = mapping(xPhys)
    #
    unsupported = np.isclose(correlate(x_img,
                                       weights=kernel,
                                       axes=(0, 1, 2)[:ndim],
                                       mode="constant",
                                       cval=0.), 
                             0.,atol=0.1) & ~np.isclose(x_img, 0.)
    # baseplate is always supported
    unsupported[slicing] = False
    #
    unsupported = invmapping(unsupported)
    #
    dg_dx = np.zeros_like(xPhys)
    dg_dx[unsupported] = 2. * xPhys[unsupported]
    #
    return np.sum(xPhys[unsupported] ** 2), dg_dx, None

def generalized_stress(u: np.ndarray,
                       i: int,
                       edofMat: np.ndarray,
                       B: np.ndarray,
                       C_es: np.ndarray,
                       xPhys: np.ndarray,
                       stress_vm: np.ndarray,
                       dsvm: np.ndarray,
                       penal_sig: float,
                       Pnorm: float,
                       obj: float,
                       dscale: np.ndarray,
                       cs: np.ndarray,
                       strain: np.ndarray,
                       **kwargs: Any) -> Tuple[float, np.ndarray, np.ndarray, bool]:
    #
    stress = B@u[edofMat,i]
    return

if __name__ == "__main__":

    from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel
    from functools import partial 
    #
    test = np.array([0]+4*[1]+2*[0]+[1]+4*[0],order="F")[:,None]
    #
    mapping = partial(map_eltoimg,
                      nelx=4,nely=3)
    invmapping = partial(map_imgtoel,
                         nelx=4,nely=3)

    constr, dconstr,_ = am_angle(xPhys=test, 
                                 p = 10.,
                                 mapping=mapping,
                                 invmapping=invmapping,
                                 ndim=2,
                                 baseplate="S")
    #
    print(constr)
    #
    print(mapping(dconstr))