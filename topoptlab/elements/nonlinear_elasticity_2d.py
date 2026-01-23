# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable

import numpy as np

from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi
from topoptlab.elements.isoparam_mapping import invjacobian
from topoptlab.elements.strain_measures import dispgrad_matrix,\
                                               lagrangian_strainvar_matrix
from topoptlab.fem import get_integrpoints

def _lk_nonlinear_elast_2d(xe: np.ndarray,
                           ue: np.ndarray,
                           const_tensor: Callable,
                           stress_2pk: Callable,
                           quadr_method: str = "gauss-legendre",
                           t: np.ndarray = np.array([1.]),
                           nquad: int = 2,
                           **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for 2D nonlinear elasticity with
    bilinear quadrilateral Lagrangian elements in terms of 2. Piola Kirchhoff
    stress.

    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    ue : np.ndarray,shape (nels,8).
        nodal displacements.
    const_tensor : np.ndarray or callable
        constitutive tensor in Voigt notation (same as stiffness tensor in 
        linear elasticity). Can only be an ndarray for St. Venant material and 
        then should either be of shape (ndim*(ndim+1)/2) or shape 
        (nel,ndim*(ndim+1)/2).
    stress_2pk : callable 
        2. PK stress  tensor in Voigt notation (same as stiffness tensor in 
        linear elasticity).
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    t : np.ndarray of shape (nels) or (1)
        thickness of element
    nquad : int
        number of quadrature points
        
    Returns
    -------
    Ke : np.ndarray, shape (nels,8,8)
        element stiffness matrix.

    """
    ndim=2
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel = xe.shape[0]
    #
    if not isinstance(const_tensor, np.ndarray): 
        if len(const_tensor.shape) == 2: 
            const_tensor = const_tensor[None,:,:]
    #
    if isinstance(t,float):
        t = np.array([t])
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    # calculate displacement gradient 
    B_h,detJ = dispgrad_matrix(xi=xi, eta=eta, zeta=None, xe=xe,
                               shape_functions_dxi=shape_functions_dxi,
                               invjacobian=invjacobian,
                               all_elems=True,
                               return_detJ=True) 
    B_h = B_h.reshape(nel, nq,  B_h.shape[-2], B_h.shape[-1])
    detJ = detJ.reshape(nel,nq)
    # calculate convert to matrix form
    F = (B_h@ue[:,None,:,None]).reshape(nel,nq,ndim,ndim) \
         + np.eye(ndim)[None,None,:,:]
    #
    B_dE = lagrangian_strainvar_matrix(xi=xi, 
                                       eta=eta, 
                                       zeta=None,
                                       xe=xe,
                                       F=F.reshape(nel*nq,ndim,ndim),
                                       shape_functions_dxi=shape_functions_dxi,
                                       all_elems=True,
                                       return_detJ=False)
    B_dE = B_dE.reshape(nel, nq,  B_dE.shape[-2], B_dE.shape[-1])
    # calculate constitutive tensor
    if isinstance(const_tensor, callable):
        c = const_tensor(F)
    c = np.repeat(c[:,None,:,:],nq,axis=1)
    # calculate stress in Voigt
    s = np.repeat(s[:,None,:],nq,axis=1)
    # convert to 
    # constitutive part
    integral = B_dE.transpose([0,1,3,2])@c@B_dE
    # geometric part
    integral += B_h.transpose([0,1,3,2])@s@B_h # finish here
    # multiply by determinant and quadrature
    Ke = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    print(Ke.shape)
    # multiply thickness
    return t[:,None,None] * Ke

if __name__ == "__main__":
    _lk_nonlinear_elast_2d(xe = np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                               [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           c = np.ones((2,3,3)),
                           s = np.ones((2,3)),
                           ue = np.array([[0.,0.,0.,0.,
                                0.,0.,0.,0.],
                               [0.,0.,0.,0.,
                                0.,0.,0.,0.],]),
                           exponent=1., kr=1.)