# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union

from symfem.symbols import x
from symfem.functions import MatrixFunction

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.parametric_map import jacobian
from topoptlab.symbolic.matrix_utils import simplify_matrix, generate_constMatrix
from topoptlab.symbolic.strain_measures import small_strain_matrix

def stiffness_matrix(ndim : int,
                     c : Union[None,MatrixFunction], 
                     plane_stress : bool = False,
                     element_type : str ="Lagrange",
                     order : int = 1) -> MatrixFunction:
    """
    Symbolically compute the stiffness matrix for linear elasticity.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    c : None or symfem.functions.MatrixFunction
        stiffness tensor . if None, generic stiffness tensor is assumed.
    plane_stress : bool
        if True, plane_stress is assumed. Only relevant for 2D.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    stiffness_matrix : symfem.functions.MatrixFunction
        symbolic stiffness matrix.

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # anisotropic stiffness tensor or equivalent in Voigt notation 
    if c is None:
        c = generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="c")
    #
    b = small_strain_matrix(ndim=ndim,
                            nd_inds=nd_inds,
                            basis=basis,
                            isoparam_kws={"element_type": element_type,
                                          "order": order})
    # create full integral and multiply with determinant
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = b.transpose()@c@b * Jdet
    return simplify_matrix( integrand.integral(ref,x) )

def strainforces(ndim : int,
                 c : Union[None,MatrixFunction], 
                 plane_stress : bool = False,
                 element_type : str ="Lagrange",
                 order : int = 1) -> MatrixFunction:
    """
    Symbolically compute the nodal forces due to a uniform strain for a 
    linear elastic material.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    c : None or symfem.functions.MatrixFunction
        stiffness tensor . if None, generic stiffness tensor is assumed.
    plane_stress : bool
        if True, plane_stress is assumed. Only relevant for 2D.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    nodal_forces : list
        symbolic force vector .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # anisotropic stiffness tensor or equivalent in Voigt notation
    if c is None:
        c = generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="c")
    # strains
    eps = generate_constMatrix(ncol=1,nrow=int((ndim**2 + ndim) /2),
                               name="eps")
    #
    b = small_strain_matrix(ndim=ndim,
                            nd_inds=nd_inds,
                            basis=basis,
                            isoparam_kws={"element_type": element_type,
                                          "order": order})
    # create full integral and multiply with determinant
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = b.transpose()@c@eps * Jdet
    return simplify_matrix( integrand.integral(ref,x) )

