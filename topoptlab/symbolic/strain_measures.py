# SPDX-License-Identifier: GPL-3.0-or-later
from sympy import expand,simplify,Expr
from symfem.functions import MatrixFunction

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.matrix_utils import simplify_matrix, generate_constMatrix
from topoptlab.symbolic.matrix_utils import eye,to_square,inverse
from topoptlab.symbolic.lin_elastic import small_strain_matrix,dispgrad_matrix
from topoptlab.symbolic.voigt import convert_from_voigt

def eng_strain(ndim: int,
               element_type: str= "Lagrange",
               order: int = 1) -> MatrixFunction:
    """
    Symbolically compute engineering strain.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    eng_strains : MatrixFunction
        engineering strains in Voigt notation shape ( (ndim**2 + ndim)/2 ) .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # small strain matrix
    b = small_strain_matrix(ndim=ndim,
                            nd_inds=nd_inds,
                            basis=basis,
                            isoparam_kws={"element_type": element_type,
                                          "order": order})
    # nodal displacements
    u = generate_constMatrix(1,b.shape[1], "u")
    return simplify_matrix( b@u )

def disp_grad(ndim: int,
              element_type: str = "Lagrange",
              order: int = 1) -> MatrixFunction:
    """
    Symbolically compute displacement gradient:
        
    H_ij = d u_i / d x_j

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    H : symfem.MatrixFunction
        symbolic displacement gradient of shape (ndim,ndim) .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # small strain matrix
    b = dispgrad_matrix(ndim=ndim,
                        nd_inds=nd_inds,
                        basis=basis,
                        isoparam_kws={"element_type": element_type,
                                      "order": order})
    # nodal displacements
    u = generate_constMatrix(1,b.shape[1], "u")
    return to_square(simplify_matrix( b@u ),order="C")

def def_grad(ndim: int,
             element_type: str = "Lagrange",
             order: int = 1):
    """
    Symbolically compute deformation gradient.
    
    F_ij = delta_ij + d u_i / d x_j

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    F : symfem.MatrixFunction
        symbolic deformation gradient of shape (ndim,ndim) .

    """
    #
    I = eye(size=ndim)
    #
    H = disp_grad(ndim=ndim,
                 element_type=element_type,
                 order=order)
    return I + H

def cauchy_strain(ndim: int,
                  element_type: str = "Lagrange",
                  order: int = 1) ->  MatrixFunction:
    """
    Symbolically compute Cauchy strain
    
    C = F.T @ F

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    C : symfem.MatrixFunction
        symbolic Cauchy strain of shape (ndim,ndim) 

    """
    #
    F = def_grad(ndim=ndim,
                 element_type=element_type,
                 order=order)
    return simplify_matrix(F.transpose()@F)

def finger_strain(ndim: int,
                  element_type: str = "Lagrange",
                  order: int = 1) ->  MatrixFunction:
    """
    Symbolically compute Finger strain tensor
    
    Finger = inv(C)

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    Finger : symfem.MatrixFunction
        symbolic Finger strain of shape (ndim,ndim) 

    """
    #
    C = cauchy_strain(ndim=ndim,
                      element_type=element_type,
                      order=order)
    return simplify_matrix( inverse(C) )

def green_strain(ndim: int,
                 element_type: str = "Lagrange",
                 order: int = 1) ->  MatrixFunction:
    """
    Symbolically compute Green strain tensor
    
    B = F @ F.T

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    B : symfem.MatrixFunction
        symbolic Green strain tensor of shape (ndim,ndim) 

    """
    #
    F = def_grad(ndim=ndim,
                 element_type=element_type,
                 order=order)
    return simplify_matrix(F@F.transpose())
