# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union

from sympy import symbols

from symfem.functions import VectorFunction,MatrixFunction
from symfem.symbols import x

from topoptlab.symbolic.cell import base_cell 
from topoptlab.symbolic.matrix_utils import  generate_constMatrix,\
                                             generate_FunctMatrix,\
                                             simplify_matrix 
from topoptlab.symbolic.parametric_map import jacobian

def aniso_laplacian(ndim: int, K: Union[None,MatrixFunction] = None,
                    element_type: str = "Lagrange",
                    order: int = 1) -> MatrixFunction:
    """
    Symbolically compute the stiffness matrix for an anisotropic Laplacian 
    operator 
    
    nabla @ K @ nabla^T phi_n,
    
    where phi_n are the nodal values of a scalar field phi. This type of 
    operator is encountered in heat conduction, diffusion, etc.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    K : None or symfem.functions.MatrixFunction
        heat conductivity tensor.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    stiffness_matrix : symfem.functions.MatrixFunction
        symbolic stiffness matrix as list of lists .

    """

    #
    vertices, nd_inds, ref, basis  = base_cell(ndim,
                                               element_type=element_type,
                                               order=order)
    # anisotropic heat conductivity or equivalent material property
    if K is None:
        K = generate_constMatrix(col=ndim,row=ndim,
                                 name="k",
                                 symmetric=True)
    #
    Jinv, Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                          return_J=False, return_inv=True, return_det=True)
    gradN = VectorFunction(basis).grad(ndim)@Jinv.transpose()
    #
    integrand = gradN@K@gradN.transpose() * Jdet
    return simplify_matrix( integrand.integral(ref,x)) 

def nonlin_laplacian(ndim: int, 
                     K: Union[None,MatrixFunction] = None,
                     linearization="picard",
                     element_type: str = "Lagrange",
                     order: int = 1) -> MatrixFunction:
    """
    Symbolically compute the (tangent) conductivity matrix for the (informal) 
    an anisotropic nonlinear Laplacian operator at point phi_0
    
    nabla @ K(phi) @ nabla phi_n,
    
    where phi_n are the nodal values of a scalar field phi. This is not(!) the 
    Laplace operator in the strict mathematical sense, but arises when 
    equations, that are usually modelled with constant material properties, 
    incorporate nonlinearities of the material property with regards to the 
    state variable. This type of operator is encountered e. g. in heat 
    conduction and diffusion with the heat conductivity/diffusion coefficient 
    depending on the scalar field (temperature, concentration) itself.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    K : None or symfem.functions.MatrixFunction
        heat conductivity tensor as function of phi.
    linearization : str
        type of linearization. Either "picard" or "newton".
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    stiffness_matrix : symfem.functions.MatrixFunction
        symbolic stiffness matrix as list of lists .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim,
                                               element_type=element_type,
                                               order=order)
    #
    phi0 = generate_constMatrix(len(basis), nrow=1, name="phi0")
    # anisotropic heat conductivity or equivalent nonlinear material property
    if K is None:
        K = generate_FunctMatrix(ncol=ndim,nrow=ndim,
                                 name="k",
                                 variables=[symbols("phi")],
                                 symmetric=True)
    print(K.subs( VectorFunction(basis).dot(phi0) ))
    import sys 
    sys.exit()
    #
    Jinv, Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                          return_J=False, return_inv=True, return_det=True)
    gradN = VectorFunction(basis).grad(ndim)@Jinv.transpose()
    #
    integrand = gradN@K@gradN.transpose()
    return simplify_matrix( (integrand* Jdet).integral(ref,x)) 

if __name__ == "__main__":
    nonlin_laplacian(ndim=1)