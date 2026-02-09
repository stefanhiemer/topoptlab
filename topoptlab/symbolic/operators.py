# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union

from sympy import symbols
from symfem.functions import VectorFunction,MatrixFunction
from symfem.symbols import x

from topoptlab.symbolic.cell import base_cell 
from topoptlab.symbolic.matrix_utils import generate_constMatrix,\
                                            generate_FunctMatrix,\
                                            simplify_matrix,kron,eye,\
                                            from_vectorfunction,flatten,inverse
from topoptlab.symbolic.parametric_map import jacobian

def aniso_laplacian(ndim: int, K: Union[None,MatrixFunction] = None,
                    element_type: str = "Lagrange",
                    order: int = 1) -> MatrixFunction:
    """
    Symbolically compute the stiffness matrix for an anisotropic Laplacian 
    operator 
    
    nabla^T @ K(phi) @ nabla phi,
    
    of a scalar field phi. This type of operator is encountered in heat 
    conduction, diffusion, etc.

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
        K = generate_constMatrix(ncol=ndim,nrow=ndim,
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
                     linearization="newton",
                     element_type: str = "Lagrange",
                     order: int = 1) -> MatrixFunction:
    """
    Symbolically compute the (tangent) conductivity matrix for the (informal) 
    an anisotropic nonlinear Laplacian operator at point phi_0
    
    nabla @ K(phi) @ nabla^T phi,
    
    of a scalar field phi. This is not(!) the Laplace operator in the strict 
    mathematical sense, but arises when equations, that are usually modelled 
    with constant material properties, incorporate nonlinearities of the 
    material property with regards to the state variable. This type of 
    operator is encountered e. g. in heat conduction and diffusion with the 
    heat conductivity/diffusion coefficient depending on the scalar field 
    (temperature, concentration) itself.

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
    phi0 = generate_constMatrix(ncol=1, nrow=len(basis), name="phi0")
    # anisotropic heat conductivity or equivalent nonlinear material property
    if K is None:
        K = generate_FunctMatrix(ncol=ndim,nrow=ndim,
                                 name="k",
                                 variables=[symbols("phi")],
                                 symmetric=True)
    # collect things related to isoparametric mapping
    Jinv, Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                          return_J=False, return_inv=True, return_det=True)
    
    #
    N = VectorFunction(basis)
    gradN = N.grad(ndim)@Jinv.transpose()
    # interpolate the node points of phi0
    phi0_interpol = (phi0.transpose()@N)[0]
    # this term occurs both in picard and newton iterations
    integrand = gradN@K@gradN.transpose()
    if linearization == "newton":
        #
        N = from_vectorfunction(N)
        integrand = integrand.subs("phi",values=phi0_interpol) + \
                    integrand.diff(variable="phi")@phi0@N.transpose()
        integrand = integrand.subs("phi",values=phi0_interpol)
    elif linearization == "picard":
        integrand = integrand.subs("phi",values=phi0_interpol)
    return simplify_matrix( (integrand* Jdet).integral(ref,x))

def hessian_matrix(scalarfield: bool, 
                   ndim: int, 
                   integrate: bool = False,
                   element_type: str = "Lagrange",
                   order: int = 1) -> MatrixFunction:
    """
    Create helper matrix to create the flattened Hessian via matrix-vector 
    multiplication of the nodal displacements u_e:
        
        hessian = B_hessian@u_e
        
    Recover the Hessian H via reshaping by hessian.reshape((...,ndim,ndim))
    
    Parameters
    ----------
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords). Coordinates 
        are assumed to be in the reference domain.
    eta : None or np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates 
        are assumed to be in the reference domain. 
    zeta : None or np.ndarray
        z coordinate in the reference domain of shape (ncoords). Coordinates 
        are assumed to be in the reference domain.
    xe : np.ndarray
        coordinates of element nodes shape (nels,n_nodes,ndim). nels must be 
        either 1, ncoords/4 or the same as ncoords. The two exceptions are if 
        ncoords = 1 or all_elems is True. 
        Please look at the definition/function of the shape function, then the 
        node ordering is clear.
    shape_functions_dxi: callable
        function to calculate the gradient of the shape functions.
    shape_functions_hessian: callable
        function to calculate hessian of shape functions per shape 
        function/node at specified coordinate(s).
    invjacobian : callable or np.ndarray
        function to calculate the inverse jacobian for the isoparametric 
        mapping.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for 
        creating elements etc.
    return_detJ : bool
        if True, return determinant of jacobian.
    check_fnc : callable
        function that checks for type and shape consistency of the inputs.
        
    Returns
    -------
    B_hessian : np.ndarray, shape (nels,ndim**3,nnodes*ndim)
        element stiffness matrix.

    """
    # anisotropic heat conductivity or equivalent nonlinear material property
    # collect things related to isoparametric mapping
    if element_type:
        vertices, nd_inds, ref, basis  = base_cell(ndim,
                                                   element_type=element_type,
                                                   order=order)
        Jinv, Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                              return_J=False, return_inv=True, return_det=True)
    else:
        basis = generate_FunctMatrix(ncol=1, nrow=2**ndim, 
                                     variables=list(symbols("x y z")[:ndim]),
                                     name="N")
        basis = [basis[i,0] for i in range(basis.shape[0])]
        J = generate_constMatrix(ncol=ndim, nrow=ndim, name="J")
        Jinv = inverse(J)
        Jdet = J.det()
    #
    n = len(basis)
    #
    N = VectorFunction(basis)
    gradN = N.grad(ndim)@Jinv.transpose()
    #
    hessian = flatten(flatten(gradN,order="C").grad(ndim)@Jinv.transpose())
    #
    hessian = [[hessian[col*n+row] for col in range(n)] for row in \
                range(ndim**2) ]
    hessian = MatrixFunction(hessian)*Jdet
    #
    if element_type is not None and integrate:
        hessian = hessian.integral(ref,x) 
    print("before simplification")
    #
    if not scalarfield:
        return kron( simplify_matrix(hessian),eye(size=ndim))
    else:
        return simplify_matrix( hessian )