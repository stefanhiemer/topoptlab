# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List, Tuple

from symfem.functions import VectorFunction, MatrixFunction

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.matrix_utils import generate_constMatrix, simplify_matrix

def scale_cell(vertices: Tuple) -> MatrixFunction:
    """
    Scale/rotate the vertices/nodes basic cell by lengths l and angles g.

    Parameters
    ----------
    vertices : tuple
        coordinates of vertices as created by base cell

    Returns
    -------
    vertices_new : symfem.functions.MatrixFunction, shape (ndim,1)
        coordinate according to ispoarametric map
    """
    from sympy.functions.elementary.trigonometric import tan
    #
    if isinstance(vertices, tuple):
        vertices = MatrixFunction(vertices)
    #
    ndim = vertices.shape[1]
    # rotation angles
    g = generate_constMatrix(ncol=1,nrow=ndim-1,name="g")
    # create rotation matrix
    R = [ [0 for j in range(ndim)] for i in range(ndim)]
    for i in range(ndim):
        R[i][i] = 1
    for i in range(ndim-1):
        R[0][i+1] = tan(g[i][0])
    R = MatrixFunction(R)
    # cell lengths
    l = generate_constMatrix(ncol=1,nrow=ndim,name="l")
    # create stretch matrix
    S = [ [0 for j in range(ndim)] for i in range(ndim)]
    for i in range(ndim):
        S[i][i] = l[i][0]
    S = MatrixFunction(S)/2
    # affine transformation matrix
    return vertices@(R@S).transpose()

def isoparametric_map(basis: List, vertices: Tuple) -> MatrixFunction:
    """
    Create the basic cell, location of vertices, the node indices, the
    reference cell and the basis functions.

    Parameters
    ----------
    basis : list
        list of basis functions.
    vertices : tuple
        coordinates of vertices as created by base cell

    Returns
    -------
    coord : symfem.functions.MatrixFunction, shape (ndim,1)
        coordinate according to ispoarametric map
    """
    #
    if isinstance(vertices, tuple):
        vertices = MatrixFunction(vertices)
    # get number of nodes and convert to Matrix function
    if isinstance(basis, list):
        basis = MatrixFunction([[b] for b in basis])
    elif isinstance(basis, (VectorFunction)):
        basis = MatrixFunction([[b] for b in basis])
    elif isinstance(basis,MatrixFunction):
        if basis.shape[1] != 1:
            raise ValueError("If basis is provided as MatrixFunction, must have shape (n_nodes,1)")
    return vertices.transpose()@basis

def jacobian(ndim: int,
             element_type: str = "Lagrange",
             order: int = 1,
             return_J: bool = True,
             return_inv: bool = True,
             return_det: bool = True,
             debug: bool = False) -> Tuple[MatrixFunction, ...]:
    """
    Symbolically compute the Jacobian of the parametric mapping.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.
    return_J : bool
        if True, return Jacobian matrix.
    return_inv : bool
        if True, return inverse of Jacobian matrix.
    return_det : bool
        if True, return determinant of Jacobian matrix.
    debug : bool
        if True, print additional information.

    Returns
    -------
    J : symfem.functions.MatrixFunction
        jacobian of isoparametric mapping.
    Jinv : symfem.functions.MatrixFunction
        inverse of jacobian of isoparametric mapping.
    Jdet : symfem.functions.MatrixFunction
        determinant of jacobian of isoparametric mapping.

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim=ndim,
                                               element_type=element_type,
                                               order=order)
    #
    gradN = VectorFunction(basis).grad(ndim)
    #
    scaled = scale_cell(vertices)
    if debug:
        print("cell in physical space:\n",scaled)
        print("basis functions in reference space:\n",basis)
        print("gradient of basis functions in reference space:\n",gradN)
    #
    J = simplify_matrix( gradN.transpose()@scaled )
    if return_det or return_inv:
        Jdet = J.det()
    if return_inv:
        # adjungate matrix
        Jinv = [[[] for j in range(ndim)] for j in range(ndim)]
        if ndim == 1:
            Jinv[0][0] = 1 / Jdet
        elif ndim == 2:
            Jinv[0][0], Jinv[1][1] = J[1][1]/Jdet, J[0][0]/Jdet
            Jinv[0][1], Jinv[1][0] = -J[0][1]/Jdet, -J[1][0]/Jdet
        elif ndim == 3:
            #
            Jinv[0][0] = (J[1][1]*J[2][2] - J[1][2]*J[2][1]) / Jdet
            Jinv[0][1] = -(J[0][1]*J[2][2] - J[0][2]*J[2][1]) / Jdet
            Jinv[0][2] = (J[0][1]*J[1][2] - J[0][2]*J[1][1]) / Jdet
            #
            Jinv[1][0] = -(J[1][0]*J[2][2] - J[1][2]*J[2][0]) / Jdet
            Jinv[1][1] = (J[0][0]*J[2][2] - J[0][2]*J[2][0] ) / Jdet
            Jinv[1][2] = -(J[0][0]*J[1][2] - J[0][2]*J[1][0]) / Jdet
            #
            Jinv[2][0] = (J[1][0]*J[2][1] - J[1][1]*J[2][0]) / Jdet
            Jinv[2][1] = -(J[0][0]*J[2][1] - J[0][1]*J[2][0]) / Jdet
            Jinv[2][2] = (J[0][0]*J[1][1] - J[0][1]*J[1][0]) / Jdet
        #
        Jinv = simplify_matrix( MatrixFunction(Jinv) )
    if all([return_J, not return_inv, not return_det]):
        return J
    elif all([not return_J, return_inv, not return_det]):
        return Jinv
    elif all([not return_J, not return_inv, return_det]):
        return Jdet
    elif all([return_J, return_inv, not return_det]):
        return J, Jinv
    elif all([return_J, not return_inv, return_det]):
        return J, Jdet
    elif all([not return_J, return_inv, return_det]):
        return Jinv, Jdet
    elif all([return_J,return_inv,return_det]):
        return J, Jinv, Jdet
    else:
        raise ValueError("At least on of the return options must be True.")
