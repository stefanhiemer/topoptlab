# SPDX-License-Identifier: GPL-3.0-or-later
from sympy import simplify
from symfem.functions import MatrixFunction


from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.matrix_utils import simplify_matrix, generate_constMatrix
from topoptlab.symbolic.lin_elastic import small_strain_matrix,dispgrad_matrix
from topoptlab.symbolic.voigt import convert_from_voigt

def engstrain(ndim,
              element_type="Lagrange",
              order=1):
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
    nodal_forces : list
        symbolic stiffness matrix as list of lists .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    print(basis)
    # small strain matrix
    b = small_strain_matrix(ndim=ndim,
                            nd_inds=nd_inds,
                            basis=basis,
                            isoparam_kws={"element_type": element_type,
                                          "order": order})
    # nodal displacements
    u = generate_constMatrix(1,b.shape[1], "u")
    return simplify_matrix( b@u )

def dispgrad(ndim,
             element_type="Lagrange",
             order=1):
    """
    Symbolically compute displacement gradient.

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
    nodal_forces : list
        symbolic stiffness matrix as list of lists .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    print(basis)
    # small strain matrix
    b = dispgrad_matrix(ndim=ndim,
                        nd_inds=nd_inds,
                        basis=basis,
                        isoparam_kws={"element_type": element_type,
                                      "order": order})
    # nodal displacements
    u = generate_constMatrix(1,b.shape[1], "u")
    return convert_from_voigt(simplify_matrix( b@u ))

def defgrad(ndim,
            element_type="Lagrange",
            order=1):
    """
    Symbolically compute deformation gradient.

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
    nodal_forces : list
        symbolic stiffness matrix as list of lists .

    """
    #
    eye = MatrixFunction([[0 if i!=j else 1 for j in range(ndim)] \
                          for i in range(ndim)])
    #
    H = dispgrad(ndim,
                 element_type="Lagrange",
                 order=1)
    return eye + H


if __name__ == "__main__":
    #
    ndim = 2
    #
    eps = convert_from_voigt(engstrain(ndim = ndim))
    print("engineering strain: ", eps)
    #
    H = dispgrad(ndim = ndim)
    print("displacement gradient: ", H)
    #
    F = defgrad(ndim = ndim)
    print("deformation gradient: ", F)
    #
    print()
    #
    test =  F + F.transpose() 
    eye = MatrixFunction([[0 if i!=j else 1 for j in range(ndim)] \
                           for i in range(ndim)])
    for i in range(ndim):
        for j in range(ndim):
            print(i,j)
            print(eps[i,j].as_sympy())
            print( simplify((H[i,j].as_sympy() + H[j,i].as_sympy()) / 2) )
            print()
