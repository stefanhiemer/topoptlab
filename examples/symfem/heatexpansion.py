# SPDX-License-Identifier: GPL-3.0-or-later
from symfem.functions import MatrixFunction
from symfem.symbols import x
from sympy import symbols

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.code_conversion import convert_to_code
from topoptlab.symbolic.matrix_utils import simplify_matrix, generate_constMatrix
from topoptlab.symbolic.parametric_map import jacobian
from topoptlab.symbolic.lin_elastic import stifftens_isotropic, small_strain_matrix


def heatexp_iso(ndim,
                element_type="Lagrange",
                order=1):
    """
    Symbolically compute the stiffness matrix for linear elasticity.

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
    stiffness_matrix : list
        symbolic stiffness matrix as list of lists .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # anisotropic stiffness tensor or equivalent in Voigt notation
    c = stifftens_isotropic(ndim)
    # transpose of shape functions
    NT = MatrixFunction([basis])
    # heat expansion coeff. tensor in Voigt notation
    a = MatrixFunction([[symbols("a")] for i in range(ndim)]+\
                        [[0] for i in range(int((ndim**2 + ndim) /2)-ndim)])
    #
    b = small_strain_matrix(ndim=ndim,
                            nd_inds=nd_inds,
                            basis=basis,
                            isoparam_kws={"element_type": element_type,
                                          "order": order})
    #
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = b.transpose()@c@a@NT * Jdet
    return simplify_matrix(integrand.integral(ref,x))

def heatexp_aniso(ndim,
                  element_type="Lagrange",
                  order=1):
    """
    Symbolically compute the stiffness matrix for linear elasticity.

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
    stiffness_matrix : list
        symbolic stiffness matrix as list of lists .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # anisotropic stiffness tensor or equivalent in Voigt notation
    c = generate_constMatrix(int((ndim**2 + ndim) /2),
                             int((ndim**2 + ndim) /2),
                             "c")
    # transpose of shape functions
    NT = MatrixFunction([basis])
    # heat expansion coeff. tensor in Voigt notation
    a = generate_constMatrix(ncol=1,
                             nrow=int((ndim**2 + ndim) /2),
                             name="a")
    #
    b = small_strain_matrix(ndim=ndim,
                            nd_inds=nd_inds,
                            basis=basis,
                            isoparam_kws={"element_type": element_type,
                                          "order": order})
    #
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = b.transpose()@c@a@NT * Jdet
    return integrand.integral(ref,x)

if __name__ == "__main__":

    #
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(heatexp_iso(ndim = dim),
                              matrices=["c"],vectors=["l","g"]),"\n") 
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(heatexp_aniso(ndim = dim),
                              matrices=["c"],vectors=["l","g","a"]),"\n")
