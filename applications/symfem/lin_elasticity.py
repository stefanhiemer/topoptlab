from symfem.symbols import x

from topoptlab.symfem_utils import base_cell, small_strain_matrix, generate_constMatrix
from topoptlab.symfem_utils import convert_to_code, jacobian, simplify_matrix

def linelast(ndim,
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

if __name__ == "__main__":
    
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(linelast(ndim = dim),
                              matrices=["c"],vectors=["l","g"]),"\n")