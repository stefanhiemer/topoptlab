from symfem.functions import MatrixFunction
from symfem.symbols import x
from topoptlab.symfem_utils import base_cell, small_strain_matrix, generate_constMatrix
from topoptlab.symfem_utils import convert_to_code,stifftens_isotropic
from topoptlab.symfem_utils import simplify_matrix

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
    a = MatrixFunction([[1] for i in range(ndim)]+\
                        [[0] for i in range(int((ndim**2 + ndim) /2)-ndim)])
    #
    b = small_strain_matrix(ndim=ndim,
                            nd_inds=nd_inds,
                            basis=basis,
                            isoparam_kws={"element_type": element_type,
                                          "order": order})
    #
    integrand = b.transpose()@c@a@NT
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
    integrand = b.transpose()@c@a@NT
    return integrand.integral(ref,x)

if __name__ == "__main__":
    
    #
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(heatexp_aniso(ndim = dim),
                              matrices=["c"],vectors=["l","g","a"]),"\n")
    import sys 
    sys.exit()
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(heatexp_iso(ndim = dim),
                              matrices=["c"],vectors=["l","g","a"]),"\n")
    
    
    