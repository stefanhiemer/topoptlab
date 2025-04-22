from symfem.symbols import x

from topoptlab.symfem_utils import base_cell, small_strain_matrix, generate_constMatrix
from topoptlab.symfem_utils import convert_to_code, jacobian

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
                            basis=basis)
    # create full integral and multiply with determinant
    integrand = b.transpose()@c@b
    return integrand.integral(ref,x)

if __name__ == "__main__":


    #
    print("1D ",convert_to_code(linelast(ndim = 1),matrices = ["c"]),"\n")
    print("2D ",convert_to_code(linelast(ndim = 2),matrices = ["c"]),"\n")
    #print("3D ",convert_to_code(linelast(ndim = 3),matrices = ["c"]),"\n")
