from itertools import product

from symfem.functions import MatrixFunction
from symfem.symbols import x
from topoptlab.symfem_utils import base_cell, bmatrix, _generate_constMatrix

def symfem_heatexp(ndim,
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
    # number nodes
    n_nds = len(nd_inds)
    # Create a matrix of zeros with the correct shape
    matrix = [[0 for i in range(n_nds*ndim)] for j in range(n_nds*ndim)]
    # anisotropic stiffness tensor or equivalent in Voigt notation
    c = _generate_constMatrix(int((ndim**2 + ndim) /2),
                              int((ndim**2 + ndim) /2),
                              "c")
    # shape functions
    N = MatrixFunction([basis])
    # heat expansion coeff. tensor in Voigt notation
    a = _generate_constMatrix(ncol=1,
                              nrow=int((ndim**2 + ndim) /2),
                              name="a")
    #
    b = bmatrix(ndim=ndim,
                nd_inds=nd_inds,
                basis=basis)
    #
    integrand = b.transpose()@c@a@N
    for i,j in product(range(integrand.shape[0]),
                       range(integrand.shape[1])):
        matrix[i][j] += integrand[i,j].integral(ref, x)
        
    return matrix

if __name__ == "__main__":
    
    
    #
    print(symfem_heatexp(ndim=2))
    
    
    