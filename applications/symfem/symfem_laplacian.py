from itertools import product

import symfem
from symfem.functions import MatrixFunction,VectorFunction
from symfem.symbols import x

from topoptlab.symfem_utils import base_cell ,_generate_constMatrix

def symfem_laplacian(ndim, 
                     element_type="Lagrange",
                     order=1):
    """
    Symbolically compute the stiffness matrix for the Laplacian operator.

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
    vertices, nd_inds, ref, basis  = base_cell(ndim,
                                               element_type=element_type,
                                               order=order)
    # number nodes
    n_nds = len(nd_inds)
    # Create a matrix of zeros with the correct shape
    matrix = [[0 for i in range(n_nds)] for j in range(n_nds)]
    # anisotropic heat conductivity or equivalent
    K = _generate_constMatrix(ndim,ndim,"k")
    #
    gradN = MatrixFunction([VectorFunction([basis[i] for i in nd_inds]).diff(var) 
                            for var in ["x","y","z"][:ndim]])
    #
    integrand = gradN.transpose()@K@gradN
    for i,j in product(range(integrand.shape[0]),
                       range(integrand.shape[1])):
        matrix[i][j] += integrand[i,j].integral(ref, x)
    return matrix

if __name__ == "__main__":
    
    #
    print(symfem_laplacian(ndim=2))