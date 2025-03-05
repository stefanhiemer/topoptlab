from itertools import product

from sympy import symbols
from symfem.functions import MatrixFunction,VectorFunction
import symfem
from symfem.symbols import x

from topoptlab.symfem_utils import base_cell ,_generate_constMatrix

def bmatrix(ndim,nd_inds,basis):
    nrows = int((ndim**2 + ndim) /2)
    ncols = int(ndim * len(nd_inds))
    # compute gradients of basis functions
    gradN = [list(VectorFunction([basis[i] for i in nd_inds]).diff(var)) 
             for var in ["x","y","z"][:ndim]]
    #print(gradN)
    #
    bmatrix = [[0 for j in range(ncols)] for i in range(nrows)]
    # tension
    for i in range(ndim):
        bmatrix[i][i::ndim] = gradN[i]
    # shear
    i,j = ndim-2,ndim-1
    for k in range(nrows-ndim):
        #
        bmatrix[ndim+k][i::ndim] = gradN[j]
        bmatrix[ndim+k][j::ndim] = gradN[i]
        #
        i,j = (i+1)%ndim , (j-1)%ndim
    return MatrixFunction(bmatrix)

def symfem_mass(ndim,
                  element_type="Lagrange",
                  order=1):
    """
    Symbolically compute the mass matrix.

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
    ndim = 2
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # number nodes
    n_nds = len(nd_inds)
    # Create a matrix of zeros with the correct shape
    matrix = [[0 for i in range(n_nds*ndim)] for j in range(n_nds*ndim)]
    # shape functions (actually the transpose)
    N = MatrixFunction([basis])
    #
    integrand = N.transpose()
    for i,j in product(range(integrand.shape[0]),
                       range(integrand.shape[1])):
        matrix[i][j] += integrand[i,j].integral(ref, x)
        
    return matrix

if __name__ == "__main__":
    
    
    #
    print(symfem_mass(ndim=2))
    
    
    