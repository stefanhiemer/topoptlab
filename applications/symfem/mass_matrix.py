from symfem.functions import MatrixFunction
from symfem.symbols import x

from topoptlab.symfem_utils import base_cell

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
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # get shape functions as a column vector/matrix
    N = MatrixFunction([[basis[i]] for i in nd_inds])
    print(N)
    #
    integrand = N@N.transpose()
    print(integrand[0,0])
    return integrand.integral(ref,x)

if __name__ == "__main__":
    
    
    #
    print(symfem_mass(ndim=2))
    
    
    