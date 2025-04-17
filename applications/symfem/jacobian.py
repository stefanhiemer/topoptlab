from symfem.functions import MatrixFunction
from symfem.symbols import x

from topoptlab.symfem_utils import base_cell,scale_cell
from topoptlab.symfem_utils import convert_to_code

def mass(ndim,
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
    #
    scaled = scale_cell(vertices)
    print(scaled)
    import sys 
    sys.exit()
    # get shape functions as a column vector/matrix
    N = MatrixFunction([[b] for b in basis])
    #
    integrand = N@N.transpose()
    return integrand.integral(ref,x)

if __name__ == "__main__":
    
    
    #
    print(convert_to_code(mass(ndim = 2)))