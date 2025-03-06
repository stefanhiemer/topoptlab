from symfem.functions import VectorFunction
from symfem.symbols import x

from topoptlab.symfem_utils import base_cell ,generate_constMatrix

def symfem_isolaplacian(ndim, 
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
    #
    gradN = VectorFunction(basis).grad(ndim)
    #
    integrand = gradN@gradN.transpose()
    return integrand.integral(ref,x)

def symfem_anisolaplacian(ndim, 
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
    # anisotropic heat conductivity or equivalent
    K = generate_constMatrix(ndim,ndim,"k")
    #
    gradN = VectorFunction(basis).grad(ndim)
    #
    integrand = gradN@K@gradN.transpose()
    return integrand.integral(ref,x)

if __name__ == "__main__":
    
    #
    print(symfem_isolaplacian(ndim=2))