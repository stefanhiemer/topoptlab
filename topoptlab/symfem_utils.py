import numpy as np

from sympy import symbols
from symfem.functions import VectorFunction


def generate_constMatrix(ncol,nrow,name):
    """
    Generate matrix full of symbolic constants as a list of lists

    Parameters
    ----------
    ncol : int
        number of rows.
    nrow : int
        number of cols.
    name : str
        name to put in front of indices.

    Returns
    -------
    M : list
        list of list e. g. for a 2x2 [[M11,M12],[M21,M22]] 

    """
    M = []
    for i in range(1,nrow+1):
        
        M.append(list( symbols( \
                " ".join([name+str(i)+str(j) for j in range(1,ncol+1)]) ) ) )
    
    return M

def multiply_vecT_symmat(vT,M):
    """
    Multiply tranposed vector function vT to matrix M: vT@M

    Parameters
    ----------
    vT : symfem.functions.VectorFunction
        tranposed vector function.
    M : list
        list of list, that mimicks a matrix.

    Returns
    -------
    params : dict
        contains some of the parameters like system size and shape, 
        optimizer etc..
    data : np.ndarray
        iteration history over objective function, volume constraint, change.

    """
    
    #
    ndim = len(M)
    #
    result = []
    for i in range(ndim):
        result.append(vT.dot(VectorFunction([M[j][i] for j in range(ndim)])))
    return VectorFunction(result)