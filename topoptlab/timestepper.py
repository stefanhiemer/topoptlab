import numpy as np

def bdf_coefficients(k: int) -> np.ndarray:
    """
    Get coefficients for backward differentiation formula BDF. Shamelessly 
    copied from Elmer Solver manual.´and also check the formula there to understand
    what the coefficients mean.

    Parameters
    ----------
    k : int
        order of BD. must be equal or smaller 6.

    Returns
    -------
    coefficients : np.ndarray shape (k+1)
        First one is multiplied with the forces and the stiffness matrix, the others on
        the right hand side with the mass matrix and the history of the function.
    """
    if k > 6:
        raise NotImplementedError("Not implemented for order higher than 6.")
    return np.array([[1,1],
                    [2/3,4/3,-1/3],
                    [6/11,18/11,-9/11,2/11],
                    [12/25,48/25,-36/25,16/25,-3/25],
                    [60/137,300/137,-300/137,200/137,-75/137,12/137],
                    [60/147,360/147,-450/147,400/147,-225/147,72/147,-10/147]])