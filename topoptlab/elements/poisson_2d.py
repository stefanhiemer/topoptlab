import numpy as np

def lk_poisson_2d():
    """
    Create element stiffness matrix for 2D Poisson with bilinear
    quadrilateral elements. Taken from the standard Sigmund textbook.
    
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    return np.array([[2/3, -1/6, -1/3, -1/6],
                     [-1/6, 2/3, -1/6, -1/3],
                     [-1/3, -1/6, 2/3, -1/6],
                     [-1/6, -1/3, -1/6, 2/3]])

def lk_poisson_aniso_2d(k):
    """
    Create element stiffness matrix for anisotropic 2D Poisson with bilinear
    quadrilateral elements. Taken from the standard Sigmund textbook.
    
    Parameters
    ----------
    k : np.ndarray, shape (2,2)
        anisotropic heat conductivity. If isotropic k would be [[k,0],[0,k]]
        
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    Ke = np.array([[k[0,0]/3 + k[0,1]/4 + k[1,0]/4 + k[1,1]/3, 
                    -k[0,0]/3 + k[0,1]/4 - k[1,0]/4 + k[1,1]/6, 
                    -k[0,0]/6 - k[0,1]/4 - k[1,0]/4 - k[1,1]/6, 
                    k[0,0]/6 - k[0,1]/4 + k[1,0]/4 - k[1,1]/3], 
                   [-k[0,0]/3 - k[0,1]/4 + k[1,0]/4 + k[1,1]/6, 
                    k[0,0]/3 - k[0,1]/4 - k[1,0]/4 + k[1,1]/3, 
                    k[0,0]/6 + k[0,1]/4 - k[1,0]/4 - k[1,1]/3, 
                    -k[0,0]/6 + k[0,1]/4 + k[1,0]/4 - k[1,1]/6], 
                   [-k[0,0]/6 - k[0,1]/4 - k[1,0]/4 - k[1,1]/6, 
                    k[0,0]/6 - k[0,1]/4 + k[1,0]/4 - k[1,1]/3, 
                    k[0,0]/3 + k[0,1]/4 + k[1,0]/4 + k[1,1]/3, 
                    -k[0,0]/3 + k[0,1]/4 - k[1,0]/4 + k[1,1]/6], 
                   [k[0,0]/6 + k[0,1]/4 - k[1,0]/4 - k[1,1]/3, 
                    -k[0,0]/6 + k[0,1]/4 + k[1,0]/4 - k[1,1]/6, 
                    -k[0,0]/3 - k[0,1]/4 + k[1,0]/4 + k[1,1]/6, 
                    k[0,0]/3 - k[0,1]/4 - k[1,0]/4 + k[1,1]/3]])
    return Ke