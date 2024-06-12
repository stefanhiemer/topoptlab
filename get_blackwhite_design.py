from numpy import flip,argsort,floor

def threshold(xPhys, volfrac):
    """
    Threshold grey scale design to black and white design.

    Parameters
    ----------
    xPhys : np.array, shape (nel)
        element densities for topology optimization used for scaling the 
        material properties. 
    volfrac : float
        volume fraction.

    Returns
    -------
    xPhys : np.array, shape (nel)
        thresholded element densities for topology optimization used for scaling the 
        material properties. 

    """
    indices = flip(argsort(xPhys))
    vt = floor(volfrac*xPhys.shape[0]).astype(int)
    xPhys[indices[:vt]] = 1.
    xPhys[indices[vt:]] = 0.
    print("Thresholded Vol.: {0:.3f}".format(vt/xPhys.shape[0]))
    return xPhys
