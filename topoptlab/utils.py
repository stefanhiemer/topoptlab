import numpy as np

def rotation_matrix(theta):
    """
    2D rotation matrix

    Parameters
    ----------
    theta : np.ndarray, shape (n)
        angle in radian.

    Returns
    -------
    R : np.ndarray shape (n,2,2)
        rotation matrices.

    """
    return np.column_stack((np.cos(theta),-np.sin(theta),
                            np.sin(theta),np.cos(theta)))\
          .reshape((theta.shape[0],2,2))