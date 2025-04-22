import numpy as np

def lf_bodyforce_3d(b=np.array([0,0,-1])):
    """
    Create body force for 3D with trilinear hexahedral Lagrangian
    elements.

    Parameters
    ----------
    b : np.ndarray shape (3)
        body force

    Returns
    -------
    Ke : np.ndarray, shape (24,1)
        element stiffness matrix.

    """

    return t*np.array([[b[0]*l1*l2*l3],
                       [b[1]*l1*l2*l3],
                       [b[2]*l1*l2*l3],
                       [b[0]*l1*l2*l3],
                       [b[1]*l1*l2*l3],
                       [b[2]*l1*l2*l3],
                       [b[0]*l1*l2*l3],
                       [b[1]*l1*l2*l3],
                       [b[2]*l1*l2*l3],
                       [b[0]*l1*l2*l3],
                       [b[1]*l1*l2*l3],
                       [b[2]*l1*l2*l3],
                       [b[0]*l1*l2*l3],
                       [b[1]*l1*l2*l3],
                       [b[2]*l1*l2*l3],
                       [b[0]*l1*l2*l3],
                       [b[1]*l1*l2*l3],
                       [b[2]*l1*l2*l3],
                       [b[0]*l1*l2*l3],
                       [b[1]*l1*l2*l3],
                       [b[2]*l1*l2*l3],
                       [b[0]*l1*l2*l3],
                       [b[1]*l1*l2*l3],
                       [b[2]*l1*l2*l3]])
