import numpy as np

def _lf_bodyforce_2d(xe,
                     p=np.array([1.]),
                     t=np.array([1.]),
                     quadr_method="gauss-legendre",
                     nquad=2):
    """
    Create element mass matrix in 2D with bilinear quadrilateral elements.

    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    b : np.ndarray of shape (nels,2) or (2)
        density of element
    t : np.ndarray of shape (nels) or (1)
        thickness of element
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    nquad : int
        number of quadrature points
    Returns
    -------
    Ke : np.ndarray, shape (nels,4,4)
        element stiffness matrix.

    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel = xe.shape[0]
    #
    if (len(b.shape[0]) == 1 and b.shape[0] == 2 and xe.shape[0] !=1):
        b = np.full((xe.shape[0],2), b)
    #
    if isinstance(t,float):
        t = np.array([t])
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    #
    N = shape_functions(xi=xi,eta=eta)
    #
    integral = N[None,:,:,None]@N[None,:,:,None].transpose([0,1,3,2])
    # calculate determinant of jacobian
    J = jacobian(xi=xi,eta=eta,xe=xe,all_elems=True)
    detJ = ((J[:,0,0]*J[:,1,1]) - (J[:,1,0]*J[:,0,1])).reshape(nel,nquad*nquad)
    # multiply by determinant and quadrature
    Ke = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    #
    return t[:,None,None] * p[:,None,None] * Ke

def lf_bodyforce_2d(b=np.array([0,-1]).,t=1.):
    """
    Create body force for 2D with bilinear quadrilateral Lagrangian
    elements.

    Parameters
    ----------
    b : np.ndarray shape (2)
        body force
    t : float
        thickness of element

    Returns
    -------
    Ke : np.ndarray, shape (8,1)
        element stiffness matrix.

    """

    return t*np.array([[b[0]*l1*l2],
                       [b[1]*l1*l2],
                       [b[0]*l1*l2],
                       [b[1]*l1*l2],
                       [b[0]*l1*l2],
                       [b[1]*l1*l2],
                       [b[0]*l1*l2],
                       [b[1]*l1*l2]])
