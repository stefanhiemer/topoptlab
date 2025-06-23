import numpy as np

from topoptlab.fem import get_integrpoints
from topoptlab.elements.bilinear_quadrilateral import shape_functions, jacobian

def vol_integral(xe,
                 t=np.array([1.]),
                 quadr_method="gauss-legendre",
                 nquad=2,
                 **kwargs):
    """
    Calculate volume integral over some nodal variable.

    Parameters
    ----------
    xe : np.ndarray, shape (nels,n_nodes,ndim)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    p : np.ndarray of shape (nels) or (1)
        density of element
    t : np.ndarray of shape (nels) or (1)
        thickness of element. Only relevant in 1D (not yet implemented) or 2D.
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    nquad : int
        number of quadrature points
        
    Returns
    -------
    Ke : np.ndarray, shape (nels,n_nodes,n_nodes)
        element matrix.

    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel,n_nodes,ndim = xe.shape 
    #
    if isinstance(t,float) and ndim == 2:
        t = np.array([t])
    #
    x,w=get_integrpoints(ndim=ndim,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [_x[:,0] for _x in np.split(x, ndim, axis=1)] + [None,None,None][ndim:]
    #
    N = shape_functions(xi=xi,eta=eta,zeta=zeta)
    # calculate determinant of jacobian
    J = jacobian(xi=xi,eta=eta,xe=xe,all_elems=True)
    detJ = ((J[:,0,0]*J[:,1,1]) - (J[:,1,0]*J[:,0,1])).reshape(nel,nq)
    # multiply by determinant and quadrature
    if ndim == 2:
        return t[:,None,None] * (w[None,:,None,None]*N[None,:,:,None]*detJ[:,:,None,None]).sum(axis=1)
    elif ndim == 3:
        return (w[None,:,None,None]*N[None,:,:,None]*detJ[:,:,None,None]).sum(axis=1)