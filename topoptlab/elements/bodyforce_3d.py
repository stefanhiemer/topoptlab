import numpy as np

from topoptlab.elements.trilinear_hexahedron import shape_functions,jacobian
from topoptlab.fem import get_integrpoints

def _lf_bodyforce_3d(xe,
                     b=np.array([0,-1.,0.]),
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
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    nquad : int
        number of quadrature points
        
    Returns
    -------
    Ke : np.ndarray, shape (nels,24,1)
        element stiffness matrix.

    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel = xe.shape[0]
    #
    if (len(b.shape) == 1) or (b.shape[0] == 1):
        b = np.full((xe.shape[0],3), b)
    #
    x,w=get_integrpoints(ndim=3,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [_x[:,0] for _x in np.split(x, 3,axis=1)]
    #
    shpfcts = shape_functions(xi=xi,eta=eta,zeta=zeta)
    N = np.zeros((xi.shape[0],3*shpfcts.shape[0],3))
    for i in np.arange(3):
        N[:,i::3,i] = shpfcts
    #
    integral = N[None,:,:,:] @ b[:,None,None,:].transpose(0,1,3,2)
    # calculate determinant of jacobian
    J = jacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,all_elems=True)
    detJ = (J[:,0,0]*(J[:,1,1]*J[:,2,2] - J[:,1,2]*J[:,2,1])-
            J[:,0,1]*(J[:,1,0]*J[:,2,2] - J[:,1,2]*J[:,2,0])+
            J[:,0,2]*(J[:,1,0]*J[:,2,1] - J[:,1,1]*J[:,2,0])).reshape(nel,nq)
    # multiply by determinant and quadrature
    fe = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    #
    return fe

def lf_bodyforce_3d(b=np.array([0.,-1.,0.]), 
                    l=np.array([1.,1.,1.])):
    """
    Create body force for 3D with trilinear hexahedral Lagrangian
    elements.

    Parameters
    ----------
    b : np.ndarray shape (3)
        body force
    l : np.ndarray (3)
        side length of element

    Returns
    -------
    Ke : np.ndarray, shape (24,1)
        element stiffness matrix.

    """
    v = l[0]*l[1]*l[2]
    return np.array([[b[0]*v],
                     [b[1]*v],
                     [b[2]*v],
                     [b[0]*v],
                     [b[1]*v],
                     [b[2]*v],
                     [b[0]*v],
                     [b[1]*v],
                     [b[2]*v],
                     [b[0]*v],
                     [b[1]*v],
                     [b[2]*v],
                     [b[0]*v],
                     [b[1]*v],
                     [b[2]*v],
                     [b[0]*v],
                     [b[1]*v],
                     [b[2]*v],
                     [b[0]*v],
                     [b[1]*v],
                     [b[2]*v],
                     [b[0]*v],
                     [b[1]*v],
                     [b[2]*v]]) 
