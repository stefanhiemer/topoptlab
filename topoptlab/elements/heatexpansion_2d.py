import numpy as np 

from topoptlab.elements.bilinear_quadrilateral import shape_functions,bmatrix
from topoptlab.fem import get_integrpoints

def _fk_linear_heatexp_2d(xe,c,
                          alpha,T,Tref,
                          quadr_method="gauss-legendre",
                          nquad = 2):
    """
    Create force vector for 2D linear heat expansion with 
    bilinear quadrilateral Lagrangian elements.
    
    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the 
        definition/function of the shape function, then the node ordering is 
        clear.
    c : np.ndarray, shape (nels,3,3) or 
        stiffness tensor.
    alpha : np.ndarray, shape (nels,2,2) or 
        linear heat expansion tensor.
    T : np.ndarray shape (nels,4)
        nodal temperatures
    Tref : float
        reference temperature.
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of 
        quadrature points and weights. Check function get_integrpoints for 
        available options. 
    nquad : int
        number of quadrature points
    Returns
    -------
    ft : np.ndarray, shape (nels,8)
        force due to thermal expansion.
        
    """
    #raise NotImplementedError("Not yet finished")
    # convert linear heat expansion tensor to Voigt notation (nel,3,1)
    if len(alpha.shape) == 2:
        alpha = alpha[None,:,:]
    alpha = alpha[:,[0,1,0],[0,1,1],None]
    #
    if len(c.shape) == 2:
        c = c[None,:,:]
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    #
    nel = xe.shape[0]
    nq =w.shape[0]
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    # shape functions at integration points
    N = shape_functions(xi,eta)[None,:,:,None]
    # delta T
    deltaT = (T-Tref)
    # 
    B = bmatrix(xi, eta, xe, all_elems=True)
    B = B.reshape(nel, nq,  B.shape[-2], B.shape[-1])
    #
    #print("B_T ", B.transpose([0,1,3,2]).shape)
    #print("c ", c.shape)
    integral = B.transpose([0,1,3,2])@c[:,None,:,:]
    #print("prelim integral ",integral.shape)
    integral = integral@alpha[:,None,:,:]
    #print(N.shape)
    integral = integral@N.transpose([0,1,3,2])
    #print(deltaT.shape)
    #print("prelast integral ",integral.shape)
    integral = integral@deltaT[:,None,:,None]
    #
    fe = (w[:,None,None]*integral).sum(axis=1)
    #print("fe ", fe.shape)
    #print(fe)
    return fe

if __name__ == "__main__":
    
    from topoptlab.elements.linear_elasticity_2d import _lk_linear_elast_2d
    from topoptlab.stiffness_tensors import isotropic_2d
    xe = np.array([ [[-1.,-1.], 
                     [1.,-1.], 
                     [1.,1.], 
                     [-1.,1.]]])
    c = isotropic_2d(E=1.,nu=0.3)
    _lk_linear_elast_2d(xe,c)
    _fk_linear_heatexp_2d(xe=xe, 
                          c = c, 
                          alpha = np.eye(2)[None,:,:], 
                          T=np.array([[1.,1.,1.,1.]]),
                          Tref=0.)
    