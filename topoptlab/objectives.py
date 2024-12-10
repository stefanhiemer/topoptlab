import numpy as np

from scipy.sparse.linalg import spsolve

def compliance(xPhys,u,KE,edofMat,
               Amax,Amin,penal,
               obj,dc):
    """
    Update objective and gradient for stiffness maximization / compliance 
    minimization. The mechanic version of this is the compliant mechanism  with 
    maximized displacement.

    Parameters
    ----------
    xPhys : np.array
        SIMP densities of shape (nel).
    u : np.array
        state variable (displacement, temperature) of shape (ndof).
    KE : np.array
        element stiffness matrix of shape (nedof).
    edofMat : np.array
        element degree of freedom matrix of shape (nel,nedof)
    Amax : float
        maximum value for material property A
    Amin : float
        minimum value for material property A. Should be small compared to Amax
        but not zero and Amax + Amin should recover the property A of the material
    penal: float
        penalty exponent for the SIMP method.
    obj : float
        objective function.
    dc : np.ndarray
        sensitivities/gradients of design variables with regards to objective 
        function. shape (ndesign)

    Returns
    -------
    obj : float
        updated objective function.
    dc : np.ndarray
        updated sensitivities/gradients of design variables with regards to 
        objective function. shape (ndesign)

    """
    ce = (np.dot(u[edofMat], KE)
             * u[edofMat]).sum(1)
    obj += ((Amin+xPhys**penal*(Amax-Amin))*ce).sum()
    dc[:] -= penal*xPhys**(penal-1)*(Amax-Amin)*ce
    return obj, dc

def var_maximization(xPhys,u,l,free,inds_out,
                     K,KE,edofMat,
                     Amax,Amin,penal,
                     obj,dc,f0=None):
    """
    Update objective and gradient for maximization of state variable in 
    specified points. The mechanic version of this is the compliant mechanism 
    with maximized displacement.

    Parameters
    ----------
    xPhys : np.array
        SIMP densities of shape (nel).
    u : np.array
        state variable (displacement, temperature) of shape (ndof).
    l : np.array
        indicator vector for state variable of shape (ndof). Is 1 at output 
        nodes.
    free : np.array
        indices of free nodes.
    inds_out : np.array
        indices of nodes where the displacement is to be maximized. shape (nout)
    K : scipy.sparse matrix/array
        global stiffness matrix of shape (ndof).
    KE : np.array
        element stiffness matrix of shape (nedof).
    edofMat : np.array
        element degree of freedom matrix of shape (nel,nedof)
    Amax : float
        maximum value for material property A
    Amin : float
        minimum value for material property A. Should be small compared to Amax
        but not zero and Amax + Amin should recover the property A of the material
    penal: float
        penalty exponent for the SIMP method.
    obj : float
        objective function.
    dc : np.ndarray
        sensitivities/gradients of design variables with regards to objective 
        function. shape (ndesign)
    f0: np.array
        if system is subjected to an affine transformation causing the loads, 
        this is the resulting load for an element of density one. shape (nedof)

    Returns
    -------
    obj : float
        updated objective function.
    dc : np.ndarray
        updated sensitivities/gradients of design variables with regards to 
        objective function. shape (ndesign)

    """
    obj += u[inds_out].sum()
    # solve adjoint problem
    h = np.zeros(l.shape)
    h[free] = spsolve(K, -l[free])
    #
    if f0 is None:
        dc[:] += penal*xPhys**(penal-1)*(Amax-Amin)*(np.dot(h[edofMat], KE)*\
                 u[edofMat]).sum(1)
    else:
        dc[:] += penal*xPhys**(penal-1)*(Amax-Amin)*(np.dot(h, KE)*\
                 (u[edofMat,0]-f0[None,:])).sum(1)
    return obj, dc

def var_squarederror(xPhys,u,u0,l,free,inds_contr,
                     K,KE,edofMat,
                     Amax,Amin,penal,
                     obj,dc,f0=None):
    """
    Update objective and gradient for forcing a state variable to a specific 
    values at certain points. The mechanic version of this is the compliant 
    mechanism with controlled displacement.

    Parameters
    ----------
    xPhys : np.array
        SIMP densities of shape (nel).
    u : np.array
        state variable (displacement, temperature) of shape (ndof).
    u0 : np.array
        value that state variable is supposed to take. shape (ncontr). 
    l : np.array
        indicator vector for state variable of shape (ndof). Is 1 at output 
        nodes.
    free : np.array
        indices of free nodes.
    inds_contr : np.array
        indices of nodes where the displacement is to be controlled.
    K : scipy.sparse matrix/array
        global stiffness matrix of shape (ndof).
    KE : np.array
        element stiffness matrix of shape (nedof).
    edofMat : np.array
        element degree of freedom matrix of shape (nel,nedof)
    Amax : float
        maximum value for material property A
    Amin : float
        minimum value for material property A. Should be small compared to Amax
        but not zero and Amax + Amin should recover the property A of the material
    penal: float
        penalty exponent for the SIMP method.
    obj : float
        objective function.
    dc : np.ndarray
        sensitivities/gradients of design variables with regards to objective 
        function. shape (ndesign)
    f0: np.array
        if system is subjected to an affine transformation causing the loads, 
        this is the resulting load for an element of density one. shape (nedof)

    Returns
    -------
    obj : float
        updated objective function.
    dc : np.ndarray
        updated sensitivities/gradients of design variables with regards to 
        objective function. shape (ndesign)

    """
    raise NotImplementedError()
    obj += ((u[inds_contr] - u0)**2).sum()
    # solve adjoint problem
    h = np.zeros(l.shape)
    h[free] = spsolve(K, (-2)*l[free]*(u[inds_contr]-u0).sum())
    #
    if f0 is None:
        dc[:] += penal*xPhys**(penal-1)*(Amax-Amin)*(np.dot(h[edofMat], KE)*\
                 u[edofMat]).sum(1)
    else:
        dc[:] += penal*xPhys**(penal-1)*(Amax-Amin)*(np.dot(h, KE)*\
                 (u[edofMat,0]-f0[None,:])).sum(1)
    return obj, dc
