import numpy as np

from scipy.sparse.linalg import spsolve

def compliance(xPhys,u,KE,edofMat,
               Amax,Amin,penal,
               obj,dc):
    """
    Objective and gradient for compliance minimization.

    Parameters
    ----------
    xPhys : np.array
        indices of degrees of freedom used to construct the stiffness matrix.
    u : np.array
        state variable (displacement, temperature) of shape (ndof).
    KE : np.array
        element stiffness matrix.
    Amax: float
        maximum value for material property A
    Amin: float
        minimum value for material property A. Should be small compared to Amax
        but not zero and Amax + Amin should recover the property A of the material
    penal: float
        penalty exponent for the SIMP method.

    Returns
    -------
    indices : np.arrays
        updated indices.

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
    Objective and gradient for maximization of state variable in specific 
    points. The mechanic version of this is the compliant mechanism with 
    maximized displacement.

    Parameters
    ----------
    xPhys : np.array
        indices of degrees of freedom used to construct the stiffness matrix.
    u : np.array
        state variable (displacement, temperature) of shape (ndof).
    l : np.array
        indicator vector for state variable of shape (ndof). Is 1 at output 
        nodes.
    free : np.array
        indices of free nodes.
    inds_out : np.array
        indices of nodes where the displacement is to be maximized.
    K : scipy.sparse matrix/array
        global stiffness matrix of shape (ndof).
    KE : np.array
        element stiffness matrix of shape (nedof).
    Amax: float
        maximum value for material property A
    Amin: float
        minimum value for material property A. Should be small compared to Amax
        but not zero and Amax + Amin should recover the property A of the material
    penal: float
        penalty exponent for the SIMP method.
    f0: np.array
        if system is subjected to an affine transformation causing the loads, 
        this is the resulting load for an element of density one. shape (nedof)

    Returns
    -------
    indices : np.arrays
        updated indices.

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
