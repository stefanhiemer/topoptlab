import numpy as np

from scipy.sparse.linalg import spsolve

def compliance(xPhys,u,KE,edofMat,
               Amax,Amin,penal,
               obj,**kwargs):
    """
    Update objective and gradient for stiffness maximization / compliance 
    minimization. The mechanic version of this is the compliant mechanism  with 
    maximized displacement.

    Parameters
    ----------
    xPhys : np.ndarray
        SIMP densities of shape (nel).
    u : np.ndarray
        state variable (displacement, temperature) of shape (ndof).
    KE : np.ndarray
        element stiffness matrix of shape (nedof).
    edofMat : np.ndarray
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

    Returns
    -------
    obj : float
        updated objective function.
    rhs_adj : np.ndarray
        right hand side of the adjoint problem. if problem is self adjoint, 
        this is already the solution to the self-adjoint problem.
    selfadjoint : bool, True
        obj. is selfadjoint, so no adjoint problem has to be solved

    """
    ce = (np.dot(u[edofMat], KE)
             * u[edofMat]).sum(1)
    obj += ((Amin+xPhys**penal*(Amax-Amin))*ce).sum()
    dc = (-1) * penal*xPhys**(penal-1)*(Amax-Amin)*ce
    return obj, dc, True

def var_maximization(u,l,
                     obj,**kwargs):
    """
    Update objective and gradient for maximization of state variable in 
    specified points. The mechanic version of this is the compliant mechanism 
    with maximized displacement.

    Parameters
    ----------
    u : np.ndarray
        state variable (displacement, temperature) of shape (ndof).
    l : np.ndarray
        indicator vector for state variable of shape (ndof). Is 1 or -1 at 
        output nodes depending on which direction of the dof you want to 
        maximize. 
    obj : float
        objective function.

    Returns
    -------
    obj : float
        updated objective function.
    dc : np.ndarray
        updated sensitivities/gradients of design variables with regards to 
        objective function. shape (ndesign)
    selfadjoint : bool, False
        obj. is not selfadjoint, so adjoint problem has to be solved

    """
    obj += u[l[:,0]!=0].sum()
    return obj, l, False

def var_squarederror(u,u0,l,
                     obj,**kwargs):
    """
    Update objective and gradient for forcing a state variable to a specific 
    values at certain points. The mechanic version of this is the compliant 
    mechanism with controlled displacement.

    Parameters
    ----------
    u : np.ndarray
        state variable (displacement, temperature) of shape (ndof).
    u0 : np.ndarray
        value that state variable is supposed to take. shape (ncontr). 
    l : np.ndarray
        indicator vector for state variable of shape (ndof). Is 1 at output 
        nodes.
    obj : float
        objective function.

    Returns
    -------
    obj : float
        updated objective function.
    dc : np.ndarray
        updated sensitivities/gradients of design variables with regards to 
        objective function. shape (ndesign)
    selfadjoint : bool, False
        obj. is not selfadjoint, so adjoint problem has to be solved

    """
    mask = l[:,0]!=0
    obj += ((u[mask] - u0)**2).sum()
    return obj, (-2)*l*(u[mask]-u0).sum(), False
