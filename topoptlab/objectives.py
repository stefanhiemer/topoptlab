import numpy as np

def compliance(xPhys, u, KE, edofMat, i,
               Amax, Amin, penal,
               obj, **kwargs):
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
    i : int
        index of the problem. i-th problem is used to compute the objective
        function.
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
    ce = (np.dot(u[edofMat,i], KE)
             * u[edofMat,i]).sum(1)
    obj += ((Amin+xPhys**penal*(Amax-Amin))*ce).sum()
    dc = (-1) * penal*xPhys**(penal-1)*(Amax-Amin)*ce
    return obj, dc, True

def compliance_squarederror(xPhys, u, c0, KE, edofMat, i,
                            Amax, Amin, penal,
                            obj, **kwargs):
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
    c0 : float
        target compliance
    KE : np.ndarray
        element stiffness matrix of shape (nedof).
    edofMat : np.ndarray
        element degree of freedom matrix of shape (nel,nedof)
    i : int
        index of the problem. i-th problem is used to compute the objective
        function.
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
    ce = (np.dot(u[edofMat,i], KE)
             * u[edofMat,i]).sum(1)
    c = ((Amin+xPhys**penal*(Amax-Amin))*ce).sum()
    if isinstance(c0,float):
        delta = c - c0
    else:
        delta = c - c0[i]
    obj += delta**2
    dc = 2*delta * (-1) * penal*xPhys**(penal-1)*(Amax-Amin)*ce
    return obj, dc, True

def var_maximization(u, l, i,
                     obj, **kwargs):
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
    i : int
        index of the problem. i-th problem is used to compute the objective
        function.
    obj : float
        objective function.

    Returns
    -------
    obj : float
        updated objective function.
    rhs_adj : np.ndarray
        right hand side for the adjoint problem
    selfadjoint : bool, False
        obj. is not selfadjoint, so adjoint problem has to be solved

    """
    obj += u[l[:,i]!=0].sum()
    return obj, l, False

def var_squarederror(u, u0, l, i,
                     obj, **kwargs):
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
    rhs_adj : np.ndarray
        right hand side for the adjoint problem. if problem is self adjoint,
        this is already the solution to the self-adjoint problem.
    selfadjoint : bool, False
        obj. is not selfadjoint, so adjoint problem has to be solved

    """
    #
    if isinstance(u0, float):
        u0 = np.array([u0])
    mask = l[:,i]!=0
    obj += ((u[mask,i] - u0)**2).mean()
    rhs_adj = np.zeros(l.shape)
    #mask = l != 0
    rhs_adj[mask,0] = (-2)*(u[mask,i]-u0) / u0.shape[0]
    return obj, rhs_adj , False
