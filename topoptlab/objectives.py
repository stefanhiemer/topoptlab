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
    edofMat : np.ndarray shape (nel,nedof)
        element degree of freedom matrix
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
    ce = (np.dot(u[edofMat,i], KE) * u[edofMat,i]).sum(1)
    obj += ((Amin+xPhys[:,0]**penal*(Amax-Amin))*ce).sum()
    dc = (-1) * penal*xPhys**(penal-1)*(Amax-Amin)*ce
    #return obj, dc, True #
    return obj,-u, True

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
    return obj, -u * (c-c0), True 

def volume(xPhys, **kwargs):
    
    return xPhys.sum(axis=0)

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

def inverse_homogenization_maximization(u, u0, edofMat, i, KE,
                                        cellVolume, xPhys,
                                        Amax, Amin, penal,
                                        results, obj,
                                        **kwargs):
    #
    if "CH" not in results.keys():
        results["CH"] = np.zeros((u.shape[-1],u.shape[-1]))
    # calculate effective elastic tensor
    du = u0[None,:] - u[edofMat]
    # # Homogenized elasticity tensor
    dobj = np.zeros(xPhys.shape[0])
    for j in range(i,u.shape[-1]):
        # calculate elemental compliance deviation
        delta_ce = (np.dot(du[edofMat,i], KE) * du[edofMat,i]).sum(1)
        deltac = ((Amin+xPhys**penal*(Amax-Amin))*delta_ce).sum()
        results["CH"][i,j] = deltac / cellVolume
        #results["CH"][i,j] = np.einsum('nj,nij,ni->n', du[:,:,i], Kes, du[:,:,j]).sum()
        dobj += (-1) * penal*xPhys**(penal-1)*(Amax-Amin)*delta_ce
    # fill off-diagonal
    results["CH"][i:,i] = results["CH"][i,i:]
    #
    obj = results["CH"][i,:].sum()
    return obj, dobj, True

def inverse_homogenization_control(u, u0, edofMat, i, KE,
                                   cellVolume, CH0, xPhys,
                                   Amax, Amin, penal,
                                   results, obj,
                                   **kwargs):
    #
    if "CH" not in results.keys():
        results["CH"] = np.zeros(CH0)
    # calculate effective elastic tensor
    du = u0[None,:] - u[edofMat]
    # # Homogenized elasticity tensor
    dobj = np.zeros(xPhys.shape[0])
    for j in range(i,CH0.shape[-1]):
        # calculate elemental compliance deviation
        delta_ce = (np.dot(du[edofMat,i], KE) * du[edofMat,i]).sum(1)
        deltac = ((Amin+xPhys**penal*(Amax-Amin))*delta_ce).sum()
        results["CH"][i,j] = deltac / cellVolume
        #results["CH"][i,j] = np.einsum('nj,nij,ni->n', du[:,:,i], Kes, du[:,:,j]).sum()
        dc = (-1) * penal*xPhys**(penal-1)*(Amax-Amin)*delta_ce
        dobj += 2*(deltac/cellVolume -CH0[i,j]) * dc
    # fill off-diagonal
    results["CH"][i:,i] = results["CH"][i,i:]
    #
    obj = (results["CH"][i,:] - CH0[i,:]).sum()**2 + \
          (results["CH"][i:,:] - CH0[i,:]).sum()**2
    return obj, dobj, True
