# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Tuple,Union

import numpy as np

def isotropic_3d(E:float = 1., 
                 nu:float = 0.3) -> np.ndarray:
    """
    3D constitutive tensor for isotropic Neo-Hooke material. 
    
    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    
    Returns
    -------
    c : np.ndarray, shape (6,6)
        stiffness tensor.
    """
    return E/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,nu,0,0,0],
                                         [nu,1-nu,nu,0,0,0],
                                         [nu,nu,1-nu,0,0,0],
                                         [0,0,0,(1-nu)/2,0,0],
                                         [0,0,0,0,(1-nu)/2,0],
                                         [0,0,0,0,0,(1-nu)/2]])

def neohookean_2pk(F : np.ndarray,
                   h : np.ndarray,
                   mu: np.ndarray,
                   Fdet: Union[None,np.ndarray] = None,
                   ndim: Union[int,None] = 3, 
                   **kwargs: Any) -> np.ndarray:
    """
    Returns 2. Piola-Kirchhoff stress (2PK) S for Neo-Hookean material:
        
        S = h*ln(det(F))*inv(B) + mu*(1-inv(B))
        
    
    where h and mu are the Lame constants, det(F) the determinant of the 
    deformation gradient F, B the left Cauchy–Green deformation tensor and inv 
    the matrix inverse.

    Parameters
    ----------
    F : None or MatrixFunction
        Deformation gradient. 
    h : None or sympy.core.symbol.Symbol or symfem.functions.ScalarFunction
        first Lame constant.
    mu : None or sympy.core.symbol.Symbol or symfem.functions.ScalarFunction
        second constant.
    Fdet : None or sympy.Expr or ScalarFunction
        determinant of deformation gradient.  
    ndim : None or int
        number of spatial dimensions. Only needed if the other two arguments 
        are None, otherwise ignored.

    Returns
    -------
    S : symfem.functions.MatrixFunction
        symbolic 2. Piola-Kirchhoff stress (2PK) in Voigt notation.

    """
    #
    ndim = F.shape[-1]
    # compute determinant if not already done
    if Fdet is None:
        Fdet = np.linalg.det(F)
    #
    B = F@F.swapaxes(-1,-2)
    #
    Binv = np.linalg.inv(B) 
    #
    return h*np.log(Fdet)*Binv + mu*(np.eye(ndim)-Binv) 