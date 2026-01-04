# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

from sympy import expand,simplify,Expr

def is_equal(expr1: Expr, expr2: Expr)-> bool:
    """
    Return True if two SymPy expressions are algebraically identical.

    The check expands the difference and asks SymPy to simplify it to zero.

    Parameters
    ----------
    expr1 : sympy.Expr
        first expression.
    expr2 : sympy.Expr
        second expression.

    Returns
    -------
    bool
        True if ``expr1 - expr2`` simplifies to zero (identical),
        False otherwise.
    """
    return simplify(expand(expr1 - expr2)) == 0


rect = np.array([[[-1,-1],[1,-1],[1,1],[-1,1]]])
hexahedron = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])
triangle = np.array([[0,0],[1,0],[0,1]])
tetrahedron = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])

def argsort_counterclock(coords: np.ndarray, 
                         vertices: np.ndarray,
                         adj_list: np.ndarray,
                         z_tol: float = 1e-9, 
                         angle_tol: float = 1e-12, 
                         round_decimals: int = 12) -> np.ndarray:
    """
    Indices to sort nodes according to coordinates beginning with nodes on 
    corners, then nodes lying on edges and finally nodes within the element in 
    a counter-clockwise manner. This should reproduce most element orderings 
    similar to Abaqus.
    
    Assume all coordinates are in [-1,1], we start from point (-1,-1,-1), sort 
    all coordinates with z=-1 in the x-y plane counterclockwise and then move 
    upwards to the next layer of equal z-value. We do this by calculating the 
    radius and angle in the xy-plane and shift the angle that points on (-1,-1)
    correspond to an angle of zero. Then we sort by angles, z coordinate and 
    radius.

    Parameters
    ----------
    coords : np.ndarray, shape (N,ndim)
        coordinates.
    z_tol : float
        tolerance for putting coordinates in same z-layer.
    angle_tol : float
        tolerance for snapping angles.
    round_decimals : int
        number of decimals to round α and r for stable ordering.
    
    Returns
    -------
    inds_sorted : np.ndarray, shape (N,ndim)
        indices to sort coordinates.
    """
    #
    coords = np.asarray(coords, float)
    N, ndim = coords.shape
    #
    # 1D 
    if ndim == 1:
        return coords[np.argsort(coords[:, 0])]
    # angle
    alpha = np.arctan2(coords[:,1], coords[:,0])
    # flip angles smaller than -3/4*np.pi
    mask = alpha<(-3/4)*np.pi
    alpha[mask] = alpha[mask] + 2*np.pi
    # snap small angle values (avoid -0.0)
    alpha[np.abs(alpha) < angle_tol] = 0.0
    # radius for edge/interior discrimination
    r = np.linalg.norm(coords[:,:2], axis=1)
    # rounding for stability
    alpha = np.round(alpha, round_decimals)
    r     = np.round(r,     round_decimals)
    # 2D
    if ndim == 2:
        # lexicographic sort by (α, r)
        idx = np.lexsort((r, alpha))
        return coords[idx]
    # 3D
    if ndim == 3: 
        # lexicographic sort: first z, then α, then r
        idx = np.lexsort((r, 
                          alpha, 
                          np.round(coords[:,2], round_decimals))) # z
        return coords[idx]

    raise ValueError("Only dimensions 1, 2, 3 are supported.")