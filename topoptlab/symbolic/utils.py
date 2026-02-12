# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List,Union
from multiprocessing import cpu_count

import numpy as np

from sympy import expand, simplify, Expr, collect, Function, Piecewise

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

def split_expression(expression: Expr, 
                     variables: List,
                     include_nonlin: bool,
                     nchunks: Union[None,int]) -> List:
    """
    Split expression into common powers of provided variables and merge to n
    chunks for independent parallel operations on each chunk.
    

    Parameters
    ----------
    expression : Expr
        sympy expression.
    variables : list
        list of sympy variables. 
    include_nonlin : bool
        if True, also 
    nchunks : Union[None,int]
        number of chunks. If None, number of processors.

    Returns
    -------
    chunked_expression : list
        list of additive sympy expressions whose sum equals `expression`.

    """
    if isinstance(variables, tuple):
        variables = list(variables)
    #
    if nchunks is None:
        nchunks = cpu_count()
    # extract nonlinear functions and treat them as if it were a polynomial 
    # power of a new variable
    if include_nonlin:
        #
        funcs = [f for f in expression.atoms(Function) \
                 if f.free_symbols & set(variables)]
    else:
        funcs = []
    # transform expression to list of terms of common powers
    terms = collect(expression, variables+funcs,
                    evaluate=False)
    terms = [terms[key]*key for key in terms.keys()]
    # determine number of chunks
    nterms = len(terms)
    if nterms < nchunks:
        chunk_size = 1
    else: 
        chunk_size = nterms // nchunks
    #
    chunks = [sum(terms[i*chunk_size:(i+1)*chunk_size]) \
              for i in range(nchunks-1)]
    chunks.append(sum(terms[(nchunks-1)*chunk_size:]))
    return chunks



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

def _choose_branch(piecewise_part) -> Expr:
    """
    Eliminate Piecewise heuristically by taking the branch with the longest 
    condition in terms of its string representation.
    
    Parameters
    ----------
    expression : Expr
        sympy expression with Piecewise.

    Returns
    -------
    expression_branch : Expr
        main branch of given expression.

    """
    # remove trivial condition (condition == True)
    nontrivial = [(expr,cond) for expr,cond in piecewise_part.args \
                  if cond != True]
    # if only trivial exists, return it
    if not nontrivial:
        return piecewise_part.args[-1][0]
    # heuristic criterion: choose branch with longest string representation
    best = max(nontrivial, key=lambda ec: len(str(ec[1])))
    return best[0]

def take_generic_branch(expression):
    """
    Eliminate piecewise definitions (Piecewise) heuristically by taking the 
    branch with the longest condition in terms of its string representation.
    
    Parameters
    ----------
    expression : Expr
        sympy expression with Piecewise.

    Returns
    -------
    expression_branch : Expr
        main branch of given expression.

    """
    return expression.replace(lambda e: isinstance(e,Piecewise), 
                              _choose_branch)

if __name__ == "__main__":
    
    from sympy import symbols,exp,sin

    # symbols
    a, x, y = symbols('a x y')
    
    # test expression
    expr = (a*2*x**2*exp(x**2)
            + 3*x**2*exp(x)*sin(a)
            + 4*y*sin(x)*exp(a)
            + 5*exp(x)
            + 6)
    
    # split
    chunks = split_expression(
        expression=expr,
        variables=[x, y],
        include_nonlin=True,
        nchunks=2
    )
    
    # check reconstruction
    reconstructed = sum(chunks)
    
    print("Chunks:")
    for c in chunks:
        print("  ", c)
    
    print("\nReconstruction correct:",
          simplify(reconstructed - expr) == 0)