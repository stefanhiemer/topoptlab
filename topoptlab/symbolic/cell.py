# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List, Tuple, Union

from sympy import roots,symbols,solve,solve_univariate_inequality
from symfem import create_element, create_reference
from symfem.references import Reference
from symfem.functions import ScalarFunction

def base_cell(ndim: int,
              element_type: str = "Lagrange",
              order: int = 1) -> Tuple[Tuple,List,Reference,List]:
    """
    Create the basic cell, location of vertices, the node indices, the
    reference cell and the basis functions.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    vertices : tuple
        coordinates of vertices
    nd_inds : list
        list of indices with regards to the reference cell of symfem. Check the
        reference cell numbering out in the symfem documentation or git.
    reference : symfem.references.vertices
        reference cell object.
    basis : list
        list of basis functions.
    """
    if order not in [1,2]:
        raise NotImplementedError("Beyond order 1 not yet implemented.")
    #
    if ndim == 1:
        cell_name = "interval"
    elif ndim == 2:
        cell_name = "quadrilateral"
    elif ndim == 3:
        cell_name = "hexahedron"
    #
    if order == 1 and element_type=="Lagrange":
        if ndim == 1:
            # define the vertices
            vertices = ((-1,), (1,))
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [0, 1]
        elif ndim == 2:
            # define the vertices
            vertices = ((-1, -1), (1, -1), (1, 1), (-1, 1))
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [0, 1, 3, 2]
        elif ndim == 3:
            # define the vertices
            vertices = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [0, 1, 3, 2,
                       4, 5, 7, 6]
    elif order == 2 and element_type=="Lagrange":
        if ndim == 1:
            # define the vertices
            vertices = ((-1,), (1,), (0,))
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [0, 1, 2]
        elif ndim == 2:
            # define the vertices
            vertices = ((-1, -1), (1, -1), (1, 1), (-1, 1),
                        (0, -1), (1, 0), (0, 1), (-1, 0),
                        (0, 0) )
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [0, 1, 3, 2,
                       4, 6, 7, 5,
                       8]
        elif ndim == 3:
            # define the vertices
            vertices = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
                        (0, -1, -1), (1, 0, -1), (0, 1, -1), (-1, 0, -1),
                        (0, 0, -1),
                        (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0),
                        (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0),
                        (0, 0, 0),
                        (0, -1, 1), (1, 0, 1), (0, 1, 1), (-1, 0, 1),
                        (0, 0, 1))
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [0, 1, 3, 2,
                       4, 5, 7, 6,
                       8, 11, 13, 9,
                       20,
                       10, 12, 15, 14,
                       21, 23, 24, 22,
                       26,
                       16, 18, 19, 17,
                       25]
    elif order == 3 and element_type=="Lagrange":
        if ndim == 1:
            # define the vertices
            vertices = ((-1,), (-1/3,), (1/3,), (1,) )
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [1, 2, 0]
    # reorder vertices according to the given node indices
    _vertices = tuple(vertices[i] for i in nd_inds)
    # create element
    element = create_element(cell_name, element_type, order=order)
    # Create a reference cell with these vertices: this will be used
    # to compute the integrals
    reference = create_reference(cell_name, vertices=_vertices)
    # map the basis functions to the cell
    basis = element.map_to_cell(_vertices)
    # reorder basis function according to the current node ordering
    basis = [basis[i] for i in nd_inds]
    return vertices, nd_inds, reference, basis

def determine_nodeinds(vertices: Tuple,
                       basis_funcs: Union[List, ScalarFunction],
                       ndim: int,
                       autoshift: bool = True) -> List:
    """
    Find index of each vertex by finding the basis function that amounts to 1.

    For a set of vertex coordinates, determine to which basis function each
    vertex corresponds. Keep in mind that the current default unit cell used
    by symfem is in the interval [0,1] whereas topoptlab's is typically in the
    interval [-1,1]. If ´autoshift´ is True, I assume the cell to be within 
    [-1,1].

    Parameters
    ----------
    vertices : tuple
        coordinates of vertices as created by base_cell
    basis_funcs : list of symfem.functions.ScalarFunction
        list of basis functions which are usually extracted by calling
        element.get_basis_functions().
    ndim : int
        number of dimensions

    Returns
    -------
    indices : list
        indices of vertices
    """
    ndim = len(vertices[0])
    inds = []
    for vertex in vertices:
        # shift from interval [-1,1] to [0,1]
        if autoshift:
            vertex = [c/2 + 1/2 for c in vertex]
        # evaluate basis functions at vertex
        ind = [i for i,func in enumerate(basis_funcs) \
               if func.subs(vars=["x","y", "z"][:ndim], values=vertex)==1]
        inds = inds + ind
    return inds

def find_node_coords(basis_funcs: Union[List, ScalarFunction]) -> List:
    """
    Find coordinate of node associated with a basis function from the delta 
    property.

    Parameters
    ----------
    basis_funcs : list of symfem.functions.ScalarFunction
        list of basis functions which are usually extracted by calling
        element.get_basis_functions().

    Returns
    -------
    coords : list
        coordinates as list of list. 
    """
    if basis_funcs[0].as_sympy().has(symbols("z")): 
        #sols = [[roots(func-1, symbols("x")),
        #         roots(func-1, symbols("y")),
        #         roots(func-1, symbols("z"))] for func in basis_funcs]
        symbs = symbols("x y z")
    elif basis_funcs[0].as_sympy().has(symbols("y")):
        #ndim = 2
        #sols = [[solve(func-1, symbols("x y"))] for func in basis_funcs]
        symbs = symbols("x y")
    elif basis_funcs[0].as_sympy().has(symbols("x")):
        #ndim = 1
        #sols = [[roots(func-1, symbols("x"))] for func in basis_funcs]
        symbs = symbols("x")
    else:
        raise ValueError("The basis functions appear not to depend on space.")
    #
    sols = [solve(func-1, symbs)[0] for func in basis_funcs]
    #
    for symb in symbs:
        for bas_sol in sols:
            for sol in bas_sol:
                solve_univariate_inequality(-1<=sol,symb) 
    return sols 
    
if __name__=="__main__":
    
    lagr = create_element("quadrilateral", 
                          "Lagrange", 
                          1)
    print(lagr.get_basis_functions())
    print(find_node_coords(lagr.get_basis_functions()))