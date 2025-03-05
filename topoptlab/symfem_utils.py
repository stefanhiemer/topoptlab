import re 

from sympy import symbols, Symbol
from symfem import create_element, create_reference
from symfem.functions import VectorFunction, MatrixFunction

def process_output(output):
    """
    Convert the printed expression by symfem to strings that can be
    converted to code.

    Parameters
    ----------
    output : str
        symfem output.

    Returns
    -------
    output : str
        symfem output converted to code.

    """
    
    # add line break after every comma
    output = output.replace(",",",\n")
    
    # replace with array entries
    output = re.sub(r'c(\d)(\d)',
                    lambda m: f'c[{int(m.group(1))-1},{int(m.group(2))-1}]',
                    output)
    
    return output

def _generate_constMatrix(ncol,nrow,name):
    """
    Generate matrix full of symbolic constants as a list of lists

    Parameters
    ----------
    ncol : int
        number of rows.
    nrow : int
        number of cols.
    name : str
        name to put in front of indices.

    Returns
    -------
    M : symfem.functions.MatrixFunction
        symfem MatrixFunction filled with symbolic entries e. g. for a 2x2 
        [[M11,M12],[M21,M22]]. 

    """
    M = []
    for i in range(1,nrow+1):
        variables = " ".join([name+str(i)+str(j) for j in range(1,ncol+1)])
        variables = symbols(variables)
        if isinstance(variables, Symbol):
            variables = [variables]
        elif isinstance(variables,tuple):
            variables = list(variables)
        
        M.append( variables )
    return MatrixFunction(M)

def base_cell(ndim, 
              element_type="Lagrange",
              order=1):
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
    if order != 1:
        raise NotImplementedError()
    if ndim == 1:
        # Define the vertived and triangles of the mesh 
        vertices = [(-1,), (1,)] 
        # node indices in reference cell of symfem. Check the git to see
        # how the numbering is done.
        nd_inds = [0, 1]
        #
        cell_name = "interval"
    elif ndim == 2:
        # Define the vertived and triangles of the mesh 
        vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)] 
        # node indices in reference cell of symfem. Check the git to see
        # how the numbering is done.
        nd_inds = [0, 1, 3, 2]
        #
        cell_name = "quadrilateral"
    elif ndim == 3:
        # Define the vertived and triangles of the mesh
        vertices = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        # node indices in reference cell of symfem. Check the git to see
        # how the numbering is done.
        nd_inds = [0, 1, 3, 2,
                   4, 5, 7, 6]
        cell_name = "hexahedron"
    # reorder vertices according to the given node indices
    vertices = tuple(vertices[i] for i in nd_inds)
    # create element
    element = create_element(cell_name, element_type, 1)
    # Create a reference cell with these vertices: this will be used
    # to compute the integrals
    reference = create_reference(cell_name, vertices=vertices)
    # map the basis functions to the cell
    basis = element.map_to_cell(vertices)
    return vertices, nd_inds, reference, basis

def bmatrix(ndim,nd_inds,basis):
    nrows = int((ndim**2 + ndim) /2)
    ncols = int(ndim * len(nd_inds))
    # compute gradients of basis functions
    gradN = [list(VectorFunction([basis[i] for i in nd_inds]).diff(var)) 
             for var in ["x","y","z"][:ndim]]
    #print(gradN)
    #
    bmatrix = [[0 for j in range(ncols)] for i in range(nrows)]
    # tension
    for i in range(ndim):
        bmatrix[i][i::ndim] = gradN[i]
    # shear
    i,j = ndim-2,ndim-1
    for k in range(nrows-ndim):
        #
        bmatrix[ndim+k][i::ndim] = gradN[j]
        bmatrix[ndim+k][j::ndim] = gradN[i]
        #
        i,j = (i+1)%ndim , (j-1)%ndim
    return MatrixFunction(bmatrix)