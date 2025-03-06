from io import StringIO
import sys
from re import sub 

from sympy import symbols, Symbol
from symfem import create_element, create_reference
from symfem.functions import VectorFunction, MatrixFunction

def convert_to_code(matrix,npndarray=True):
    """
    Convert the printed expression by symfem to strings that can be
    converted to code.

    Parameters
    ----------
    matrix : symfem.functions.MatrixFunction
        symfem output.
    npndarray: bool
        if True, writes the output as numpy ndarray
        
    Returns
    -------
    lines : str
        symfem output converted to code that can be copy pasted into a 
        function.

    """
    # convert symfem.MatrixFunction to list to better print it
    ls = []
    for i in range(matrix.shape[0]):
        ls.append([])
        for j in range(matrix.shape[0]):
            ls[-1].append(matrix[i,j])
    # create a StringIO object to capture print output
    stringio_capturer = StringIO()
    # redirect stdout to the StringIO object
    sys.stdout = stringio_capturer
    # feed the matrix into the capturer
    print(ls)
    # reset stdout back to normal
    sys.stdout = sys.__stdout__
    # convert printed output to string
    lines = stringio_capturer.getvalue() 
    stringio_capturer.close()
    #
    if not npndarray:
        # add line break after every comma
        lines = lines.replace(",",",\n")
    else:
        # add np.array
        lines = "np.array(" + lines        
        lines = lines[:-2] + ")"
        #
        first_line = lines.split(",",1)[0]
        #
        delta = len("np.array("+first_line) - len(first_line) + 1
        # add line break after every comma
        lines = lines.replace(",",",\n"+"".join([" "]*delta))
    # replace with array entries
    lines = sub(r'c(\d)(\d)',
                lambda m: f'c[{int(m.group(1))-1},{int(m.group(2))-1}]',
                lines)
    return lines

def generate_constMatrix(ncol,nrow,name):
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
        if nrow != 1 and ncol !=1:
            variables = " ".join([name+str(i)+str(j) for j in range(1,ncol+1)])
        elif nrow == 1:
            variables = " ".join([name+str(j) for j in range(1,ncol+1)])
        elif ncol == 1:
            variables = " ".join([name+str(i) for j in range(1,ncol+1)])
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
    # reorder basis function according to the current node ordering
    basis = [basis[i] for i in nd_inds]
    return vertices, nd_inds, reference, basis

def bmatrix(ndim,nd_inds,basis):
    """
    Create the small strain matrix commonly referred to as bmatrix.

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
    bmatrix : symfem.functions.Matrixfunction
        small displacement matrix.
    """
    
    nrows = int((ndim**2 + ndim) /2)
    ncols = int(ndim * len(nd_inds))
    # compute gradients of basis functions
    gradN_T = VectorFunction(basis).grad(ndim).transpose()
    #
    bmatrix = [[0 for j in range(ncols)] for i in range(nrows)]
    # tension
    for i in range(ndim):
        bmatrix[i][i::ndim] = gradN_T[i]
    # shear
    i,j = ndim-2,ndim-1
    for k in range(nrows-ndim):
        #
        bmatrix[ndim+k][i::ndim] = gradN_T[j]
        bmatrix[ndim+k][j::ndim] = gradN_T[i]
        #
        i,j = (i+1)%ndim , (j+1)%ndim
    return MatrixFunction(bmatrix)