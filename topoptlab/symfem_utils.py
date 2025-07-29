from itertools import product
from io import StringIO
import sys
from re import sub
import math

from sympy import symbols, Symbol
from symfem import create_element, create_reference
from symfem.functions import VectorFunction, MatrixFunction

def convert_to_code(matrix,
                    matrices=[],vectors=[],
                    np_functions=["cos","sin","tan"],
                    npndarray=True,
                    max_line_length=200):
    """
    Convert the printed expression by symfem to strings that can be
    converted to code.

    Parameters
    ----------
    matrix : symfem.functions.MatrixFunction
        symfem output.
    matrices : list
        list of strs for the tensor indices to be converted to array indices.
        E. g. the tensor "c" appears in the equation, the current element
        derivation routines will return function that contain the elements of
        this tensor in the format c11,c12,etc. This function converts these
        entries to c[0,0],c[0,1],etc.
    vectors : list
        list of strs with same logic as matrices, but instead c1,c2,etc. are
        converted to c[0],c[1],etc.
    npndarray: bool
        if True, writes the output as numpy ndarray
    max_line_length : int
        counts number of length until first "]". If larger than the specified
        value, line breaks occur at every ",", otherwise at every "],".

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
        for j in range(matrix.shape[1]):
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
    first_line = lines.split("],",1)[0]
    #
    if npndarray:
        #
        delta = len("np.array("+first_line) - len(first_line)
        # add np.array
        lines = "np.array(" + lines
        lines = lines[:-1] + ")"
        #
        # add line break after every comma
        if len(first_line) > max_line_length:
            lines = lines.replace(",",",\n"+"".join([" "]*(delta+1)))
            lines = lines.replace(" [","[")
        # add line break after every "],"
        else:
            lines = lines.replace("],","],\n"+"".join([" "]*delta))
    else:
        # add line break after every comma
        if len(first_line) > max_line_length:
            lines = lines.replace(",",",\n")
        # add line break after every "],"
        else:
            lines = lines.replace("],","],\n")
    # add numpy prefix to functions
    for npfunc in np_functions:
        lines = lines.replace(npfunc,"np."+npfunc)
    # replace entries ala "c11" with corresponding array entries c[0,0]
    for matrix in matrices:
        lines = sub(matrix + r'(\d)(\d)',
              lambda m: matrix +  f'[{int(m.group(1))-1},{int(m.group(2))-1}]',
              lines)
    for vector in vectors:
        # replace entries ala "c1" with corresponding array entries c[0]
        lines = sub(vector + r'(\d)',
                    lambda m: vector + f'[{int(m.group(1))-1}]',
                    lines)
    return lines

def generate_constMatrix(ncol,nrow,name,
                         symmetric=False, 
                         return_symbols=False):
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
    symmetric : bool
        if True, matrix generated in symmetric fashion
    return_symbols : bool
        if True, returns the list of symbols of the entries. May be needed if 
        you want to apply assumptions on them in a Sympy assumptions context.

    Returns
    -------
    M : symfem.functions.MatrixFunction
        symfem MatrixFunction filled with symbolic entries e. g. for a 2x2
        [[M11,M12],[M21,M22]].
    symbol_list : list
        list of all symbols used for the matrix/vector

    """
    #
    M = []
    if return_symbols:
        symbol_list = []
    for i in range(1,nrow+1):
        # create the matrix as list of lists 
        if nrow != 1 and ncol !=1:
            variables = " ".join([name+str(i)+str(j) for j in range(1,ncol+1)])
        # create row vector as list 
        elif nrow == 1:
            variables = " ".join([name+str(j) for j in range(1,ncol+1)])
        # create column vector as list 
        elif ncol == 1:
            variables = " ".join([name+str(i) for j in range(1,ncol+1)])
        # create symbols
        variables = symbols(variables)
        # convert to list
        if isinstance(variables, Symbol):
            variables = [variables]
        elif isinstance(variables,tuple):
            variables = list(variables)
        #
        M.append( variables )
        if return_symbols:
            symbol_list = symbol_list + variables
    # make matrix symmetric
    if symmetric:
        if ncol != nrow:
            raise ValueError("For matrix to be symmetric, it must be square. Currenly ncol != nrow")
        #
        for i in range(nrow):
            for j in range(i+1,nrow):
                M[j][i] = M[i][j]
    #
    if not return_symbols:
        return MatrixFunction(M)
    else:
        return MatrixFunction(M), symbol_list

def stifftens_isotropic(ndim,plane_stress=True):
    """
    stiffness tensor for isotropic material expressed in Terms of Young's
    modulus E and Poisson's ratio v.

    Parameters
    ----------
    ndim : int
        number of dimensions
    plane_stress : bool
        if True, return stiffness tensor for plane stress, otherwise return
        stiffness tensor for plane strain

    Returns
    -------
    c : symfem.functions.MatrixFunction
        stiffness tensor.
    """
    E,nu = symbols("E nu")
    if ndim == 1:
        return MatrixFunction([[E]])
    elif ndim == 2:
        if plane_stress:
            return E/(1-nu**2)*MatrixFunction([[1,nu,0],
                                               [nu,1,0],
                                               [0,0,(1-nu)/2]])
        else:
            return E/((1+nu)*(1-2*nu))*MatrixFunction([[1-nu,nu,0],
                                                       [nu,1-nu,0],
                                                       [0,0,(1-nu)/2]])
    elif ndim == 3:
        return E/((1+nu)*(1-2*nu))*MatrixFunction([[1-nu,nu,nu,0,0,0],
                                                   [nu,1-nu,nu,0,0,0],
                                                   [nu,nu,1-nu,0,0,0],
                                                   [0,0,0,(1-nu)/2,0,0],
                                                   [0,0,0,0,(1-nu)/2,0],
                                                   [0,0,0,0,0,(1-nu)/2]])

def simplify_matrix(M):
    """
    simplify element-wise the given MatrixFunction.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction or list
        matrix to be simplified.

    Returns
    -------
    M_new : symfem.functions.MatrixFunction
        simplified matrix.
    """
    if isinstance(M, MatrixFunction):
        nrow,ncol = M.shape[0],M.shape[1]
    elif isinstance(M, list):
        nrow,ncol = len(M),len(M[0])
    M_new = [[0 for j in range(ncol)] for i in range(nrow)]
    for i,j in product(range(M.shape[0]),range(M.shape[1])):
        M_new[i][j] = M[i,j].as_sympy().simplify()
    return MatrixFunction(M_new)

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
            # Define the vertived and triangles of the mesh
            vertices = ((-1,), (1,))
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [0, 1]
        elif ndim == 2:
            # Define the vertived and triangles of the mesh
            vertices = ((-1, -1), (1, -1), (1, 1), (-1, 1))
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [0, 1, 3, 2]
        elif ndim == 3:
            # Define the vertived and triangles of the mesh
            vertices = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [0, 1, 3, 2,
                       4, 5, 7, 6]
    elif order == 2 and element_type=="Lagrange":
        if ndim == 1:
            # Define the vertived and triangles of the mesh
            vertices = ((-1,), (1,), (0,))
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [0, 1, 2]
        elif ndim == 2:
            # Define the vertived and triangles of the mesh
            vertices = ((-1, -1), (1, -1), (1, 1), (-1, 1),
                        (0, -1), (1, 0), (0, 1), (-1, 0), 
                        (0, 0) )
            # node indices in reference cell of symfem. Check the git to see
            # how the numbering is done.
            nd_inds = [0, 1, 3, 2,
                       4, 6, 7, 5, 
                       8]
        elif ndim == 3:
            # Define the vertived and triangles of the mesh
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
            # Define the vertived and triangles of the mesh
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

def determine_nodeinds(vertices, basis_funcs, ndim):
    """
    Find index of each vertex by finding the basis function that amounts to 1.
    
    For a set of vertex coordinates, determine to which basis function each
    vertex corresponds. Keep in mind that the current default unit cell used 
    by symfem is in the interval [0,1] whereas mine is typically in the 
    interval [-1,1].

    Parameters
    ----------
    vertices : tuple
        coordinates of vertices as created by base cell
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
        vertex = [c/2 + 1/2 for c in vertex]
        # evaluate basis functions at vertex
        ind = [i for i,func in enumerate(basis_funcs) \
               if func.subs(vars=["x","y", "z"][:ndim], values=vertex)==1]
        inds = inds + ind
    return inds

def shape_function_matrix(basis,nedof,
                          mode="col"):
    """
    Generate the shape function matrix for scalar or vector fields.

    Parameters
    ----------
    basis : list
        list of basis functions as generated by base_cell
    nedof : int
        number of nodal degrees of freedom.
    mode : str
        either "row" or "col" which results in shape (nedof,n_nodes) or
        (n_nodes,nedof)

    Returns
    -------
    shape_function_matrix : symfem.functions.MatrixFunction
        shape function matrix either of shape
    """
    #
    if isinstance(basis, list):
        n_nodes = len(basis)
    elif isinstance(basis, (VectorFunction,MatrixFunction)):
        n_nodes = basis.shape[0]
    if mode in ["row","col"]:
        #
        shpfc_matr = [[0 for j in range(nedof*n_nodes)] for i in range(nedof)]
        #
        for i in range(nedof):
            shpfc_matr[i][i::nedof] = basis
    else:
        raise ValueError("Unknown construction mode.")
    if mode == "col":
        return MatrixFunction(shpfc_matr).transpose()
    else:
        return MatrixFunction(shpfc_matr)

def small_strain_matrix(ndim,nd_inds,basis,isoparam_kws):
    """
    Create the small strain matrix commonly referred to as B matrix.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions.
    element_type : str
        type of element.
    order : int
        order of element.
    basis : list
        list of basis functions as generated by base_cell

    Returns
    -------
    bmatrix : symfem.functions.Matrixfunction
        small displacement matrix.
    """

    nrows = int((ndim**2 + ndim) /2)
    ncols = int(ndim * len(nd_inds))
    # compute gradients of basis functions
    Jinv = jacobian(ndim=ndim,
                    return_J=False, return_inv=True, return_det=False,
                    **isoparam_kws)
    gradN_T = (VectorFunction(basis).grad(ndim)@Jinv.transpose()).transpose()
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

def scale_cell(vertices):
    """
    Scale/rotate the vertices/nodes basic cell by lengths l and angles g.

    Parameters
    ----------
    vertices : tuple
        coordinates of vertices as created by base cell

    Returns
    -------
    vertices_new : symfem.functions.MatrixFunction, shape (ndim,1)
        coordinate according to ispoarametric map
    """
    from sympy.functions.elementary.trigonometric import tan
    #
    if isinstance(vertices, tuple):
        vertices = MatrixFunction(vertices)
    #
    ndim = vertices.shape[1]
    # rotation angles
    g = generate_constMatrix(ncol=1,nrow=ndim-1,name="g")
    # create rotation matrix
    R = [ [0 for j in range(ndim)] for i in range(ndim)]
    for i in range(ndim):
        R[i][i] = 1
    for i in range(ndim-1):
        R[0][i+1] = tan(g[i][0])
    R = MatrixFunction(R)
    # cell lengths
    l = generate_constMatrix(ncol=1,nrow=ndim,name="l")
    # create stretch matrix
    S = [ [0 for j in range(ndim)] for i in range(ndim)]
    for i in range(ndim):
        S[i][i] = l[i][0]
    S = MatrixFunction(S)/2
    # affine transformation matrix
    return vertices@(R@S).transpose()

def isoparametric_map(basis,vertices):
    """
    Create the basic cell, location of vertices, the node indices, the
    reference cell and the basis functions.

    Parameters
    ----------
    basis : list
        list of basis functions.
    vertices : tuple
        coordinates of vertices as created by base cell

    Returns
    -------
    coord : symfem.functions.MatrixFunction, shape (ndim,1)
        coordinate according to ispoarametric map
    """
    #
    if isinstance(vertices, tuple):
        vertices = MatrixFunction(vertices)
    # get number of nodes and convert to Matrix function
    if isinstance(basis, list):
        basis = MatrixFunction([[b] for b in basis])
    elif isinstance(basis, (VectorFunction)):
        basis = MatrixFunction([[b] for b in basis])
    elif isinstance(basis,MatrixFunction):
        if basis.shape[1] != 1:
            raise ValueError("If basis is provided as MatrixFunction, must have shape (n_nodes,1)")
    return vertices.tranpose()@basis

def jacobian(ndim,
             element_type="Lagrange",
             order=1,
             return_J=True,
             return_inv=True,
             return_det=True,
             debug=False):
    """
    Symbolically compute the Jacobian of the isoparametric mapping.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.
    return_J : bool
        if True, return Jacobian matrix.
    return_inv : bool
        if True, return inverse of Jacobian matrix.
    return_det : bool
        if True, return determinant of Jacobian matrix.
    debug : bool
        if True, print additional information.

    Returns
    -------
    J : symfem.functions.MatrixFunction
        jacobian of isoparametric mapping.
    Jinv : symfem.functions.MatrixFunction
        inverse of jacobian of isoparametric mapping.
    Jdet : symfem.functions.MatrixFunction
        determinant of jacobian of isoparametric mapping.

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim=ndim,
                                               element_type=element_type,
                                               order=order)
    #
    gradN = VectorFunction(basis).grad(ndim)
    #
    scaled = scale_cell(vertices)
    if debug:
        print("cell in physical space:\n",scaled)
        print("basis functions in reference space:\n",basis)
        print("gradient of basis functions in reference space:\n",gradN)
    #
    J = simplify_matrix( gradN.transpose()@scaled )
    if return_det or return_inv:
        Jdet = J.det()
    if return_inv:
        # adjungate matrix
        Jinv = [[[] for j in range(ndim)] for j in range(ndim)]
        if ndim == 1:
            Jinv[0][0] = 1 /J[0][0] / Jdet
        elif ndim == 2:
            Jinv[0][0], Jinv[1][1] = J[1][1]/Jdet, J[0][0]/Jdet
            Jinv[0][1], Jinv[1][0] = -J[0][1]/Jdet, -J[1][0]/Jdet
        elif ndim == 3:
            #
            Jinv[0][0] = (J[1][1]*J[2][2] - J[1][2]*J[2][1]) / Jdet
            Jinv[0][1] = -(J[0][1]*J[2][2] - J[0][2]*J[2][1]) / Jdet
            Jinv[0][2] = (J[0][1]*J[1][2] - J[0][2]*J[1][1]) / Jdet
            #
            Jinv[1][0] = -(J[1][0]*J[2][2] - J[1][2]*J[2][0]) / Jdet
            Jinv[1][1] = (J[0][0]*J[2][2] - J[0][2]*J[2][0] ) / Jdet
            Jinv[1][2] = -(J[0][0]*J[1][2] - J[0][2]*J[1][0]) / Jdet
            #
            Jinv[2][0] = (J[1][0]*J[2][1] - J[1][1]*J[2][0]) / Jdet
            Jinv[2][1] = -(J[0][0]*J[2][1] - J[0][1]*J[2][0]) / Jdet
            Jinv[2][2] = (J[0][0]*J[1][1] - J[0][1]*J[1][0]) / Jdet
        #
        Jinv = simplify_matrix( MatrixFunction(Jinv) )
    if all([return_J, not return_inv, not return_det]):
        return J
    elif all([not return_J, return_inv, not return_det]):
        return Jinv
    elif all([not return_J, not return_inv, return_det]):
        return Jdet
    elif all([return_J, return_inv, not return_det]):
        return J, Jinv
    elif all([return_J, not return_inv, return_det]):
        return J, Jdet
    elif all([not return_J, return_inv, return_det]):
        return Jinv, Jdet
    elif all([return_J,return_inv,return_det]):
        return J, Jinv, Jdet
    else:
        raise ValueError("At least on of the return options must be True.")

def rotation_matrix(ndim,mode=None):
    """
    rotation matrix around y and z axis with angles phi (y axis) and theta
    (z axis).

    Parameters
    ----------
    ndim : int
        number of spatial dimensions.
    mode : str or None
        Either None or "voigt". If None, returns the standard rotation matrix.
        Either None or "voigt". If None, returns the standard rotation matrix.
        If "voigt" rotation matrix for 2nd rank tensors in Voigt notation
        ("Voigt vectors")  or 4th rank tensors ("Voigt matrices").

    Returns
    -------
    R : symfem.functions.MatrixFunction, shape (ndim,ndim) or
        ((ndim**2 + ndim) /2,(ndim**2 + ndim) /2)
        rotation matrix.

    """
    from sympy.functions.elementary.trigonometric import sin,cos
    # introduce angle variables
    if ndim == 1:
        pass
    elif ndim == 2:
        theta = symbols("theta")
    elif ndim == 3:
        theta,phi = symbols("theta phi")
    else:
        raise ValueError("ndim has to be integer and between 1 and 3.")
    # standard rotation matrix
    if mode is None:
        if ndim == 1:
            R =  MatrixFunction([[1]])
        elif ndim == 2:
            theta = symbols("theta")
            R =  MatrixFunction([[cos(theta),-sin(theta)],
                                 [sin(theta),cos(theta)]])
        elif ndim == 3:
            theta,phi = symbols("theta phi")
            R = MatrixFunction([[cos(theta)*cos(phi),-sin(theta),cos(theta)*sin(phi)],
                                [sin(theta)*cos(phi),cos(theta),sin(theta)*sin(phi)],
                                [-sin(phi),0,cos(phi)]])
        return R
    elif mode == "voigt":
        if ndim == 1:
            R =  MatrixFunction([[1]])
        elif ndim == 2:
            R = MatrixFunction([[cos(theta)**2, sin(theta)**2, -sin(2*theta)/2],
                                [sin(theta)**2, cos(theta)**2,  sin(2*theta)/2],
                                [ sin(2*theta), -sin(2*theta),    cos(2*theta)]])
        elif ndim == 3:
            R = MatrixFunction([[cos(phi)**2*cos(theta)**2,
                                 sin(theta)**2,
                                 sin(phi)**2*cos(theta)**2,
                                 0,
                                 sin(phi)*cos(phi)*cos(theta)**2,
                                 0],
                                [sin(theta)**2*cos(phi)**2,
                                 cos(theta)**2,
                                 sin(phi)**2*sin(theta)**2,
                                 0,
                                 sin(phi)*sin(theta)**2*cos(phi),
                                 0],
                                [sin(phi)**2,
                                 0,
                                 cos(phi)**2,
                                 0,
                                 -sin(2*phi)/2,
                                 0],
                                [-cos(2*phi - theta)/2 + cos(2*phi + theta)/2,
                                 0,
                                 cos(2*phi - theta)/2 - cos(2*phi + theta)/2,
                                 0,
                                 -sin(2*phi - theta)/2 + sin(2*phi + theta)/2,
                                 0],
                                [-sin(2*phi - theta)/2 - sin(2*phi + theta)/2,
                                 0,
                                 sin(2*phi - theta)/2 + sin(2*phi + theta)/2,
                                 0,
                                 cos(2*phi - theta)/2 + cos(2*phi + theta)/2,
                                 0],
                                [2*sin(theta)*cos(phi)**2*cos(theta),
                                 -sin(2*theta),
                                 2*sin(phi)**2*sin(theta)*cos(theta),
                                 0,
                                 cos(2*phi - 2*theta)/4 - cos(2*phi + 2*theta)/4,
                                 0]])
        return R

def rotation_matrix_dangle(ndim,mode=None):
    """
    1st derivative of rotation matrix around y and z axis with angles phi
    (y axis) and theta (z axis).

    Parameters
    ----------
    ndim : int
        number of spatial dimensions.
    mode : str or None
        Either None or "voigt". If None, returns derivatives of the standard
        rotation matrix. If "voigt" rotation matrix for 2nd rank tensors in
        Voigt notation ("Voigt vectors")  or 4th rank tensors
        ("Voigt matrices").

    Returns
    -------
    dRdtheta : symfem.functions.MatrixFunction, shape (ndim,ndim) or
        ((ndim**2 + ndim) /2,(ndim**2 + ndim) /2)
        1st derivative of rotation matrix with regards to theta.
    dRdphi : symfem.functions.MatrixFunction, shape (ndim,ndim) or
        ((ndim**2 + ndim) /2,(ndim**2 + ndim) /2)
        1st derivative of rotation matrix with regards to phi.


    """
    from sympy.functions.elementary.trigonometric import sin,cos
    # introduce angle variables
    if ndim == 1:
        pass
    elif ndim == 2:
        theta = symbols("theta")
    elif ndim == 3:
        theta,phi = symbols("theta phi")
    else:
        raise ValueError("ndim has to be integer and between 1 and 3.")
    # standard rotation matrix
    if mode is None:
        if ndim == 1:
            R =  MatrixFunction([[1]])
        elif ndim == 2:
            theta = symbols("theta")
            dRdtheta =  MatrixFunction([[-sin(theta), -cos(theta)],
                                        [cos(theta), -sin(theta)]])
            return dRdtheta

        elif ndim == 3:
            theta,phi = symbols("theta phi")
            dRdtheta = MatrixFunction([[-sin(theta)*cos(phi), -cos(theta), -sin(phi)*sin(theta)],
                                       [cos(phi)*cos(theta), -sin(theta),  sin(phi)*cos(theta)],
                                       [0, 0, 0]])
            dRdphi = MatrixFunction([[-sin(phi)*cos(theta), 0, cos(phi)*cos(theta)],
                                     [-sin(phi)*sin(theta), 0, sin(theta)*cos(phi)],
                                     [-cos(phi), 0, -sin(phi)]])

            return dRdtheta,dRdphi
    elif mode == "voigt":
        if ndim == 1:
            R =  MatrixFunction([[1]])
        elif ndim == 2:
            dRdtheta = MatrixFunction([[-sin(2*theta), sin(2*theta), -cos(2*theta)],
                                       [sin(2*theta),   -sin(2*theta),    cos(2*theta)],
                                       [2*cos(2*theta), -2*cos(2*theta), -2*sin(2*theta)]])
            return dRdtheta
        elif ndim == 3:
            dRdtheta = MatrixFunction([[-2*sin(theta)*cos(phi)**2*cos(theta),
                                        sin(2*theta),
                                        -2*sin(phi)**2*sin(theta)*cos(theta),
                                        0,
                                        -cos(2*phi - 2*theta)/4 + cos(2*phi + 2*theta)/4,
                                        0],
                                       [2*sin(theta)*cos(phi)**2*cos(theta),
                                        -sin(2*theta),
                                        2*sin(phi)**2*sin(theta)*cos(theta),
                                        0,
                                        cos(2*phi - 2*theta)/4 - cos(2*phi + 2*theta)/4,
                                        0],
                                       [0, 0, 0, 0, 0, 0],
                                       [-sin(2*phi - theta)/2 - sin(2*phi + theta)/2,
                                        0,
                                        sin(2*phi - theta)/2 + sin(2*phi + theta)/2,
                                        0,
                                        cos(2*phi - theta)/2 + cos(2*phi + theta)/2,
                                        0],
                                       [cos(2*phi - theta)/2 - cos(2*phi + theta)/2,
                                        0,
                                        -cos(2*phi - theta)/2 + cos(2*phi + theta)/2,
                                        0,
                                        (2*sin(phi)**2 - 1)*sin(theta),
                                        0],
                                       [2*cos(phi)**2*cos(2*theta),
                                        -2*cos(2*theta),
                                        2*sin(phi)**2*cos(2*theta),
                                        0,
                                        sin(2*phi - 2*theta)/2 + sin(2*phi + 2*theta)/2,
                                        0]])
            dRdphi = MatrixFunction([[-2*sin(phi)*cos(phi)*cos(theta)**2,
                                      0,
                                      2*sin(phi)*cos(phi)*cos(theta)**2,
                                      0,
                                      cos(2*phi)*cos(theta)**2,
                                      0],
                                     [-2*sin(phi)*sin(theta)**2*cos(phi),
                                      0,
                                      2*sin(phi)*sin(theta)**2*cos(phi),
                                      0,
                                      sin(theta)**2*cos(2*phi),
                                      0],
                                     [sin(2*phi),
                                      0,
                                      -sin(2*phi),
                                      0,
                                      -cos(2*phi),
                                      0],
                                     [2*(2*sin(phi)**2 - 1)*sin(theta),
                                      0,
                                      -sin(2*phi - theta) + sin(2*phi + theta),
                                      0,
                                      -cos(2*phi - theta) + cos(2*phi + theta),
                                      0],
                                     [2*(2*sin(phi)**2 - 1)*cos(theta),
                                      0,
                                      cos(2*phi - theta) + cos(2*phi + theta),
                                      0,
                                      -sin(2*phi - theta) - sin(2*phi + theta),
                                      0],
                                     [-cos(2*phi - 2*theta)/2 + cos(2*phi + 2*theta)/2,
                                      0,
                                      cos(2*phi - 2*theta)/2 - cos(2*phi + 2*theta)/2,
                                      0,
                                      -sin(2*phi - 2*theta)/2 + sin(2*phi + 2*theta)/2,
                                      0]])
            return dRdtheta,dRdphi
        return R

def convert_to_voigt(A):
    """
    Convert 2nd rank tensor into its Voigt representation.

    Parameters
    ----------
    A : symfem.functions.MatrixFunction, shape (ndim,ndim)
        2nd rank tensor

    Returns
    -------
    A_v : symfem.functions.MatrixFunction, shape ((ndim**2 + ndim) /2, 1)
        2nd rank tensor in voigt notation
    """
    #
    if isinstance(A,MatrixFunction):
        ndim = A.shape[0]
    elif isinstance(A,list):
        ndim = len(A)
    #
    A_v = [[A[i][i]] for i in range(ndim)]
    if ndim == 3:
        A_v += [[A[1][-1]]]
    A_v += [[A[0][i]] for i in range(ndim-1,0,-1)]
    return MatrixFunction(A_v)

def convert_from_voigt(A_v):
    """
    Convert 2nd rank tensor into from its Voigt representation to the standard
    matrix represenation.

    Parameters
    ----------
    A_v : symfem.functions.MatrixFunction, shape ((ndim**2 + ndim) /2, 1)
        2nd rank tensor in Voigt represenation (so a column vector)

    Returns
    -------
    A : symfem.functions.MatrixFunction, shape (ndim,ndim)
        2nd rank tensor in matrix notation
    """
    #
    if isinstance(A_v,MatrixFunction):
        l = A_v.shape[0]
    elif isinstance(A_v,list):
        l = len(A_v)
    #
    ndim = int( -1/2 + math.sqrt(1/2+2*l) )
    #
    A = [[0 for j in range(ndim)] for i in range(ndim)]
    for i in range(ndim):
        A[i][i] = A_v[i][0]
    if ndim > 1:
        A[0][1] = A_v[-1][0]
        A[1][0] = A_v[-1][0]
    if ndim > 2:
        A[1][2] =  A_v[3][0]
        A[2][1] =  A_v[3][0]
        A[0][2] =  A_v[4][0]
        A[2][0] =  A_v[4][0]
    return MatrixFunction(A)
