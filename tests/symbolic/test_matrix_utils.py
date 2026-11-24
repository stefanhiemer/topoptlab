from symfem.functions import MatrixFunction

from topoptlab.symbolic.matrix_utils import eye

import pytest


@pytest.mark.parametrize('ndim,sol',
                         [(3,
                           MatrixFunction([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]])) ])

def test_eye(ndim,sol):
    
    assert sol == eye(ndim)
    return

from topoptlab.symbolic.matrix_utils import diag

@pytest.mark.parametrize('diagonal,sol',
                         [([1,2,3],
                           MatrixFunction([[1, 0, 0],
                                           [0, 2, 0],
                                           [0, 0, 3]]))  ])

def test_diag(diagonal,sol):
    
    assert sol == diag(diagonal)
    return

from topoptlab.symbolic.matrix_utils import kron

@pytest.mark.parametrize('',
                         [()])

def test_kron():
    #
    A = generate_constMatrix(nrow=2,ncol=2,name="A")
    B = generate_constMatrix(nrow=2,ncol=2,name="B")
    #
    sol = MatrixFunction([[A[0,0]*B[0,0], 
                           A[0,0]*B[0,1], 
                           A[0,1]*B[0,0], 
                           A[0,1]*B[0,1]],
                          [A[0,0]*B[1,0], 
                           A[0,0]*B[1,1], 
                           A[0,1]*B[1,0], 
                           A[0,1]*B[1,1]], 
                          [A[1,0]*B[0,0], 
                           A[1,0]*B[0,1], 
                           A[1,1]*B[0,0], 
                           A[1,1]*B[0,1]], 
                          [A[1,0]*B[1,0], 
                           A[1,0]*B[1,1], 
                           A[1,1]*B[1,0], 
                           A[1,1]*B[1,1]]])
    #       
    assert kron(A,B) == sol
    return

from topoptlab.symbolic.matrix_utils import eig,inverse,simplify_matrix

@pytest.mark.parametrize('ndim',
                         [(2)])

def test_eig(ndim):
    #
    M = generate_constMatrix(nrow=ndim,ncol=ndim,name="M")
    #
    D,P = eig(M=M)
    #       
    assert simplify_matrix(P@diag(D)@inverse(P)) == M
    return

from topoptlab.symbolic.matrix_utils import trace

@pytest.mark.parametrize('size',
                         [(6)])

def test_trace(size):
    #
    M = generate_constMatrix(nrow=size,
                             ncol=size,
                             name="M")
    #
    t = 0
    for i in range(size):
        t = t + M[i,i]
    #
    assert trace(M) == t
    return

from topoptlab.symbolic.matrix_utils import to_square

@pytest.mark.parametrize('size,order,sol',
                         [(9,"F",
                           MatrixFunction([[0, 3, 6],
                                           [1, 4, 7],
                                           [2, 5, 8]])),
                           (9,"C",
                             MatrixFunction([[0, 1, 2],
                                             [3, 4, 5],
                                             [6, 7, 8]])) ])

def test_tosquare(size,order,sol):
    v = MatrixFunction([[i] for i in range(size)])
    assert sol == to_square(v,order=order)
    return

from topoptlab.symbolic.matrix_utils import to_column

@pytest.mark.parametrize('size,order',
                         [(9,"F"),
                          (9,"C") ])

def test_tocolumn(size,order):
    M = generate_constMatrix(nrow=size,ncol=size,name="M",symmetric=False)
    assert M == to_square( to_column(M,order=order), order=order)
    return

from topoptlab.symbolic.matrix_utils import to_voigt,generate_constMatrix

@pytest.mark.parametrize('ndim',
                         [ (2),
                           (3)])

def test_tovoigt(ndim):
    #
    M = generate_constMatrix(ndim,ndim,"M",symmetric=True)
    M_v = to_voigt(M)
    if ndim==2:
        sol = MatrixFunction( [[M[0,0]],[M[1,1]],
                               [M[0,1]]] )
    elif ndim==3:
        sol = MatrixFunction( [[M[0,0]],[M[1,1]],[M[2,2]],
                               [M[1,2]],[M[0,2]],[M[0,1]]] )
    assert sol == M_v
    return

from topoptlab.symbolic.matrix_utils import from_voigt

@pytest.mark.parametrize('ndim',
                         [ (2),
                           (3)])

def test_fromvoigt(ndim):
    #
    M = generate_constMatrix(ndim,ndim,"M",symmetric=True) 
    assert M == from_voigt(to_voigt(M))
    return

from sympy.abc import x,y,z
from topoptlab.symbolic.matrix_utils import factor_matrix

@pytest.mark.parametrize('pref,remainder,solution,parallel,symmetry',
                         [(2*x**3,
                           [[z**2,y**2+z],[1,x**2+y**2+z**2]],
                           8*x**3,
                           False,
                           False), 
                          (2*x**3,
                           [[z**2,y**2+z],
                            [1,x**2+y**2+z**2]],
                           8*x**3,
                           True,
                           False),
                          (2*x**3,
                           [[z**2,1],
                            [1,x**2+y**2+z**2]],
                           8*x**3,
                           True,
                           True),
                          (2*x**3,
                           [[z**2,1],
                            [1,x**2+y**2+z**2]],
                           8*x**3,
                           True,
                           True)])

def test_factor(pref,remainder,solution,parallel,symmetry):
    
    M = MatrixFunction([[pref*expr for expr in row] for row in remainder])
    remainder, factor = factor_matrix(M=M, variables=[y,z], 
                                      parallel=parallel, symmetry=symmetry)
    
    assert factor == solution
    return

from symfem.symbols import x,t
from symfem import create_element, create_reference

from topoptlab.symbolic.matrix_utils import flatten,integrate
from topoptlab.symbolic.utils import is_equal

@pytest.mark.parametrize('el,shape,order,ndim,mode,symmetry,parallel,backend',
                         [("Lagrange","triangle",1,2,"vector",False,False,"sympy"), 
                          ("Lagrange","triangle",1,2,"matrix",False,False,"sympy"), 
                          ("Lagrange","triangle",1,2,"matrix",True,False,"sympy"),
                          ("Lagrange","triangle",1,2,"vector",False,True,"sympy"),
                          ("Lagrange","triangle",1,2,"matrix",False,True,"sympy"), 
                          ("Lagrange","triangle",1,2,"matrix",True,True,"sympy"),
                          ("Lagrange","triangle",1,2,"vector",False,False,"symint"),
                          ("Lagrange","triangle",1,2,"matrix",True,True,"symint")])

def test_integrate(el,shape,order,ndim,mode, 
                   symmetry,parallel,backend):
    """
    Integrate shape functions N if vectorfunction, if matrixfunc 
    integrate N@N.T.
    """
    # define the vertices
    vertice = (0, 0), (1, 0), (0, 1)
    # create a Lagrange element
    element = create_element(shape, el, order)
    # create a reference cell with these vertices: this will be used
    # to compute the integral over the triangle
    ref = create_reference(shape, vertices=vertice)
    # map the basis functions to the cell and apply gradient
    N = MatrixFunction( [[bfunc] for bfunc in element.map_to_cell(vertice)])
    #
    if mode == "vector":
        integrand = flatten(N) 
    elif mode == "matrix":
        integrand = N@N.transpose()
    #
    sol = integrand.integral(ref,x)
    # actual
    actual = integrate(M=integrand, 
                       domain=ref, 
                       variables=x, 
                       dummy_vars=t, 
                       parallel=parallel,
                       symmetry=symmetry,
                       backend=backend)
    #
    if mode == "vector":
        assert all(is_equal(actual[i].as_sympy(), sol[i].as_sympy()) \
                   for i in range(sol.shape[0]))
    else:
        assert all(is_equal(actual[i][j].as_sympy(), sol[i][j].as_sympy()) \
                   for i in range(sol.shape[0]) for j in range(sol.shape[1]))
    return

from topoptlab.symbolic.matrix_utils import integrate_separated, generate_constMatrix
from topoptlab.symbolic.utils import is_equal, separate_terms

def test_separate_terms():
    from sympy import symbols
    a, b, X, Y = symbols("a b X Y")
    expr = (a + b*X)*(1 + Y) + a/(1 + X*Y)
    terms = separate_terms(expr, [X, Y])
    assert set(terms) == {1, X, Y, X*Y, 1/(X*Y + 1)}
    assert terms[1] == a and terms[X*Y] == b
    assert is_equal(sum(c*m for m, c in terms.items()), expr)
    return

@pytest.mark.parametrize('simplify', [False, True])
@pytest.mark.parametrize('shape,mode,symmetry,parallel,backend',
                         [("triangle","matrix",True,False,"sympy"),
                          ("quadrilateral","vector",False,False,"sympy"),
                          ("quadrilateral","matrix",False,True,"sympy"),
                          ("hexahedron","matrix",True,True,"sympy"),
                          ("quadrilateral","laplace",False,False,"sympy"),
                          ("quadrilateral","laplace",True,False,"symint"),
                          ("hexahedron","matrix",True,True,"symint")])
def test_integrate_separated(shape, mode, symmetry, parallel, backend,
                             simplify):
    """
    Compare integrate_separated with integrate for mass matrices, vectors of
    shape functions and Laplacian with generic conductivity.
    """
    element = create_element(shape,
                             "Lagrange" if shape == "triangle" else "Q", 1)
    ref = create_reference(shape)
    N = MatrixFunction([[b] for b in element.get_basis_functions()])
    if mode == "vector":
        integrand = flatten(N)
    elif mode == "matrix":
        integrand = N@N.transpose()
    elif mode == "laplace":
        G = MatrixFunction([[b.grad(2)[0], b.grad(2)[1]] \
                            for b in element.get_basis_functions()])
        c = generate_constMatrix(2, 2, "c", symmetric=True)
        integrand = G@c@G.transpose()
    #
    sol = integrate(M=integrand, domain=ref, variables=x, dummy_vars=t,
                    parallel=False)
    actual = integrate_separated(M=integrand, domain=ref, variables=x,
                                 dummy_vars=t, parallel=parallel,
                                 symmetry=symmetry, backend=backend,
                                 simplify=simplify)
    if mode == "vector":
        assert all(is_equal(actual[i].as_sympy(), sol[i].as_sympy()) \
                   for i in range(sol.shape[0]))
    else:
        assert all(is_equal(actual[i][j].as_sympy(), sol[i][j].as_sympy()) \
                   for i in range(sol.shape[0]) for j in range(sol.shape[1]))
    return

from topoptlab.symbolic.matrix_utils import common_factor

def test_common_factor():
    from sympy import symbols, tan
    l1, l2, k, g = symbols("l1 l2 k g")
    assert common_factor([2*k/(3*l1*l2), 4*k**2/(9*l1), 0,
                          6*k*tan(g)/(l1*l2**2)]) == 2*k/(9*l1)
    assert common_factor([l1, 1/l1]) == 1
    assert common_factor([0, 0]) == 1
    return

def test_integrate_separated_factor():
    """
    Common prefactor is extracted and returned separately.
    """
    from sympy import symbols
    k, l = symbols("k l")
    element = create_element("quadrilateral", "Q", 1)
    ref = create_reference("quadrilateral")
    N = MatrixFunction([[b] for b in element.get_basis_functions()])
    integrand = (N@N.transpose())*(6*k*l)
    full = integrate(M=integrand, domain=ref, variables=x, dummy_vars=t)
    M, factor = integrate_separated(M=integrand, domain=ref, variables=x,
                                    dummy_vars=t, return_factor=True)
    assert factor == 6*k*l
    assert all(is_equal(M[i][j].as_sympy()*factor, full[i][j].as_sympy()) \
               for i in range(4) for j in range(4))
    return
