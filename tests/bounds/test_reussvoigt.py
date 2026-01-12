from numpy import array, eye, diag, full, stack, zeros
from numpy.testing import assert_allclose

from scipy.differentiate import derivative

import pytest

from topoptlab.bounds.voigt_reuss import voigt

@pytest.mark.parametrize('x, props, solution',
                         [(full( (10,2), 1/3 ),
                           array([1.,2.,3.]),
                           full(10,2)),
                          (full( (10,2), 1/3 ),
                           stack([eye(3)*i for i in range(1,4) ]),
                           full(10,2)[:,None,None]*eye(3)[None,:,:]),
                          (full((1,2),1/3),
                           array([[[0.19772511, 0.03175103, 0.81942621],
                                   [0.83287207, 0.73212012, 0.01908369],
                                   [0.19992185, 0.35020118, 0.06610353]],
                                  [[0.83796466, 0.79872648, 0.43820849],
                                   [0.75480036, 0.73244855, 0.88014594],
                                   [0.41668254, 0.66517702, 0.63454342]],
                                  [[0.50869907, 0.47631109, 0.42599609],
                                   [0.92821878, 0.74012699, 0.67978724],
                                   [0.74510659, 0.32426825, 0.99907238]]]),
                           array([[[0.51479628, 0.4355962 , 0.56121026],
                                   [0.8386304 , 0.73489855, 0.52633896],
                                   [0.45390366, 0.44654882, 0.56657311]]])),])

def test_voigt(x,props,solution):
    assert_allclose(voigt(x=x, props=props), 
                    solution)
    return

from topoptlab.bounds.voigt_reuss import reuss

@pytest.mark.parametrize('x, props, solution',
                         [(full( (10,2), 1/3 ),
                           array([1.,2.,3.]),
                           full(10,18/11)),
                          (full( (10,2), 1/3 ),
                           stack([eye(3)*i for i in range(1,4) ]),
                           full(10,18/11)[:,None,None]*eye(3)[None,:,:]),
                          (full((1,2),1/3),
                           array([[[0.19772511, 0.03175103, 0.81942621],
                                   [0.83287207, 0.73212012, 0.01908369],
                                   [0.19992185, 0.35020118, 0.06610353]],
                                  [[0.83796466, 0.79872648, 0.43820849],
                                   [0.75480036, 0.73244855, 0.88014594],
                                   [0.41668254, 0.66517702, 0.63454342]],
                                  [[0.50869907, 0.47631109, 0.42599609],
                                   [0.92821878, 0.74012699, 0.67978724],
                                   [0.74510659, 0.32426825, 0.99907238]]]),
                           array([[[0.08517243, -0.32570093, 1.43604163],
                                   [1.10987728, 1.25160166, -0.26413251],
                                   [1.11279168, 2.16307384, -2.12868829]]])),])

def test_reuss(x,props,solution):
    assert_allclose(reuss(x=x, props=props), 
                    solution)
    return 


from topoptlab.bounds.voigt_reuss import voigt_dx

@pytest.mark.parametrize('x, props',
                         [(full( (10,2), 1/3 ),
                           array([1.,2.,3.])),
                          (full( (10,2), 1/3 ),
                           stack([eye(3)*i for i in range(1,4) ])),
                           (full((1,2),1/3), 
                            array([[[1., 0.375, 0.18],
                                    [0.375, 1., 0.35],
                                    [0.18, 0.35, 1.]],
                                   [[1., 0.75, 0.43],
                                    [0.75, 1., 0.06],
                                    [0.43, 0.06, 1.]],
                                   [[1., 0.4, 0.47],
                                    [0.4, 1., 0.7],
                                    [0.47, 0.7, 1.]]])),])

def test_voigt_dx(x,props):
    #
    dx = 5e-9
    #
    actual = voigt_dx(x=x, props=props)
    #
    finite_diff = zeros(actual.shape)
    for i in range(actual.shape[1]):
        x_dx = x.copy()
        x_dx[:,i] = x[:,i] + dx
        finite_diff[:,i]=(voigt(x=x_dx, props=props)-voigt(x=x,props=props))/dx
    #
    assert_allclose(actual, 
                    finite_diff,
                    atol=1e-7)
    return

from topoptlab.bounds.voigt_reuss import reuss_dx

@pytest.mark.parametrize('x, props',
                         [(full( (10,2), 1/3 ),
                           array([1.,2.,3.])),
                          (full( (10,2), 1/3 ),
                           stack([eye(3)*i for i in range(1,4) ])),
                          (full((1,2),1/3), 
                           array([[[1., 0.375, 0.18],
                                   [0.375, 1., 0.35],
                                   [0.18, 0.35, 1.]],
                                  [[1., 0.75, 0.43],
                                   [0.75, 1., 0.06],
                                   [0.43, 0.06, 1.]],
                                  [[1., 0.4, 0.47],
                                   [0.4, 1., 0.7],
                                   [0.47, 0.7, 1.]]])),])

def test_reuss_dx(x,props): 
    #
    dx = 1e-9
    #
    actual = reuss_dx(x=x, props=props)
    #
    finite_diff = zeros(actual.shape)
    for i in range(actual.shape[1]):
        x_dx = x.copy()
        x_dx[:,i] = x[:,i] + dx
        finite_diff[:,i]=(reuss(x=x_dx, props=props)-reuss(x=x,props=props))/dx
    #
    assert_allclose(actual, 
                    finite_diff,
                    atol=1e-6)
    return