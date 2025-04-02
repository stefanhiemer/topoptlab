from numpy import array, linspace, zeros
from numpy.testing import assert_almost_equal

from scipy.differentiate import derivative, jacobian

import pytest

from topoptlab.bounds.hashin_shtrikman_3d import conductivity_binary_low, conductivity_nary_low

@pytest.mark.parametrize('npoints, kmin, kmax',
                         [(11,1e-2,1.),
                          (11,0.5,1.),])

def test_binary_low(npoints,kmin,kmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(conductivity_binary_low(x, kmin = kmin, kmax = kmax), 
                        conductivity_nary_low(x[:,None], 
                                              ks = array([kmax,kmin])))
    return

from topoptlab.bounds.hashin_shtrikman_3d import conductivity_binary_upp, conductivity_nary_upp

@pytest.mark.parametrize('npoints, kmin, kmax',
                         [(11,1e-2,1.),
                          (11,0.5,1.),])

def test_binary_upp(npoints,kmin,kmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(conductivity_binary_upp(x, kmin = kmin, kmax = kmax), 
                        conductivity_nary_upp(x[:,None], 
                                              ks = array([kmax,kmin])))
    return

from topoptlab.bounds.hashin_shtrikman_3d import conductivity_binary_low_dx

@pytest.mark.parametrize('npoints, kmin, kmax',
                         [(11,1e-2,1.),
                          (11,0.5,1.),])

def test_binary_low_dx(npoints,kmin,kmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(conductivity_binary_low_dx(x, kmin = kmin, kmax = kmax), 
                        derivative(conductivity_binary_low,x,args=(kmin,kmax)).df)
    return

from topoptlab.bounds.hashin_shtrikman_3d import conductivity_binary_upp_dx

@pytest.mark.parametrize('npoints, kmin, kmax',
                         [(11,1e-2,1.),
                          (11,0.5,1.),])

def test_binary_upp_dx(npoints,kmin,kmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(conductivity_binary_upp_dx(x, kmin = kmin, kmax = kmax), 
                        derivative(conductivity_binary_upp,x,args=(kmin,kmax)).df)
    return

from topoptlab.bounds.hashin_shtrikman_3d import conductivity_nary_low_dx

@pytest.mark.parametrize('npoints, ks',
                         [( 11, array([1e-2,1.])),
                          #( 11, array([1.,1e-2])),
                          (11, array([0.5,1.]) ),])

def test_nary_low_dx(npoints,ks):
    
    x = linspace(0,1,npoints)[:,None]
    #
    jac = zeros(x.shape)
    k0 = conductivity_nary_low(x, ks = ks) 
    d = 1e-8
    for i in range(x.shape[1]):
        dx = zeros(x.shape)
        dx[:,i] += d
        jac[:,i] = (conductivity_nary_low(x+dx, ks = ks)-k0)/d
    
    assert_almost_equal(conductivity_nary_low_dx(x, ks = ks),jac,
                        decimal=5)
    return

from topoptlab.bounds.hashin_shtrikman_3d import conductivity_nary_upp_dx

@pytest.mark.parametrize('npoints, ks',
                         [( 11, array([1e-2,1.])),
                          ( 11, array([1.,1e-2])),
                          ( 11, array([0.5,1.]) ),])

def test_nary_upp_dx(npoints,ks):
    
    x = linspace(0,1,npoints)[:,None]
    #
    jac = zeros(x.shape)
    k0 = conductivity_nary_upp(x, ks = ks) 
    d = 1e-8
    for i in range(x.shape[1]):
        dx = zeros(x.shape)
        dx[:,i] += d
        jac[:,i] = (conductivity_nary_upp(x+dx, ks = ks)-k0)/d
    
    assert_almost_equal(conductivity_nary_upp_dx(x, ks = ks),jac,
                        decimal=6)
    return
