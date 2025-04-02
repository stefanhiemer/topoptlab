from numpy import array, linspace, zeros,flip
from numpy.testing import assert_almost_equal

from scipy.differentiate import derivative

import pytest

from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_binary_low,bulkmod_nary_low

@pytest.mark.parametrize('npoints, Kmin, Kmax, Gmin, Gmax',
                         [(11,1e-2,1.,2e-2,1.),
                          (11,0.5,1.,0.4,1.),])

def test_binary_bulk_low(npoints,Kmin,Kmax,Gmin,Gmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(bulkmod_binary_low(x, 
                                           Kmin = Kmin, Kmax = Kmax,
                                           Gmin = Gmin, Gmax = Gmax), 
                        bulkmod_nary_low(x[:,None], 
                                         Ks = array([Kmax,Kmin]),
                                         Gs = array([Gmax,Gmin])))
    assert_almost_equal(flip(bulkmod_binary_low(x, 
                                           Kmin = Kmin, Kmax = Kmax,
                                           Gmin = Gmin, Gmax = Gmax)), 
                        bulkmod_nary_low(x[:,None], 
                                         Ks = array([Kmin,Kmax]),
                                         Gs = array([Gmin,Gmax])))
    return

from topoptlab.bounds.hashin_shtrikman_3d import shearmod_binary_low,shearmod_nary_low

@pytest.mark.parametrize('npoints, Kmin, Kmax, Gmin, Gmax',
                         [(11,1e-2,1.,2e-2,1.),
                          (11,0.5,1.,0.4,1.),])

def test_binary_shear_low(npoints,Kmin,Kmax,Gmin,Gmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(shearmod_binary_low(x, 
                                           Kmin = Kmin, Kmax = Kmax,
                                           Gmin = Gmin, Gmax = Gmax), 
                        shearmod_nary_low(x[:,None], 
                                         Ks = array([Kmax,Kmin]),
                                         Gs = array([Gmax,Gmin])))
    assert_almost_equal(flip(shearmod_binary_low(x, 
                                           Kmin = Kmin, Kmax = Kmax,
                                           Gmin = Gmin, Gmax = Gmax)), 
                        shearmod_nary_low(x[:,None], 
                                         Ks = array([Kmin,Kmax]),
                                         Gs = array([Gmin,Gmax])))
    return

from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_binary_upp,bulkmod_nary_upp

@pytest.mark.parametrize('npoints, Kmin, Kmax, Gmin, Gmax',
                         [(11,1e-2,1.,2e-2,1.),
                          (11,0.5,1.,0.4,1.),])

def test_binary_bulk_upp(npoints,Kmin,Kmax,Gmin,Gmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(bulkmod_binary_upp(x, 
                                           Kmin = Kmin, Kmax = Kmax,
                                           Gmin = Gmin, Gmax = Gmax), 
                        bulkmod_nary_upp(x[:,None], 
                                         Ks = array([Kmax,Kmin]),
                                         Gs = array([Gmax,Gmin])))
    assert_almost_equal(flip(bulkmod_binary_upp(x, 
                                           Kmin = Kmin, Kmax = Kmax,
                                           Gmin = Gmin, Gmax = Gmax)), 
                        bulkmod_nary_upp(x[:,None], 
                                         Ks = array([Kmin,Kmax]),
                                         Gs = array([Gmin,Gmax])))
    return

from topoptlab.bounds.hashin_shtrikman_3d import shearmod_binary_upp,shearmod_nary_upp

@pytest.mark.parametrize('npoints, Kmin, Kmax, Gmin, Gmax',
                         [(11,1e-2,1.,2e-2,1.),
                          (11,0.5,1.,0.4,1.),])

def test_binary_shear_upp(npoints,Kmin,Kmax,Gmin,Gmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(shearmod_binary_upp(x, 
                                           Kmin = Kmin, Kmax = Kmax,
                                           Gmin = Gmin, Gmax = Gmax), 
                        shearmod_nary_upp(x[:,None], 
                                         Ks = array([Kmax,Kmin]),
                                         Gs = array([Gmax,Gmin])))
    assert_almost_equal(flip(shearmod_binary_upp(x, 
                                           Kmin = Kmin, Kmax = Kmax,
                                           Gmin = Gmin, Gmax = Gmax)), 
                        shearmod_nary_upp(x[:,None], 
                                         Ks = array([Kmin,Kmax]),
                                         Gs = array([Gmin,Gmax])))
    return

from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_binary_low_dx

@pytest.mark.parametrize('npoints, Kmin, Kmax, Gmin, Gmax',
                         [(11,1e-2,1.,2e-2,1.),
                          (11,0.5,1.,0.4,1.),])

def test_binary_bulk_low_dx(npoints,Kmin,Kmax,Gmin,Gmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(bulkmod_binary_low_dx(x, 
                                              Kmin = Kmin, Kmax = Kmax,
                                              Gmin = Gmin, Gmax = Gmax), 
                        derivative(bulkmod_binary_low,x,args=(Kmin,Kmax,
                                                              Gmin,Gmax)).df)
    return

from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_binary_upp_dx

@pytest.mark.parametrize('npoints, Kmin, Kmax, Gmin, Gmax',
                         [(11,1e-2,1.,2e-2,1.),
                          (11,0.5,1.,0.4,1.),])

def test_binary_bulk_upp_dx(npoints,Kmin,Kmax,Gmin,Gmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(bulkmod_binary_upp_dx(x, 
                                              Kmin = Kmin, Kmax = Kmax,
                                              Gmin = Gmin, Gmax = Gmax), 
                        derivative(bulkmod_binary_upp,x,args=(Kmin,Kmax,
                                                              Gmin,Gmax)).df)
    return

from topoptlab.bounds.hashin_shtrikman_3d import shearmod_binary_low_dx

@pytest.mark.parametrize('npoints, Kmin, Kmax, Gmin, Gmax',
                         [(11,1e-2,1.,2e-2,1.),
                          (11,0.5,1.,0.4,1.),])

def test_binary_shear_low_dx(npoints,Kmin,Kmax,Gmin,Gmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(shearmod_binary_low_dx(x, 
                                              Kmin = Kmin, Kmax = Kmax,
                                              Gmin = Gmin, Gmax = Gmax), 
                        derivative(shearmod_binary_low,x,args=(Kmin,Kmax,
                                                              Gmin,Gmax)).df)
    return

from topoptlab.bounds.hashin_shtrikman_3d import shearmod_binary_upp_dx

@pytest.mark.parametrize('npoints, Kmin, Kmax, Gmin, Gmax',
                         [(11,1e-2,1.,2e-2,1.),
                          (11,0.5,1.,0.4,1.),])

def test_binary_shear_upp_dx(npoints,Kmin,Kmax,Gmin,Gmax):
    
    x = linspace(0,1,npoints)
    
    assert_almost_equal(shearmod_binary_upp_dx(x, 
                                              Kmin = Kmin, Kmax = Kmax,
                                              Gmin = Gmin, Gmax = Gmax), 
                        derivative(shearmod_binary_upp,x,args=(Kmin,Kmax,
                                                              Gmin,Gmax)).df)
    return

from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_nary_low_dx

@pytest.mark.parametrize('npoints, Ks, Gs',
                         [(11,array([1e-2,1.]),array([2e-2,1.])),
                          (11,array([0.5,1.]),array([0.4,1.])),])

def test_nary_bulk_low_dx(npoints,Ks,Gs):
    
    x = linspace(0,1,npoints)[:,None]
    #
    jac = zeros(x.shape)
    k0 = bulkmod_nary_low(x, 
                          Ks = Ks,
                          Gs = Gs)
    d = 1e-9
    for i in range(x.shape[1]):
        dx = zeros(x.shape)
        dx[:,i] += d
        jac[:,i] = (bulkmod_nary_low(x+dx,
                                     Ks = Ks,
                                     Gs = Gs) - k0)/d
    
    assert_almost_equal(bulkmod_nary_low_dx(x, 
                                            Ks = Ks,
                                            Gs = Gs),
                        jac,
                        decimal=5)
    return

from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_nary_upp_dx

@pytest.mark.parametrize('npoints, Ks, Gs',
                         [(11,array([1e-2,1.]),array([2e-2,1.])),
                          (11,array([0.5,1.]),array([0.4,1.])),])

def test_nary_bulk_upp_dx(npoints,Ks,Gs):
    
    x = linspace(0,1,npoints)[:,None]
    #
    jac = zeros(x.shape)
    k0 = bulkmod_nary_upp(x, 
                          Ks = Ks,
                          Gs = Gs)
    d = 1e-9
    for i in range(x.shape[1]):
        dx = zeros(x.shape)
        dx[:,i] += d
        jac[:,i] = (bulkmod_nary_upp(x+dx,
                                     Ks = Ks,
                                     Gs = Gs) - k0)/d
    
    assert_almost_equal(bulkmod_nary_upp_dx(x, 
                                            Ks = Ks,
                                            Gs = Gs),
                        jac,
                        decimal=5)
    return

from topoptlab.bounds.hashin_shtrikman_3d import shearmod_nary_low_dx

@pytest.mark.parametrize('npoints, Ks, Gs',
                         [(11,array([1e-2,1.]),array([2e-2,1.])),
                          (11,array([0.5,1.]),array([0.4,1.])),])

def test_nary_shear_low_dx(npoints,Ks,Gs):
    
    x = linspace(0,1,npoints)[:,None]
    #
    jac = zeros(x.shape)
    k0 = shearmod_nary_low(x, 
                           Ks = Ks,
                           Gs = Gs)
    d = 1e-9
    for i in range(x.shape[1]):
        dx = zeros(x.shape)
        dx[:,i] += d
        jac[:,i] = (shearmod_nary_low(x+dx,
                                      Ks = Ks,
                                      Gs = Gs) - k0)/d
    
    assert_almost_equal(shearmod_nary_low_dx(x, 
                                             Ks = Ks,
                                             Gs = Gs),
                        jac,
                        decimal=5)
    return

from topoptlab.bounds.hashin_shtrikman_3d import shearmod_nary_upp_dx

@pytest.mark.parametrize('npoints, Ks, Gs',
                         [(11,array([1e-2,1.]),array([2e-2,1.])),
                          (11,array([0.5,1.]),array([0.4,1.])),])

def test_nary_shear_upp_dx(npoints,Ks,Gs):
    
    x = linspace(0,1,npoints)[:,None]
    #
    jac = zeros(x.shape)
    k0 = shearmod_nary_upp(x, 
                           Ks = Ks,
                           Gs = Gs)
    d = 1e-9
    for i in range(x.shape[1]):
        dx = zeros(x.shape)
        dx[:,i] += d
        jac[:,i] = (shearmod_nary_upp(x+dx,
                                     Ks = Ks,
                                     Gs = Gs) - k0)/d
    
    assert_almost_equal(shearmod_nary_upp_dx(x, 
                                             Ks = Ks,
                                             Gs = Gs),
                        jac,
                        decimal=5)
    return
