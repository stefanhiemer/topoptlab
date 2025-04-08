from numpy import linspace
from numpy.testing import assert_almost_equal

from scipy.differentiate import derivative

import pytest

from topoptlab.bounds.hashin_rosen_3d import heatexp_binary_low,heatexp_binary_low_dx

@pytest.mark.parametrize('npoints, K1, K2, G1, G2, a1, a2',
                         [(11,76.,170.,26.,82.,22.87,12.87),])

def test_binary_low_dx(npoints,K1, K2, G1, G2, a1, a2):
    if K1 > K2:
        raise ValueError("K2 must be greater or equal K1")
    if G1 > G2:
        raise ValueError("G2 must be greater or equal G1")
    #
    x = linspace(0,1,npoints)
    #   
    assert_almost_equal(heatexp_binary_low_dx(x,
                                              Kmin=K1,Kmax=K2,
                                              Gmin=G1,Gmax=G2,
                                              amin=a1,amax=a2), 
                        derivative(heatexp_binary_low,x,args=(K1,K2, 
                                                              G1,G2, 
                                                              a1,a2)).df)
    return

from topoptlab.bounds.hashin_rosen_3d import heatexp_binary_upp,heatexp_binary_upp_dx

@pytest.mark.parametrize('npoints, K1, K2, G1, G2, a1, a2',
                         [(11,76.,170.,26.,82.,22.87,12.87),])

def test_binary_upp_dx(npoints,K1, K2, G1, G2, a1, a2):
    if K1 > K2:
        raise ValueError("K2 must be greater or equal K1")
    if G1 > G2:
        raise ValueError("G2 must be greater or equal G1")
    #
    x = linspace(0,1,npoints)
    #   
    assert_almost_equal(heatexp_binary_upp_dx(x,
                                              Kmin=K1,Kmax=K2,
                                              Gmin=G1,Gmax=G2,
                                              amin=a1,amax=a2), 
                        derivative(heatexp_binary_upp,x,args=(K1,K2, 
                                                              G1,G2, 
                                                              a1,a2)).df)
    return
