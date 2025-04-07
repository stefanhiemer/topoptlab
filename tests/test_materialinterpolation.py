from numpy import array, linspace, zeros
from numpy.testing import assert_almost_equal

from scipy.differentiate import derivative, jacobian

import pytest

from topoptlab.material_interpolation import simp,simp_dx,bound_interpol
from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_binary_low,bulkmod_binary_upp

def test_bdinterpol():
    #
    K1 = 76
    K2 = 170
    #
    G1 = 26
    G2 = 82
    #
    x = linspace(0,1,21)
    #
    assert_almost_equal(bound_interpol(xPhys=x,w=0.5,
                          bd_low=bulkmod_binary_low,
                          bd_upp=bulkmod_binary_upp,
                          bd_kws={"Kmin": K1, "Kmax": K2,
                                  "Gmin": G1, "Gmax": G2}), 
    
                        0.5*(bulkmod_binary_upp(x, 
                                                Kmin = K1, Kmax = K2,
                                                Gmin = G1, Gmax = G2)+\
                             bulkmod_binary_low(x,
                                                Kmin = K1, Kmax = K2,
                                                Gmin = G1, Gmax = G2)))
    
    return

@pytest.mark.parametrize('npoints, eps, penal',
                         [(11,1e-9,3.),])

def test_simpdx(npoints, eps, penal):
    #
    x = linspace(0,1,npoints)
    #   
    assert_almost_equal(simp_dx(xPhys=x,eps=eps,penal=penal),
                        derivative(simp,x,args=(eps,penal)).df)
    return

from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_binary_low,bulkmod_binary_upp
from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_binary_low_dx,bulkmod_binary_upp_dx
from topoptlab.material_interpolation import heatexpcoeff_binary_iso,heatexpcoeff_binary_iso_dx

@pytest.mark.parametrize('npoints, K1, K2, G1, G2, a1, a2',
                         [(11,76.,170.,26.,82.,22.87,12.87),])

def test_heatexpcoeff_binary_dx(npoints,K1, K2, G1, G2, a1, a2):
    if K1 > K2:
        raise ValueError("K2 must be greater or equal K1")
    if G1 > G2:
        raise ValueError("G2 must be greater or equal G1")
    #
    x = linspace(0,1,npoints)
    # interpolate bulk modulus
    K=0.5*(bulkmod_binary_upp(x, 
                              Kmin = K1, Kmax = K2,
                              Gmin = G1, Gmax = G2)+\
           bulkmod_binary_low(x, 
                              Kmin = K1, Kmax = K2,
                              Gmin = G1, Gmax = G2))
    dKdx=0.5*(bulkmod_binary_upp_dx(x, 
                                    Kmin = K1, Kmax = K2,
                                    Gmin = G1, Gmax = G2)+\
              bulkmod_binary_low_dx(x, 
                                    Kmin = K1, Kmax = K2,
                                    Gmin = G1, Gmax = G2))
    #
    dx = 1e-8
    dadx = (heatexpcoeff_binary_iso(x+dx,0.5*(bulkmod_binary_upp(x+dx, 
                                                                 Kmin = K1, 
                                                                 Kmax = K2,
                                                                 Gmin = G1, 
                                                                 Gmax = G2)+\
                                              bulkmod_binary_low(x+dx, 
                                                                 Kmin = K1, 
                                                                 Kmax = K2,
                                                                 Gmin = G1, 
                                                                 Gmax = G2)),
                                    Kmin=K1,Kmax=K2,
                                    amin=a1,amax=a2) - \
            heatexpcoeff_binary_iso(x,K=K,
                                    Kmin=K1,Kmax=K2,
                                    amin=a1,amax=a2))/dx
    #   
    assert_almost_equal(heatexpcoeff_binary_iso_dx(x,K=K,dKdx=dKdx,
                                                   Kmin=K1,Kmax=K2,
                                                   amin=a1,amax=a2), 
                        dadx,decimal=6)
    return
