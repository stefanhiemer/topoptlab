from numpy import zeros

from numpy.testing import assert_almost_equal

import pytest

from topoptlab.topology_optimization import main 
from topoptlab.example_bc.lin_elast import forceinverter_2d
from topoptlab.objectives import var_maximization

@pytest.mark.parametrize('ft, rmin, filter_mode, obj_ref',
                         [(0,1.2,"matrix",-1.117),
                          (1,1.2,"matrix",-1.022),
                          (0,2.0,"helmholtz",-0.9957),
                          (1,2.0,"helmholtz",-0.797)]) 

def test_force_inverter(ft,rmin,filter_mode,obj_ref):
    #
    nelx=40
    nely=20
    #
    l = zeros((2*(nelx+1)*(nely+1),1))
    l[2*nelx*(nely+1),0] = -1
    # Default input parameters
    x, obj = main(nelx=nelx, nely=nely, volfrac=0.3, rmin=rmin, ft=ft, 
                  filter_mode=filter_mode, optimizer="ocm",
                  bcs=forceinverter_2d , obj_func=var_maximization ,obj_kw={"l": l},
                  output_kw = {"file": None,
                               "display": False,
                               "export": False,
                               "write_log": False,
                               "profile": False,
                               "debug": 0})
    #
    assert_almost_equal(obj,obj_ref,decimal=2)
    return 