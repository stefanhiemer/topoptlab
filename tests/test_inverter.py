from numpy.testing import assert_almost_equal

import pytest

from topoptlab.compliant_mechanisms import main 

@pytest.mark.parametrize('ft, rmin, pde, obj_ref',
                         [(0,1.2,False,-1.117),
                          (1,1.2,False,-1.022),
                          (0,2.0,True,-0.9957),
                          (1,2.0,True,-0.797)]) 

def test_force_inverter(ft,rmin,pde,obj_ref):
    # Default input parameters
    x, obj = main(nelx=40, nely=20, volfrac=0.3, penal=3.0, rmin=rmin, ft=ft, 
                  passive=False,pde=pde, solver="oc",
                  display=False,export=False,write_log=False)
    #
    assert_almost_equal(obj,obj_ref,decimal=2)
    return 