from numpy.testing import assert_almost_equal

import pytest

from topoptlab.compliance_minimization import main 

@pytest.mark.parametrize('ft, rmin, pde, obj_ref',
                         [(0,2.4,False,216.81),
                          (1,2.4,False,233.71),
                          (0,2.4,True,218.79),
                          (1,2.4,True,237.60),]) 

def test_mbb_filter(ft,rmin,pde,obj_ref):
    
    #
    x, obj = main(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=rmin, ft=ft, 
                  passive=False,pde=pde,solver="oc",
                  display=False,export=False,write_log=False)
    #
    assert_almost_equal(obj,obj_ref,decimal=2)
    return 