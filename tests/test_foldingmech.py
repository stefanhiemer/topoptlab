from numpy.testing import assert_almost_equal

import pytest

from topoptlab.legacy.folding_mechanism import main

@pytest.mark.parametrize('ft, rmin,, obj_ref',
                         [(0,1.5,-0.63242477),]) 

def test_folding_mechanism(ft,rmin,obj_ref):
    #
    nelx = 30
    # Default input parameters
    x, obj = main(nelx=nelx, nely=int(nelx/3), volfrac=0.4, penal=5.0, 
                  rmin=rmin, ft=ft, 
                  passive=False,pde=False, solver="oc",
                  nouteriter=10,expansion=0.05,
                  display=False,export=False,write_log=False)
    #
    assert_almost_equal(obj,obj_ref,decimal=2)
    return 