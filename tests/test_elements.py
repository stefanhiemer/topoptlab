from numpy import array,stack
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.elements.linear_elasticity_2d import _lk_linear_elast_2d,lk_linear_elast_2d,lk_linear_elast_aniso_2d
from topoptlab.elements.linear_elasticity_3d import _lk_linear_elast_3d,lk_linear_elast_3d
from topoptlab.stiffness_tensors import isotropic_2d,isotropic_3d

@pytest.mark.parametrize('Es, nus, c, xe',
                         [([1.],[0.3],isotropic_2d(), 
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          ([1.,2.],[0.3,0.4],
                           stack([isotropic_2d(1.0,0.3),isotropic_2d(2.,0.4)]), 
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          ([1.,2.],[0.3,0.4],
                           stack([isotropic_2d(1.0,0.3),isotropic_2d(2.,0.4)]), 
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          ([1.,2.],[0.3,0.4],
                           stack([isotropic_2d(1.0,0.3),isotropic_2d(2.,0.4)]), 
                           array([[-1,-1],[1,-1],[1,1],[-1,1]])),])

def test_isotrop_linelast_2d(Es,nus,c,xe):
    
    #
    Kes = stack([lk_linear_elast_2d(E,nu) for E,nu in zip(Es,nus)])
    #
    assert_almost_equal(_lk_linear_elast_2d(xe,c),
                        Kes)
    return

@pytest.mark.parametrize('c, xe',
                         [(array([[1,2,0],[2,1,0],[0,0,1]]), 
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (stack([array([[1,2,0],[2,1,0],[0,0,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]), 
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (stack([array([[1,2,0],[2,1,0],[0,0,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]), 
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (stack([array([[1,2,0.5],[2,1,4],[0.5,4,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]), 
                           array([[-1,-1],[1,-1],[1,1],[-1,1]])),])

def test_anisotrop_linelast_2d(c,xe):
    
    #
    if len(c.shape) == 2:
        Kes = lk_linear_elast_aniso_2d(c)[None,:,:]
    else:
        Kes = stack([lk_linear_elast_aniso_2d(c[i]) for i in range(c.shape[0])])
    #
    assert_almost_equal(_lk_linear_elast_2d(xe,c),
                        Kes)
    return

def test_isotrop_linelast_3d(Es,nus,c,xe):
    
    #
    Kes = stack([lk_linear_elast_2d(E,nu) for E,nu in zip(Es,nus)])
    #
    assert_almost_equal(_lk_linear_elast_3d(xe,c),
                        Kes)
    return

if __name__ == "__main__":
    Es,nus = [1.,2.],[0.3,0.4]
    Ke_rect = stack([lk_linear_elast_3d(E,nu) for E,nu in zip(Es,nus)])
    #
    c = stack([isotropic_3d(1.0,0.3),isotropic_3d(2.,0.4)])
    #
    #xe = array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    #             [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
    #            [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    #             [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])
    xe = array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                 [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])
    Ke = _lk_linear_elast_3d(xe,isotropic_3d(1.0,0.3))
    
    #
    from topoptlab.fem import get_integrpoints
    x,w=get_integrpoints(ndim=3,nq=2)
    print(Ke_rect[0]/Ke)
    print(Ke)
    