from numpy import array,stack,vstack
from numpy.testing import assert_allclose

import pytest


from topoptlab.elements.strain_measures import eng_strain  

from topoptlab.elements.bilinear_quadrilateral import invjacobian as \
                                                      invjacobian_bilin
from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi as \
                                                      shape_functions_dxi_bilin
from topoptlab.elements.bilinear_quadrilateral import check_inputs as \
                                                      check_inputs_bilin
from topoptlab.elements.bilinear_quadrilateral import bmatrix as \
                                                      bmatrix_bilin

@pytest.mark.parametrize('xe,xi,eta',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-2,-2.1],[2.1,-2],[2,2],[-2,2]]]), 
                           0.,0.)])

def test_consistency_bilinear(xe,xi,eta): 
    
    
    
    bmat_general = eng_strain(eta=eta,xi=xi,xe=xe, all_elems=False,
                              invjacobian=invjacobian_bilin,
                              shape_functions_dxi=shape_functions_dxi_bilin,
                              check_fnc=check_inputs_bilin)
    #
    bmat_quadri = bmatrix_bilin(eta=eta,xi=xi,xe=xe)
    #
    assert_allclose(bmat_general,
                    bmat_quadri)
    return

from topoptlab.elements.trilinear_hexahedron import invjacobian as \
                                                    invjacobian_trilin
from topoptlab.elements.trilinear_hexahedron import shape_functions_dxi as \
                                                    shape_functions_dxi_trilin
from topoptlab.elements.trilinear_hexahedron import check_inputs as \
                                                    check_inputs_trilin
from topoptlab.elements.trilinear_hexahedron import bmatrix as \
                                                    bmatrix_trilin

@pytest.mark.parametrize('xe,xi,eta,zeta',
                         [(array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                 [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                [[-2.1,-2,-2],[2.1,-2,-2],[2,2.1,-2],[-2,2,-2],
                                 [-2,-2,2],[2,-2,2.1],[2,2,2],[-2,2.1,2]]]), 
                           0.,0.,0.)])

def test_consistency_trilinear(xe,xi,eta,zeta): 
    
    bmat_general = eng_strain(eta=eta,xi=xi,xe=xe, all_elems=False,
                              zeta=zeta,
                              invjacobian=invjacobian_trilin,
                              shape_functions_dxi=shape_functions_dxi_trilin,
                              check_fnc=check_inputs_trilin)
    #
    bmat_hex = bmatrix_trilin(eta=eta,xi=xi,zeta=zeta,xe=xe)
    #
    assert_allclose(bmat_general,
                    bmat_hex)
    return

@pytest.mark.parametrize('xe,xi,eta,zeta,u,eps',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-2,-2],[2,-2],[2,2],[-2,2]]]), 
                           0.,0.,None, 
                           array([0.,0.,
                                  0.,0.,
                                  0.,2.,
                                  0.,2.])[None,:],
                           array([[[0.],[1.],[0.]], 
                                  [[0.],[0.5],[0.]]])),
                          (array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                   [[-2,-2],[2,-2],[2,2],[-2,2]]]), 
                            0.,0.,None, 
                            array([-2.,0.,
                                   -2.,0.,
                                   2.,0.,
                                   2.,0.])[None,:],
                            array([[[0.],[0.],[2.]], 
                                   [[0.],[0.],[1.]]])),
                          (array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1], 
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                   [[-2,-2,-1],[2,-2,-1],[2,2,-1],[-2,2,-1],
                                    [-2,-2,1],[2,-2,1],[2,2,1],[-2,2,1]]]), 
                            0.,0.,0., 
                            array([0.,0.,0.,
                                   0.,0.,0.,
                                   0.,2.,0.,
                                   0.,2.,0., 
                                   0.,0.,0.,
                                   0.,0.,0.,
                                   0.,2.,0.,
                                   0.,2.,0.])[None,:],
                            array([[[0.],[1.],[0.],[0.],[0.],[0.]], 
                                   [[0.],[0.5],[0.],[0.],[0.],[0.]]])),
                          (array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1], 
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                   [[-2,-2,-1],[2,-2,-1],[2,2,-1],[-2,2,-1],
                                    [-2,-2,1],[2,-2,1],[2,2,1],[-2,2,1]]]), 
                            0.,0.,0., 
                            array([-2.,0.,0.,
                                   -2.,0.,0.,
                                   2.,0.,0.,
                                   2.,0.,0., 
                                   -2.,0.,0.,
                                   -2.,0.,0.,
                                   2.,0.,0.,
                                   2.,0.,0.])[None,:],
                            array([[[0.],[0.],[0.],[0.],[0.],[2.]], 
                                   [[0.],[0.],[0.],[0.],[0.],[1.]]])),
                          ])

def test_eng_strain(xe,xi,eta,zeta,u,eps):
    
    if zeta is None:
        bmat = eng_strain(eta=eta,xi=xi,xe=xe, all_elems=False,
                          zeta=zeta,
                          invjacobian=invjacobian_bilin,
                          shape_functions_dxi=shape_functions_dxi_bilin,
                          check_fnc=check_inputs_bilin)
    else:
        bmat = eng_strain(eta=eta,xi=xi,xe=xe, all_elems=False,
                          zeta=zeta,
                          invjacobian=invjacobian_trilin,
                          shape_functions_dxi=shape_functions_dxi_trilin,
                          check_fnc=check_inputs_trilin)
    #
    assert_allclose(bmat@u.T,
                    eps)
    return    

from topoptlab.elements.strain_measures import disp_gradient 

@pytest.mark.parametrize('xe,xi,eta,zeta,u,eps',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-2,-2],[2,-2],[2,2],[-2,2]]]), 
                           0.,0.,None, 
                           array([0.,0.,
                                  0.,0.,
                                  0.,2.,
                                  0.,2.])[None,:],
                           array([[[0.],[0.],[0.],[1.]], 
                                  [[0.],[0.],[0.],[0.5]]])),
                          (array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1], 
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                   [[-2,-2,-1],[2,-2,-1],[2,2,-1],[-2,2,-1],
                                    [-2,-2,1],[2,-2,1],[2,2,1],[-2,2,1]]]), 
                            0.,0.,0., 
                            array([0.,0.,0.,
                                   0.,0.,0.,
                                   0.,2.,0.,
                                   0.,2.,0., 
                                   0.,0.,0.,
                                   0.,0.,0.,
                                   0.,2.,0.,
                                   0.,2.,0.])[None,:],
                            array([[[0.],[0.],[0.],
                                    [0.],[1.],[0.], 
                                    [0.],[0.],[0.]], 
                                   [[0.],[0.],[0.],
                                    [0.],[0.5],[0.], 
                                    [0.],[0.],[0.]]])),
                          (array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                   [[-2,-2],[2,-2],[2,2],[-2,2]]]), 
                            0.,0.,None, 
                            array([-2.,0.,
                                   -2.,0.,
                                   2.,0.,
                                   2.,0.])[None,:],
                            array([[[0.],[2.],[0.],[0.]], 
                                   [[0.],[1.],[0.],[0.]]])),
                           (array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1], 
                                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                    [[-2,-2,-1],[2,-2,-1],[2,2,-1],[-2,2,-1],
                                     [-2,-2,1],[2,-2,1],[2,2,1],[-2,2,1]]]), 
                             0.,0.,0., 
                             array([-2.,0.,0.,
                                    -2.,0.,0.,
                                    2.,0.,0.,
                                    2.,0.,0., 
                                    -2.,0.,0.,
                                    -2.,0.,0.,
                                    2.,0.,0.,
                                    2.,0.,0.])[None,:],
                             array([[[0.],[2.],[0.],
                                     [0.],[0.],[0.], 
                                     [0.],[0.],[0.]], 
                                    [[0.],[1.],[0.],
                                     [0.],[0.],[0.], 
                                     [0.],[0.],[0.]]]))
                          ])

def test_disp_grad(xe,xi,eta,zeta,u,eps):
    
    if zeta is None:
        bmat = disp_gradient(eta=eta,xi=xi,xe=xe, all_elems=False,
                             zeta=zeta,
                             invjacobian=invjacobian_bilin,
                             shape_functions_dxi=shape_functions_dxi_bilin,
                             check_fnc=check_inputs_bilin)
    else:
        bmat = disp_gradient(eta=eta,xi=xi,xe=xe, all_elems=False,
                             zeta=zeta,
                             invjacobian=invjacobian_trilin,
                             shape_functions_dxi=shape_functions_dxi_trilin,
                             check_fnc=check_inputs_trilin)
    #
    assert_allclose(bmat@u.T,
                    eps)
    return    