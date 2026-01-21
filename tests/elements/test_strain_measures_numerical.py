from numpy import array,eye,stack,tile,vstack
from numpy.testing import assert_allclose

import pytest


from topoptlab.elements.strain_measures import infini_strain_matrix  

from topoptlab.elements.bilinear_quadrilateral import invjacobian as \
                                                      invjacobian_bilin
from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi as \
                                                      shape_functions_dxi_bilin
from topoptlab.elements.bilinear_quadrilateral import bmatrix as \
                                                      bmatrix_bilin

@pytest.mark.parametrize('xe,xi,eta',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-2,-2.1],[2.1,-2],[2,2],[-2,2]]]), 
                           0.,0.)])

def test_consistency_bilinear(xe,xi,eta): 
    
    
    
    bmat_general = infini_strain_matrix(eta=eta,xi=xi,zeta=None,
                                        xe=xe, all_elems=False,
                                        invjacobian=invjacobian_bilin,
                                        shape_functions_dxi=shape_functions_dxi_bilin)
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
from topoptlab.elements.trilinear_hexahedron import bmatrix as \
                                                    bmatrix_trilin

@pytest.mark.parametrize('xe,xi,eta,zeta',
                         [(array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                 [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                [[-2.1,-2,-2],[2.1,-2,-2],[2,2.1,-2],[-2,2,-2],
                                 [-2,-2,2],[2,-2,2.1],[2,2,2],[-2,2.1,2]]]), 
                           0.,0.,0.)])

def test_consistency_trilinear(xe,xi,eta,zeta): 
    
    bmat_general = infini_strain_matrix(eta=eta,xi=xi,xe=xe, all_elems=False,
                              zeta=zeta,
                              invjacobian=invjacobian_trilin,
                              shape_functions_dxi=shape_functions_dxi_trilin)
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
        bmat = infini_strain_matrix(eta=eta,xi=xi,zeta=zeta, 
                                    xe=xe, all_elems=False,
                                    invjacobian=invjacobian_bilin,
                                    shape_functions_dxi=shape_functions_dxi_bilin)
    else:
        bmat = infini_strain_matrix(eta=eta,xi=xi,xe=xe, all_elems=False,
                          zeta=zeta,
                          invjacobian=invjacobian_trilin,
                          shape_functions_dxi=shape_functions_dxi_trilin)
    #
    assert_allclose(bmat@u.T,
                    eps)
    return    

from topoptlab.elements.strain_measures import dispgrad_matrix 

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
        bmat = dispgrad_matrix(eta=eta,xi=xi,xe=xe, all_elems=False,
                             zeta=zeta,
                             invjacobian=invjacobian_bilin,
                             shape_functions_dxi=shape_functions_dxi_bilin)
    else:
        bmat = dispgrad_matrix(eta=eta,xi=xi,xe=xe, all_elems=False,
                             zeta=zeta,
                             invjacobian=invjacobian_trilin,
                             shape_functions_dxi=shape_functions_dxi_trilin)
    #
    assert_allclose(bmat@u.T,
                    eps)
    return    

from topoptlab.elements.strain_measures import lagrangian_strainvar_matrix 

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

def test_consistency_lagrangian_strainvar_matrix(xe,xi,eta,zeta,u,eps):
    
    
    nel=xe.shape[0]
    
    if zeta is None:
        bmat = infini_strain_matrix(eta=eta,xi=xi,zeta=zeta, 
                                    xe=xe, all_elems=False,
                                    invjacobian=invjacobian_bilin,
                                    shape_functions_dxi=shape_functions_dxi_bilin)
        smat = lagrangian_strainvar_matrix(eta=eta,xi=xi,zeta=zeta,
                                           F=tile(eye(2),(nel,1,1)),
                                           xe=xe, all_elems=False,
                                           invjacobian=invjacobian_bilin,
                                           shape_functions_dxi=shape_functions_dxi_bilin)
    else:
        bmat = infini_strain_matrix(eta=eta,xi=xi,zeta=zeta,
                                    xe=xe, all_elems=False,
                                    invjacobian=invjacobian_trilin,
                                    shape_functions_dxi=shape_functions_dxi_trilin)
        smat = lagrangian_strainvar_matrix(eta=eta,xi=xi,zeta=zeta,
                                           F=tile(eye(3),(nel,1,1)),
                                           xe=xe, all_elems=False,
                                           invjacobian=invjacobian_trilin,
                                           shape_functions_dxi=shape_functions_dxi_trilin)
    #
    assert_allclose(bmat@u.T,
                    smat@u.T)
    return   