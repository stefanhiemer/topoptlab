import numpy as np
# different elements/physics
from topoptlab.stiffness_tensors import isotropic_2d, isotropic_3d
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d,_lk_linear_elast_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d,_lk_linear_elast_3d,lk_linear_elast_aniso_3d
from topoptlab.elements.poisson_2d import lk_poisson_2d,_lk_poisson_2d
from topoptlab.elements.poisson_3d import lk_poisson_3d,_lk_poisson_3d
from topoptlab.elements.mass_2d import _lm_mass_2d, lm_mass_symfem
from topoptlab.elements.mass_3d import _lm_mass_3d, lm_mass_3d
from topoptlab.elements.heatexpansion_2d import _fk_linear_heatexp_2d

def compare_mass_2d(xe = np.array([[[-1.,-1.],
                                    [1.,-1.], 
                                    [1.,1.], 
                                    [-1.,1.]]])):
    #
    Ke_quad = _lm_mass_2d(xe=xe)
    #
    Ke_analyt = lm_mass_symfem()
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_mass_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])):
    #
    Ke_quad = _lm_mass_3d(xe=xe)
    #
    Ke_analyt = lm_mass_3d()
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_laplacian_2d(xe = np.array([[[-1.,-1.], 
                                         [1.,-1.], 
                                         [1.,1.], 
                                         [-1.,1.]]]),
                         k = np.eye(2)):
    #
    Ke_quad = _lk_poisson_2d(xe=xe,k=k)
    #
    Ke_analyt = lk_poisson_2d(k=1) 
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_laplacian_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                         [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                         k = np.eye(3)):
    #
    Ke_quad = _lk_poisson_3d(xe=xe,k=k)
    #
    Ke_analyt = lk_poisson_3d(k=1)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt,
                               rtol=0,
                               atol=1e-14)
    return

def compare_elast_2d(xe = np.array([[[-1.,-1.], 
                                     [1.,-1.], 
                                     [1.,1.], 
                                     [-1.,1.]]]),
                     c = isotropic_2d()):
    #
    Ke_quad = _lk_linear_elast_2d(xe=xe,c=c)
    #
    Ke_analyt = lk_linear_elast_2d() 
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_elast_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                               [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                     c = isotropic_3d()):
    #
    Ke_quad = _lk_linear_elast_3d(xe=xe,c=c)
    #
    Ke_analyt = lk_linear_elast_aniso_3d(c) 
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return


if __name__ == "__main__":
    #
    compare_mass_2d()
    #
    compare_mass_3d()
    
