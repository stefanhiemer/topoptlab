import numpy as np
# different elements/physics
from topoptlab.stiffness_tensors import isotropic_2d, isotropic_3d
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d,_lk_linear_elast_2d,lk_linear_elast_aniso_2d
from topoptlab.elements.linear_elasticity_3d import _lk_linear_elast_3d,lk_linear_elast_3d,lk_linear_elast_aniso_3d
from topoptlab.elements.poisson_2d import lk_poisson_2d,_lk_poisson_2d,lk_poisson_aniso_2d
from topoptlab.elements.poisson_3d import lk_poisson_3d,_lk_poisson_3d,lk_poisson_aniso_3d
from topoptlab.elements.mass_2d import _lm_mass_2d, lm_mass_2d
from topoptlab.elements.mass_3d import _lm_mass_3d, lm_mass_3d
from topoptlab.elements.heatexpansion_2d import fk_heatexp_2d,_fk_heatexp_2d,fk_heatexp_aniso_2d
from topoptlab.elements.heatexpansion_3d import fk_heatexp_3d,_fk_heatexp_3d,fk_heatexp_aniso_3d
from topoptlab.elements.bodyforce_2d import _lf_bodyforce_2d, lf_bodyforce_2d
from topoptlab.elements.bodyforce_3d import _lf_bodyforce_3d, lf_bodyforce_3d


def compare_heatexp_iso_2d(xe = np.array([[[-1.,-1.],
                                           [1.,-1.],
                                           [1.,1.],
                                           [-1.,1.]]]),
                           DeltaT=np.ones(4)):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    c = isotropic_2d()
    #
    a = 0.05
    #
    fe_quad = _fk_heatexp_2d(xe=xe,
                             c=c,
                             a=np.eye(2)*a,
                             DeltaT=DeltaT)
    #
    fe_analyt = fk_heatexp_2d(E=1.,nu=0.3,a=a,
                              DeltaT=DeltaT,
                              l=l)
    #
    np.testing.assert_allclose(fe_quad[0],
                               fe_analyt)
    return

def compare_heatexp_aniso_2d(xe = np.array([[[-1.,-1.],
                                             [1.,-1.],
                                             [1.,1.],
                                             [-1.,1.]]]),
                             DeltaT=np.ones(4)):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    c = np.random.rand(3,3)
    c = c + c.T
    #
    a = np.random.rand(2,2)
    a = a + a.T
    #
    fe_quad = _fk_heatexp_2d(xe=xe,
                             c=c,
                             a=a,
                             DeltaT=DeltaT)
    #
    fe_analyt = fk_heatexp_aniso_2d(c=c,a=a,
                                    DeltaT=DeltaT,
                                    l=l)
    #
    np.testing.assert_allclose(fe_quad[0],
                               fe_analyt)
    return

def compare_heatexp_iso_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                           [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           DeltaT=np.ones(8)):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    c = isotropic_3d(E=1.,nu=0.3)
    #
    a = 0.05
    #
    fe_quad = _fk_heatexp_3d(xe=xe,
                             c=c,
                             a=np.eye(3)*a,
                             DeltaT=DeltaT)
    #
    fe_analyt = fk_heatexp_3d(E=1.,nu=0.3,a=a,
                              DeltaT=DeltaT,
                              l=l)
    #
    np.testing.assert_allclose(fe_quad[0],
                               fe_analyt)
    return

def compare_heatexp_aniso_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                             [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                             DeltaT=np.ones(8)):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    c = np.random.rand(6,6)
    c = c + c.T
    #
    a = np.random.rand(3,3)
    a = a + a.T
    #
    fe_quad = _fk_heatexp_3d(xe=xe,
                             c=c,
                             a=a,
                             DeltaT=DeltaT)
    #
    fe_analyt = fk_heatexp_aniso_3d(c=c,a=a,
                                    DeltaT=DeltaT,
                                    l=l)
    #
    np.testing.assert_allclose(fe_quad[0],
                               fe_analyt)
    return

def compare_bodyforce_2d(xe = np.array([[[-1.,-1.],
                                         [1.,-1.],
                                         [1.,1.],
                                         [-1.,1.]]])):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lf_bodyforce_2d(xe=xe)
    #
    Ke_analyt = lf_bodyforce_2d(l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_bodyforce_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                         [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lf_bodyforce_3d(xe=xe)
    #
    Ke_analyt = lf_bodyforce_3d(l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_mass_2d(xe = np.array([[[-1.,-1.],
                                    [1.,-1.],
                                    [1.,1.],
                                    [-1.,1.]]])):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lm_mass_2d(xe=xe)
    #
    Ke_analyt = lm_mass_2d(l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_mass_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lm_mass_3d(xe=xe)
    #
    Ke_analyt = lm_mass_3d(l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_laplacian_2d(xe = np.array([[[-1.,-1.],
                                         [1.,-1.],
                                         [1.,1.],
                                         [-1.,1.]]]),
                         k = np.eye(2)):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lk_poisson_2d(xe=xe,k=k)
    #
    Ke_analyt = lk_poisson_2d(k=1,l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_laplacian_aniso_2d(xe = np.array([[[-1.,-1.],
                                               [1.,-1.],
                                               [1.,1.],
                                               [-1.,1.]]]),
                               k = np.eye(2)):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lk_poisson_2d(xe=xe,k=k)
    #
    Ke_analyt = lk_poisson_aniso_2d(k=k,l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_laplacian_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                         [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                         k = np.eye(3)):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lk_poisson_3d(xe=xe,k=k)
    #
    Ke_analyt = lk_poisson_3d(k=1,l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt,
                               rtol=0,
                               atol=1e-14)
    return

def compare_laplacian_iso_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],
                                             [1,1,-1],[-1,1,-1],
                                             [-1,-1,1],[1,-1,1],
                                             [1,1,1],[-1,1,1]]]),
                             k = np.eye(3)):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lk_poisson_3d(xe=xe,k=k)
    #
    Ke_analyt = lk_poisson_aniso_3d(k=k,l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt,
                               rtol=0,
                               atol=1e-14)
    return

def compare_laplacian_aniso_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],
                                               [1,1,-1],[-1,1,-1],
                                               [-1,-1,1],[1,-1,1],
                                               [1,1,1],[-1,1,1]]]),
                         k = np.eye(3)):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lk_poisson_3d(xe=xe,k=k)
    #
    Ke_analyt = lk_poisson_3d(k=1,l=l)
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
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lk_linear_elast_2d(xe=xe,c=c)
    #
    Ke_analyt = lk_linear_elast_aniso_2d(c=c,l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_elast_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                     [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                     c = isotropic_3d()):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lk_linear_elast_3d(xe=xe,c=c)
    #
    Ke_analyt = lk_linear_elast_aniso_3d(c=c,l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_elast_iso_2d(xe = np.array([[[-1.,-1.],
                                     [1.,-1.],
                                     [1.,1.],
                                     [-1.,1.]]]),
                     c = isotropic_2d()):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lk_linear_elast_2d(xe=xe,c=c)
    #
    Ke_analyt = lk_linear_elast_2d(l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return

def compare_elast_iso_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                     [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                     c = isotropic_3d()):
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    #
    Ke_quad = _lk_linear_elast_3d(xe=xe,c=c)
    #
    Ke_analyt = lk_linear_elast_3d(E=1.,nu=0.3,l=l)
    #
    np.testing.assert_allclose(Ke_quad[0],
                               Ke_analyt)
    return


if __name__ == "__main__":
    #
    compare_bodyforce_2d(xe=2*np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                        [[-1,-1],[1,-1],[1,1],[-1,1]]]))
    compare_bodyforce_3d(xe=2*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                         [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]))
    #
    compare_mass_2d(xe=2*np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                   [[-1,-1],[1,-1],[1,1],[-1,1]]]))
    compare_mass_3d(xe=2*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]))
    #
    compare_laplacian_aniso_2d(xe=2*np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                              [[-1,-1],[1,-1],[1,1],[-1,1]]]))
    compare_laplacian_aniso_3d(xe=2*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                               [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]))
    #
    compare_laplacian_2d(xe=2*np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                        [[-1,-1],[1,-1],[1,1],[-1,1]]]))
    compare_laplacian_iso_3d(xe=2*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                             [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]))
    #
    compare_elast_iso_2d(xe=2*np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                        [[-1,-1],[1,-1],[1,1],[-1,1]]]))
    compare_elast_iso_3d(xe=2*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                         [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]))
    #
    compare_elast_2d(xe=2*np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                    [[-1,-1],[1,-1],[1,1],[-1,1]]]))
    compare_elast_3d(xe=2*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                     [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]))
    #
    compare_heatexp_iso_2d(xe=2*np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                          [[-1,-1],[1,-1],[1,1],[-1,1]]]))
    compare_heatexp_iso_2d(xe=2*np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                          [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           DeltaT=None)
    #
    compare_heatexp_aniso_2d(xe=2*np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                            [[-1,-1],[1,-1],[1,1],[-1,1]]]))
    compare_heatexp_aniso_2d(xe=2*np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                            [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                             DeltaT=None)
    #
    compare_heatexp_iso_3d(xe=2*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                           [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]))
    compare_heatexp_iso_3d(xe=2*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                           [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           DeltaT=None)

    compare_heatexp_aniso_3d(xe=2*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                             [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]))
    compare_heatexp_aniso_3d(xe=2*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                             [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                             DeltaT=None)
