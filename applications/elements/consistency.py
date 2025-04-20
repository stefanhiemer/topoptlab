import numpy as np
# different elements/physics
from topoptlab.stiffness_tensors import isotropic_2d, isotropic_3d
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d,_lk_linear_elast_2d
from topoptlab.elements.linear_elasticity_3d import _lk_linear_elast_3d,lk_linear_elast_aniso_3d
from topoptlab.elements.poisson_2d import lk_poisson_2d,_lk_poisson_2d,lk_poisson_aniso_2d
from topoptlab.elements.poisson_3d import lk_poisson_3d,_lk_poisson_3d,lk_poisson_aniso_3d
from topoptlab.elements.mass_2d import _lm_mass_2d, lm_mass_symfem
from topoptlab.elements.mass_3d import _lm_mass_3d, lm_mass_3d
from topoptlab.elements.heatexpansion_2d import fk_heatexp_2d,_fk_heatexp_2d,fk_heatexp_aniso_2d
from topoptlab.elements.heatexpansion_3d import fk_heatexp_3d,_fk_heatexp_3d,fk_heatexp_aniso_3d

def compare_heatexp_aniso_2d(xe = np.array([[[-1.,-1.],
                                             [1.,-1.], 
                                             [1.,1.], 
                                             [-1.,1.]]]),
                             DeltaT=np.ones(4)):
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
                                    DeltaT=DeltaT)
    #
    np.testing.assert_allclose(fe_quad[0],
                               fe_analyt)
    return

def compare_heatexp_iso_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           DeltaT=np.ones(8)):
    #
    c = isotropic_3d()
    #
    a = 0.05
    #
    fe_quad = _fk_heatexp_3d(xe=xe,
                             c=c,
                             a=np.eye(3)*a,
                             DeltaT=DeltaT)
    #
    fe_analyt = fk_heatexp_3d(E=1.,nu=0.3,a=a,
                              DeltaT=DeltaT)
    #
    np.testing.assert_allclose(fe_quad[0],
                               fe_analyt)
    return

def compare_heatexp_aniso_3d(xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                             DeltaT=np.ones(8)):
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
                                    DeltaT=DeltaT)
    #
    np.testing.assert_allclose(fe_quad[0],
                               fe_analyt)
    return

def compare_mass(xe = np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                [[-2,-2.1],[2.1,-2],[2,2],[-2,2]]])):
    if xe.shape[-1] == 2:
        #
        Kes = np.vstack([_lm_mass_2d(xe[i]) for i in range(xe.shape[0])])
        #
        np.testing.assert_allclose(_lm_mass_2d(xe),
                        Kes)
    elif xe.shape[-1] == 3:
        #
        Kes = np.vstack([_lm_mass_3d(xe[i]) for i in range(xe.shape[0])])
        #
        np.testing.assert_allclose(_lm_mass_3d(xe),
                                   Kes)
    return

def compare_laplacian(xe = np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                     [[-2,-2.1],[2.1,-2],[2,2],[-2,2]]]),
                      k = np.stack((np.eye(2),2*np.eye(2)))):
    #
    if xe.shape[-1] == 2:
        #
        Kes = np.vstack([_lk_poisson_2d(xe=xe[i],k=k[i]) \
                         for i in range(xe.shape[0])])
        #
        np.testing.assert_allclose(_lk_poisson_2d(xe=xe,k=k),
                                   Kes)
    elif xe.shape[-1] == 3:
        #
        Kes = np.vstack([_lk_poisson_3d(xe=xe[i],k=k[i]) \
                         for i in range(xe.shape[0])])
        #
        np.testing.assert_allclose(_lk_poisson_3d(xe=xe,k=k),
                                   Kes)
    return

def compare_linelast(xe = np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                     [[-2,-2.1],[2.1,-2],[2,2],[-2,2]]]),
                     c = np.stack((isotropic_2d(),2*isotropic_2d())) ):
    #
    if xe.shape[-1] == 2:
        #
        Kes = np.vstack([_lk_linear_elast_2d(xe=xe[i],c=c[i]) \
                         for i in range(xe.shape[0])])
        #
        np.testing.assert_allclose(_lk_linear_elast_2d(xe=xe,c=c),
                                   Kes)
    elif xe.shape[-1] == 3:
        #
        Kes = np.vstack([_lk_linear_elast_3d(xe=xe[i],c=c[i]) \
                         for i in range(xe.shape[0])])
        #
        np.testing.assert_allclose(_lk_linear_elast_3d(xe=xe,c=c),
                                   Kes)
    return

def compare_heatexp(xe = np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                   [[-2,-2.1],[2.1,-2],[2,2],[-2,2]]]),
                    DeltaT=np.ones(4)):
    ndim = xe.shape[-1]
    a = np.stack([ np.eye(ndim)*(i+1) for i in range(xe.shape[0])] )
    #
    if xe.shape[-1] == 2:
        c = np.stack([isotropic_2d()*(i+1) for i in range(xe.shape[0])])
        #
        Kes = np.vstack([_fk_heatexp_2d(xe=xe[i],c=c[i],a=a[i]) \
                         for i in range(xe.shape[0])])
        #
        np.testing.assert_allclose(_fk_heatexp_2d(xe=xe,c=c,a=a),
                                   Kes)
    elif xe.shape[-1] == 3:
        c = np.stack([isotropic_3d()*(i+1) for i in range(xe.shape[0])])
        #
        Kes = np.vstack([_fk_heatexp_3d(xe=xe[i],c=c[i],a=a[i]) \
                         for i in range(xe.shape[0])])
        #
        np.testing.assert_allclose(_fk_heatexp_3d(xe=xe,c=c,a=a),
                                   Kes)
    return


if __name__ == "__main__":
    #
    compare_mass()
    compare_mass(xe=np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                               [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                              [[-2.1,-2,-2],[2.1,-2,-2],[2,2.1,-2],[-2,2,-2],
                                [-2,-2,2],[2,-2,2.1],[2,2,2],[-2,2.1,2]]]))
    compare_laplacian()
    compare_laplacian(xe=np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                   [[-2.1,-2,-2],[2.1,-2,-2],[2,2.1,-2],[-2,2,-2],
                                    [-2,-2,2],[2,-2,2.1],[2,2,2],[-2,2.1,2]]]),
                      k = np.stack((np.eye(3),2*np.eye(3))))
    #
    compare_linelast()
    compare_linelast(xe=np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                   [[-2.1,-2,-2],[2.1,-2,-2],[2,2.1,-2],[-2,2,-2],
                                    [-2,-2,2],[2,-2,2.1],[2,2,2],[-2,2.1,2]]]),
                      c = np.stack((isotropic_3d(),2*isotropic_3d())))
    #
    compare_heatexp()
    compare_heatexp(DeltaT=None)
    compare_heatexp(xe=np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                 [[-2.1,-2,-2],[2.1,-2,-2],[2,2.1,-2],[-2,2,-2],
                                  [-2,-2,2],[2,-2,2.1],[2,2,2],[-2,2.1,2]]]),
                     DeltaT=np.ones(8))
    compare_heatexp(xe=np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                 [[-2.1,-2,-2],[2.1,-2,-2],[2,2.1,-2],[-2,2,-2],
                                  [-2,-2,2],[2,-2,2.1],[2,2,2],[-2,2.1,2]]]),
                     DeltaT=None)
    
