from functools import partial

import numpy as np

from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
# map element data to img/voxel
from topoptlab.material_interpolation import heatexpcoeff 
from topoptlab.bounds.hashin_shtrikman_3d import conductivity_binary_low, conductivity_binary_upp
from topoptlab.bounds.hashin_shtrikman_3d import conductivity_nary_low, conductivity_nary_upp

def even_spaced_ternary(npoints):
    fracs = [] 
    for i,a in enumerate(np.linspace(0.0,1.0,npoints)):
        for b in np.linspace(0.0,1.0,npoints)[:(npoints-i)]:
            fracs.append([a,b,1-a-b])
    return fracs

def show_conductivities(ncomp=3):
    #
    npoints = 11
    #
    x = np.linspace(0,1,npoints)
    #
    if ncomp == 2:
        #
        fig,ax = plt.subplots(1,1)
        
        #
        ax.plot(x,conductivity_binary_low(x, kmin = 1e-2, kmax = 1.), 
                label="binary")
        ax.plot(x,conductivity_binary_upp(x, kmin = 1e-2, kmax = 1.), 
                label="binary")
        #
        x = x[:,None]
        ax.plot(x,conductivity_nary_low(x, ks = np.array([1,1e-2])), 
                label="nary")
        ax.plot(x,conductivity_nary_upp(x, ks = np.array([1,1e-2])), 
                label="nary")
        #
        ax.set_xlabel("vol. frac phase 1")
        ax.set_ylabel("conductivity")
        #
        ax.set_xlim(0,1)
        #
        ax.legend()
    elif ncomp == 3:
        #
        x = np.array(even_spaced_ternary(npoints))[:,:2]
        #
        klow = conductivity_nary_low(x, 
                                     ks=np.array([1,1e-1,1e-2]))
        kupp = conductivity_nary_upp(x, 
                                     ks=np.array([1,1e-1,1e-2]))
        #
        fig = plt.figure(figsize=plt.figaspect(2.))
        #
        ax3d = fig.add_subplot(2, 1, 1, projection='3d')
        #
        ax3d.scatter(x[:,0], x[:,1], 
                     klow, 
                     c="b",
                     linewidth=0, 
                     antialiased=False,
                     label="low")
        ax3d.scatter(x[:,0], x[:,1], 
                     kupp, 
                     c="r",
                     linewidth=0, 
                     antialiased=False,
                     label="upp")
        #
        ax3d.set_xlabel("vol. frac phase 1")
        ax3d.set_ylabel("vol. frac phase 2")
        ax3d.set_zlabel("conductivity")
        #
        ax3d.set_xlim(0,1)
        ax3d.set_ylim(0,1)
        ax3d.set_zlim(0,1)
        #
        ax2d = fig.add_subplot(2, 1, 2)
        for x_i in np.linspace(0,1,npoints):
            #
            mask = x[:,1] == x_i
            #
            #ax2d.scatter(x[mask,0],klow[mask],
            #             c="b")
            ax2d.plot(x[mask,0],klow[mask],
                      c="b")
            #ax2d.scatter(x[mask,0],kupp[mask],
            #             c="r")
            ax2d.plot(x[mask,0],kupp[mask],
                      c="r")
        #
        ax2d.set_xlabel("vol. frac phase 1")
        ax2d.set_ylabel("conductivity")
        
    plt.show()
    return

def show_heat_exp():
    
    fig,ax = plt.subplots(2,2)
    for kappa2 in np.linspace(0,2,11)[1:]:
        for a2 in np.linspace(0,2,11)[1:]: 
            if kappa2 == 1:
                continue 
            elif a2 == 1:
                continue
            elif kappa2 < 1 and a2 < 1:
                i,j = 0,0
            elif kappa2 > 1 and a2 < 1:
                i,j = 0,1
            elif kappa2 < 1 and a2 > 1:
                i,j = 1,0
            elif kappa2 > 1 and a2 > 1:
                i,j = 1,1
            #
            if kappa2 > 1:
                kappa = np.linspace(1.,kappa2,11) 
            else:
                kappa = np.linspace(kappa2,1.,11) 
            ax[i,j].plot(kappa,
                    heatexpcoeff(kappa=kappa, 
                                 a1=np.ones(kappa.shape)*1e-3,
                                 a2=np.ones(kappa.shape)*a2*1e-3,
                                 kappa1=np.ones(kappa.shape),
                                 kappa2=np.ones(kappa.shape)*kappa2))
    plt.show()
    
    return

# The real main driver
if __name__ == "__main__":
    #show_conductivities()
    x = np.linspace(0,1,11)[:,None]
    #
    from topoptlab.bounds.hashin_shtrikman_3d import conductivity_binary_low_dx,conductivity_nary_low_dx
    from scipy.differentiate import derivative,jacobian
    from functools import partial
    #
    jac = np.zeros(x.shape)
    k0 = conductivity_nary_low(x, ks = np.array([1e-2,1.])) 
    d = 1e-9
    for i in np.arange(x.shape[1]):
        dx = np.zeros(x.shape)
        dx[:,i] += d
        jac[:,i] = (conductivity_nary_low(x+dx, ks = np.array([1e-2,1.]))-k0)/d
    #
    print(conductivity_nary_low_dx(x, ks = np.array([1e-2,1.])))
    print(jac)