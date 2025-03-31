from functools import partial

import numpy as np

from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
# map element data to img/voxel
from topoptlab.material_interpolation import heatexpcoeff 
from topoptlab.bounds.hashin_shtrikman import conductivity_nary_upp

def show_conductivities():
    
    #
    fig,ax = plt.subplots(1,1)
    #
    x = np.linspace(0,1,11)[:,None]
    ax.plot(x,conductivity_nary_upp(x, ks = np.array([1,0.1])) )
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
    show_conductivities()