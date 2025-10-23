# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

import matplotlib.pyplot as plt
from topoptlab.utils import even_spaced_ternary

from topoptlab.bounds.hashin_shtrikman_3d import conductivity_binary_low, conductivity_binary_upp
from topoptlab.bounds.hashin_shtrikman_3d import conductivity_nary_low, conductivity_nary_upp

def show_conductivities(ncomp=3):
    #
    npoints = 21
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
        ax2d.set_xlim(0,1)
        
    plt.show()
    return

from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_binary_low, bulkmod_binary_upp
from topoptlab.bounds.hashin_shtrikman_3d import shearmod_binary_low, shearmod_binary_upp
from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_nary_low,bulkmod_nary_upp
from topoptlab.bounds.hashin_shtrikman_3d import shearmod_nary_low,shearmod_nary_upp

def show_bulkshearmodulus(ncomp=3):
    #
    npoints = 21
    #
    x = np.linspace(0,1,npoints)
    #
    if ncomp == 2:
        #
        fig,axs = plt.subplots(1,2,sharex=True,sharey=True)
        
        #
        axs[0].plot(x,bulkmod_binary_low(x, 
                                         Kmin = 1e-2, Kmax = 1,
                                         Gmin=1e-2,Gmax=1.), 
                    label="binary lower")
        axs[0].plot(x,bulkmod_binary_upp(x, 
                                         Kmin=1e-2, Kmax=1,
                                         Gmin=1e-2, Gmax=1.), 
                    label="binary upper")
        #
        axs[1].plot(x,shearmod_binary_low(x, 
                                          Kmin = 1e-2, Kmax = 1,
                                          Gmin=1e-2,Gmax=1.))
        axs[1].plot(x,shearmod_binary_upp(x, 
                                         Kmin=1e-2, Kmax=1,
                                         Gmin=1e-2, Gmax=1.))
        #
        x = x[:,None]
        axs[0].plot(x,bulkmod_nary_low(x, 
                                       Ks = np.array([1,1e-2]),
                                       Gs = np.array([1,1e-2])), 
                    label="n-ary lower")
        axs[0].plot(x,bulkmod_nary_upp(x, 
                                       Ks = np.array([1,1e-2]),
                                       Gs = np.array([1,1e-2])), 
                    label="n-nary upper")
        #
        axs[1].plot(x,shearmod_nary_low(x, 
                                       Ks = np.array([1,1e-2]),
                                       Gs = np.array([1,1e-2])))
        axs[1].plot(x,shearmod_nary_upp(x, 
                                       Ks = np.array([1,1e-2]),
                                       Gs = np.array([1,1e-2])))
        
        axs[0].set_xlabel("vol. frac phase 1")
        axs[0].set_ylabel("bulk modulus")
        axs[1].set_xlabel("vol. frac phase 1")
        axs[1].set_ylabel("shear modulus")
        #
        axs[0].set_xlim(0,1)
        #
        fig.legend()
    elif ncomp == 3:
        #
        x = np.array(even_spaced_ternary(npoints))[:,:2]
        #
        Klow = bulkmod_nary_low(x,
                                Ks = np.array([1,1e-1,1e-2]),
                                Gs = np.array([1,1e-1,1e-2]))
        Kupp = bulkmod_nary_upp(x,
                                Ks = np.array([1,1e-1,1e-2]),
                                Gs = np.array([1,1e-1,1e-2]))
        #
        Glow = shearmod_nary_low(x,
                                 Ks = np.array([1,1e-1,1e-2]),
                                 Gs = np.array([1,1e-1,1e-2]))
        Gupp = shearmod_nary_upp(x,
                                 Ks = np.array([1,1e-1,1e-2]),
                                 Gs = np.array([1,1e-1,1e-2]))
        #
        fig = plt.figure(figsize=plt.figaspect(2.))
        #
        ax3d = fig.add_subplot(2, 2, 1, projection='3d')
        #
        ax3d.scatter(x[:,0], x[:,1], 
                     Klow, 
                     c="b",
                     linewidth=0, 
                     antialiased=False,
                     label="K low")
        ax3d.scatter(x[:,0], x[:,1], 
                     Kupp, 
                     c="r",
                     linewidth=0, 
                     antialiased=False,
                     label="K upp")
        #
        ax3d.set_xlabel("vol. frac phase 1")
        ax3d.set_ylabel("vol. frac phase 2")
        ax3d.set_zlabel("bulk modulus")
        #
        ax3d.set_xlim(0,1)
        ax3d.set_ylim(0,1)
        ax3d.set_zlim(0,1)
        #
        ax3d = fig.add_subplot(2, 2, 2, projection='3d')
        #
        ax3d.scatter(x[:,0], x[:,1], 
                     Glow, 
                     c="b",
                     linewidth=0, 
                     antialiased=False,
                     label="G low")
        ax3d.scatter(x[:,0], x[:,1], 
                     Gupp, 
                     c="r",
                     linewidth=0, 
                     antialiased=False,
                     label="G upp")
        #
        ax3d.set_xlabel("vol. frac phase 1")
        ax3d.set_ylabel("vol. frac phase 2")
        ax3d.set_zlabel("shear modulus")
        #
        ax3d.set_xlim(0,1)
        ax3d.set_ylim(0,1)
        ax3d.set_zlim(0,1)
        #
        ax2d = fig.add_subplot(2, 2, 3)
        for x_i in np.linspace(0,1,npoints):
            #
            mask = x[:,1] == x_i
            #
            #ax2d.scatter(x[mask,0],klow[mask],
            #             c="b")
            ax2d.plot(x[mask,0],Klow[mask],
                      c="b")
            #ax2d.scatter(x[mask,0],kupp[mask],
            #             c="r")
            ax2d.plot(x[mask,0],Kupp[mask],
                      c="r")
        #
        ax2d.set_xlabel("vol. frac phase 1")
        ax2d.set_ylabel("bulk modulus")
        ax2d.set_xlim(0,1)
        #
        ax2d = fig.add_subplot(2, 2, 4)
        for x_i in np.linspace(0,1,npoints):
            #
            mask = x[:,1] == x_i
            #
            #ax2d.scatter(x[mask,0],klow[mask],
            #             c="b")
            ax2d.plot(x[mask,0],Glow[mask],
                      c="b")
            #ax2d.scatter(x[mask,0],kupp[mask],
            #             c="r")
            ax2d.plot(x[mask,0],Gupp[mask],
                      c="r")
        #
        ax2d.set_xlabel("vol. frac phase 1")
        ax2d.set_ylabel("shear modulus")
        ax2d.set_xlim(0,1)
    plt.show()
    return

from topoptlab.bounds.hashin_shtrikman_3d import emod_binary_low,emod_binary_upp
from topoptlab.bounds.hashin_shtrikman_3d import poiss_binary_low,poiss_binary_upp
from topoptlab.bounds.hashin_shtrikman_3d import emod_nary_low,emod_nary_upp
from topoptlab.bounds.hashin_shtrikman_3d import poiss_nary_low,poiss_nary_upp

def show_youngmoduluspoiss(ncomp=2):
    #
    npoints = 21
    #
    x = np.linspace(0,1,npoints)
    #
    if ncomp == 2:
        #
        fig,axs = plt.subplots(1,2,sharex=True,sharey=False)
        
        #
        axs[0].plot(x,emod_binary_low(x, 
                                      Kmin = 1e-2, Kmax = 1,
                                      Gmin=1e-2,Gmax=1.), 
                    label="binary lower")
        axs[0].plot(x,emod_binary_upp(x, 
                                      Kmin=1e-2, Kmax=1,
                                      Gmin=1e-2, Gmax=1.), 
                    label="binary upper")
        #
        axs[1].plot(x,poiss_binary_low(x, 
                                       Kmin = 1e-2, Kmax = 1,
                                       Gmin=1e-2,Gmax=1.))
        axs[1].plot(x,poiss_binary_upp(x, 
                                       Kmin=1e-2, Kmax=1,
                                       Gmin=1e-2, Gmax=1.))
        #
        x = x[:,None]
        axs[0].plot(x,emod_nary_low(x, 
                                    Ks = np.array([1,1e-2]),
                                    Gs = np.array([1,1e-2])), 
                    label="n-ary lower")
        axs[0].plot(x,emod_nary_upp(x, 
                                    Ks = np.array([1,1e-2]),
                                    Gs = np.array([1,1e-2])), 
                    label="n-nary upper")
        #
        axs[1].plot(x,poiss_nary_low(x, 
                                     Ks = np.array([1,1e-2]),
                                     Gs = np.array([1,1e-2])))
        axs[1].plot(x,poiss_nary_upp(x, 
                                     Ks = np.array([1,1e-2]),
                                     Gs = np.array([1,1e-2])))
        axs[0].set_xlabel("vol. frac phase 1")
        axs[0].set_ylabel("Young's modulus")
        axs[1].set_xlabel("vol. frac phase 1")
        axs[1].set_ylabel("Poisson's ratio")
        #
        axs[0].set_xlim(0,1)
        #
        fig.legend()
    elif ncomp == 3:
        #
        x = np.array(even_spaced_ternary(npoints))[:,:2]
        #
        Ylow = emod_nary_low(x,
                             Ks = np.array([1,1e-1,1e-2]),
                             Gs = np.array([1,1e-1,1e-2]))
        Yupp = emod_nary_upp(x,
                             Ks = np.array([1,1e-1,1e-2]),
                             Gs = np.array([1,1e-1,1e-2]))
        #
        vlow = poiss_nary_low(x,
                              Ks = np.array([1,1e-1,1e-2]),
                              Gs = np.array([1,1e-1,1e-2]))
        vupp = poiss_nary_upp(x,
                              Ks = np.array([1,1e-1,1e-2]),
                              Gs = np.array([1,1e-1,1e-2]))
        #
        fig = plt.figure(figsize=plt.figaspect(2.))
        #
        ax3d = fig.add_subplot(2, 2, 1, projection='3d')
        #
        ax3d.scatter(x[:,0], x[:,1], 
                     Ylow, 
                     c="b",
                     linewidth=0, 
                     antialiased=False,
                     label="E low")
        ax3d.scatter(x[:,0], x[:,1], 
                     Yupp, 
                     c="r",
                     linewidth=0, 
                     antialiased=False,
                     label="E upp")
        #
        ax3d.set_xlabel("vol. frac phase 1")
        ax3d.set_ylabel("vol. frac phase 2")
        ax3d.set_zlabel("Young's modulus")
        #
        ax3d.set_xlim(0,1)
        ax3d.set_ylim(0,1)
        ax3d.set_zlim(bottom=0)
        #
        ax3d = fig.add_subplot(2, 2, 2, projection='3d')
        #
        ax3d.scatter(x[:,0], x[:,1], 
                     vlow, 
                     c="b",
                     linewidth=0, 
                     antialiased=False,
                     label="v low")
        ax3d.scatter(x[:,0], x[:,1], 
                     vupp, 
                     c="r",
                     linewidth=0, 
                     antialiased=False,
                     label="v upp")
        #
        ax3d.set_xlabel("vol. frac phase 1")
        ax3d.set_ylabel("vol. frac phase 2")
        ax3d.set_zlabel("Poisson's ratio")
        #
        ax3d.set_xlim(0,1)
        ax3d.set_ylim(0,1)
        ax3d.set_zlim(-1,1)
        #
        ax2d = fig.add_subplot(2, 2, 3)
        for x_i in np.linspace(0,1,npoints):
            #
            mask = x[:,1] == x_i
            #
            #ax2d.scatter(x[mask,0],klow[mask],
            #             c="b")
            ax2d.plot(x[mask,0],Ylow[mask],
                      c="b")
            #ax2d.scatter(x[mask,0],kupp[mask],
            #             c="r")
            ax2d.plot(x[mask,0],Yupp[mask],
                      c="r")
        #
        ax2d.set_xlabel("vol. frac phase 1")
        ax2d.set_ylabel("Young's modulus")
        ax2d.set_xlim(0,1)
        #
        ax2d = fig.add_subplot(2, 2, 4)
        for x_i in np.linspace(0,1,npoints):
            #
            mask = x[:,1] == x_i
            #
            #ax2d.scatter(x[mask,0],klow[mask],
            #             c="b")
            ax2d.plot(x[mask,0],vlow[mask],
                      c="b")
            #ax2d.scatter(x[mask,0],kupp[mask],
            #             c="r")
            ax2d.plot(x[mask,0],vupp[mask],
                      c="r")
        #
        ax2d.set_xlabel("vol. frac phase 1")
        ax2d.set_ylabel("Poisson's ratio")
        ax2d.set_xlim(0,1)
    plt.show()
    return

from topoptlab.material_interpolation import heatexpcoeff_binary_iso, simp, bound_interpol
from topoptlab.bounds.hashin_rosen_3d import heatexp_binary_low, heatexp_binary_upp

def show_heat_exp():
    #
    K1 = 76
    K2 = 170
    #
    G1 = 26
    G2 = 82
    #
    a1 = 22.87
    a2 = 12.87
    #
    fig,ax = plt.subplots(1,1)
    #
    x = np.linspace(0,1,21)
    K_bd = bound_interpol(xPhys=x,w=0.5,
                          bd_low=bulkmod_binary_low,
                          bd_upp=bulkmod_binary_upp,
                          bd_kws={"Kmin": K1, "Kmax": K2,
                                  "Gmin": G1, "Gmax": G2})
    a_hs = heatexpcoeff_binary_iso(xPhys=x, K=K_bd,
    #                               K=0.5*(bulkmod_binary_upp(x, 
    #                                                    Kmin = K1, Kmax = K2,
    #                                                    Gmin = G1, Gmax = G2)+\
    #                                      bulkmod_binary_low(x, 
    #                                                    Kmin = K1, Kmax = K2,
    #                                                    Gmin = G1, Gmax = G2)),
                                   amax=a2, amin=a1,
                                   Kmin=K1, Kmax=K2)
    #
    alow = heatexp_binary_low(x,
                              Kmin=K1,Kmax=K2,
                              Gmin=G1,Gmax=G2,
                              amin=a1,amax=a2)
    aupp = heatexp_binary_upp(x,
                              Kmin=K1,Kmax=K2,
                              Gmin=G1,Gmax=G2,
                              amin=a1,amax=a2)
    ax.plot(x,a_hs,label="HS-interpolation",c="b")
    ax.plot(x,aupp,label="upper bound")
    ax.plot(x,alow,label="lower bound")
    ax.legend()
    ax.set_xlabel("vol. frac phase 1")
    ax.set_ylabel("coeff. of thermal expansion")
    ax.set_xlim(0,1)
    plt.show()
    
    return

# The real main driver
if __name__ == "__main__":
    show_conductivities(3)
    show_bulkshearmodulus(3)
    show_heat_exp()
    show_youngmoduluspoiss(3)