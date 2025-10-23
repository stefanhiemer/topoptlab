# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

import matplotlib.pyplot as plt
from topoptlab.utils import even_spaced_ternary

from topoptlab.bounds.hashin_shtrikman_3d import conductivity_binary_low, conductivity_binary_upp
from topoptlab.material_interpolation import simp,ramp

def show_conductivities():
    #
    npoints = 21
    font=16
    #
    kmin=1e-2
    kmax=1.
    #
    x = np.linspace(0,1,npoints) 
    #
    fig,ax = plt.subplots(1,1)
    # HS bounds
    ax.plot(x,conductivity_binary_low(x, kmin = kmin, kmax = kmax), 
            label="HS bounds",c="k",linestyle="--")
    ax.plot(x,conductivity_binary_upp(x, kmin = kmin, kmax = kmax), 
            c="k",linestyle="--")
    # interpolations
    ax.plot(x,simp(xPhys=x, eps=kmin/kmax, penal=2.5)*kmax,
                   label="SIMP")
    ax.plot(x,ramp(xPhys=x, eps=kmin/kmax, penal=3)*kmax,
                   label="RAMP")
    #
    ax.set_xlabel("vol. frac.", fontsize=font)
    ax.set_ylabel("conductivity", fontsize=font)
    #
    ax.set_xlim(0,1)
    #
    ax.legend(fontsize=font)
    #
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_yticks(np.linspace(0,1,11))
    ax.set_xticklabels(labels=np.round(np.linspace(0,1,11),1),
                       fontdict={"fontsize": font})
    ax.set_yticklabels(labels=np.round(np.linspace(0,1,11),1), 
                       fontdict={"fontsize": font})
    #
    plt.show()
    return

from topoptlab.bounds.hashin_shtrikman_3d import bulkmod_binary_low, bulkmod_binary_upp
from topoptlab.bounds.hashin_shtrikman_3d import shearmod_binary_low, shearmod_binary_upp

def show_bulkshearmodulus(ncomp=3):
    #
    npoints = 21
    font=16
    #
    Kmin = 1e-2 
    Kmax = 1.
    Gmin=1e-2
    Gmax=1.
    #
    x = np.linspace(0,1,npoints)
    #
    fig,axs = plt.subplots(1,2,sharex=True,sharey=True)
    
    #
    axs[0].plot(x,bulkmod_binary_low(x, 
                                     Kmin=Kmin, Kmax=Kmax,
                                     Gmin=Gmin, Gmax=Gmax), 
                label="HS bounds",c="k",linestyle="--")
    axs[0].plot(x,bulkmod_binary_upp(x, 
                                     Kmin=Kmin, Kmax=Kmax,
                                     Gmin=Gmin, Gmax=Gmax), 
                c="k",linestyle="--")
    #
    axs[1].plot(x,shearmod_binary_low(x, 
                                      Kmin=Kmin, Kmax=Kmax,
                                      Gmin=Gmin, Gmax=Gmax),
                c="k",linestyle="--")
    axs[1].plot(x,shearmod_binary_upp(x, 
                                      Kmin=Kmin, Kmax=Kmax,
                                      Gmin=Gmin, Gmax=Gmax),
                c="k",linestyle="--")
    # interpolations
    axs[0].plot(x,simp(xPhys=x, eps=Kmin/Kmax, penal=3)*Kmax,
                   label="SIMP")
    axs[1].plot(x,simp(xPhys=x, eps=Gmin/Gmax, penal=3)*Gmax)
    axs[0].plot(x,ramp(xPhys=x, eps=Kmin/Kmax, penal=3)*Kmax,
                   label="RAMP")
    axs[1].plot(x,ramp(xPhys=x, eps=Gmin/Gmax, penal=3)*Gmax)
    #
    axs[0].set_xlabel("vol. frac.", fontsize=font)
    axs[0].set_ylabel("bulk modulus", fontsize=font)
    axs[1].set_xlabel("vol. frac.", fontsize=font)
    axs[1].set_ylabel("shear modulus", fontsize=font)
    #
    axs[0].set_xlim(0,1)
    #
    for i in range(2):
        axs[i].set_xticklabels(labels=np.round(np.linspace(0,1,11),1),
                               fontdict={"fontsize": font})
        axs[i].set_yticklabels(labels=np.round(np.linspace(0,1,11),1), 
                               fontdict={"fontsize": font})
    #
    fig.legend(fontsize=font)
    plt.show()
    return

from topoptlab.bounds.hashin_shtrikman_3d import emod_binary_low,emod_binary_upp
from topoptlab.bounds.hashin_shtrikman_3d import poiss_binary_low,poiss_binary_upp

def show_youngmoduluspoiss():
    #
    npoints = 21
    font=15
    #
    Kmin = 1e-2 
    Kmax = 1.
    Gmin=2e-2
    Gmax=1.
    #
    Emin = 9*Kmin*Gmin / (3*Kmin + Gmin)
    Emax = 9*Kmax*Gmax / (3*Kmax + Gmax)
    numin = (3*Kmin - 2*Gmin) / (2*(3*Kmin + Gmin))
    numax = (3*Kmax - 2*Gmax) / (2*(3*Kmax + Gmax))
    #
    x = np.linspace(0,1,npoints)
    #
    fig,axs = plt.subplots(1,2,
                           sharex=True,sharey=False)
    
    #
    axs[0].plot(x,emod_binary_low(x, 
                                  Kmin = Kmin, Kmax = Kmax,
                                  Gmin = Gmin, Gmax = Gmax), 
                label="HS bounds", c="k",linestyle="--")
    axs[0].plot(x,emod_binary_upp(x, 
                                  Kmin = Kmin, Kmax = Kmax,
                                  Gmin = Gmin, Gmax = Gmax), 
                c="k",linestyle="--")
    #
    axs[1].plot(x,poiss_binary_low(x, 
                                   Kmin = Kmin, Kmax = Kmax,
                                   Gmin = Gmin, Gmax = Gmax),
                c="k",linestyle="--")
    axs[1].plot(x,poiss_binary_upp(x, 
                                   Kmin = Kmin, Kmax = Kmax,
                                   Gmin = Gmin, Gmax = Gmax),
                c="k",linestyle="--")
    #
    # interpolations
    axs[0].plot(x,simp(xPhys=x, eps=Emin/Emax, penal=2.5)*Emax,
                   label="SIMP")
    axs[1].plot(x,simp(xPhys=x, eps=numin/numax, penal=3)*numax)
    axs[0].plot(x,ramp(xPhys=x, eps=Emin/Emax, penal=3)*Emax,
                   label="RAMP")
    axs[1].plot(x,ramp(xPhys=x, eps=numin/numax, penal=3)*numax)
    #
    axs[0].set_xlabel("vol. frac.", fontsize=font)
    axs[0].set_ylabel("Young's modulus", fontsize=font)
    axs[1].set_xlabel("vol. frac.", fontsize=font)
    axs[1].set_ylabel("Poisson's ratio", fontsize=font)
    #
    axs[0].set_xlim(0,1)
    #
    for i in range(2):
        axs[i].set_xticklabels(labels=np.round(np.linspace(0,1,11),1),
                               fontdict={"fontsize": font})
        axs[i].set_yticklabels(labels=np.round(np.linspace(0,1,11),1), 
                               fontdict={"fontsize": font})
    #
    axs[0].legend(fontsize=font,loc="upper left",frameon=False)
    #
    fig.subplots_adjust(wspace=0.4) 
    #
    plt.show()
    return

from topoptlab.material_interpolation import heatexpcoeff_binary_iso, bound_interpol
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
    #
    a_simp = heatexpcoeff_binary_iso(xPhys=x, 
                                     K = K2*simp(xPhys=x, eps=K1/K2, penal=3),
                                amax=a2, amin=a1,
                                Kmin=K1, Kmax=K2)
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
    ax.plot(x,a_simp,label="SIMP-interpolation",c="b",linestyle="--")
    ax.plot(x,a_hs,label="HS-interpolation",c="b")
    ax.plot(x,aupp,label="upper bound")
    ax.plot(x,alow,label="lower bound")
    ax.legend()
    ax.set_xlabel("vol. frac.")
    ax.set_ylabel("coeff. of thermal expansion")
    ax.set_xlim(0,1)
    plt.show()
    
    return

# The real main driver
if __name__ == "__main__":
    #show_conductivities()
    #show_bulkshearmodulus()
    #show_heat_exp()
    show_youngmoduluspoiss()