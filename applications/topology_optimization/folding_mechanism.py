from itertools import product
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from topoptlab.legacy.folding_mechanism import main

def run_examples():
    #
    volfrac = 0.4
    penal = 5.0
    #
    for nelx,rmin,ft,solver in product([60,120,240],
                                       [2.4,4.8,9.6],
                                       [0,1],
                                       ["mma","oc"]):
        filename = "folding_"+"-".join([str(nelx),str(rmin),
                                        ["density","sensitivity"][ft],
                                         solver])
        main(nelx=nelx, nely=int(nelx/3), volfrac=volfrac, penal=penal, rmin=rmin, 
         ft=ft, passive=False,pde=False,solver=solver,expansion=0.05,
         nouteriter=100,display=False,
         file=filename)
    
    return

def postprocess():
    
    from topoptlab.utils import parse_logfile_old
    #
    files = glob("*log")
    #
    data = [parse_logfile_old(file) for file in files]
    #
    nely = np.array([int(d[0]["nelx"]/3) for d in data])
    #
    u_final = np.array([d[-1][0] for d in data])
    #
    print([d[1] for d in data])
    u_last = np.array([d[1][-1,1] for d in data])
    #
    rmin = np.array([d[0]["rmin"] for d in data])
    #
    res = u_final/nely
    #
    inds = np.nonzero(res+0.05 < 0)
    print(inds)
    #
    print([data[i][0] for i in inds[0]])
    #
    fig,axs = plt.subplots(1,2)
    #
    axs[0].scatter(rmin/nely,u_final/nely)
    axs[0].axhline(y=-0.05)
    axs[0].set_xlabel(r"$r/n_{y}$")
    axs[0].set_ylabel(r"$-u_{bw}/n_{y}$")
    #
    axs[1].scatter(u_last/nely,u_final/nely)
    axs[1].set_xlabel(r"$-u_{grey}/n_{y}$")
    axs[1].set_ylabel(r"$-u_{bw}/n_{y}$")
    #
    plt.show()
    return

if __name__ == "__main__":
    # Default input parameters
    nelx = 120
    nely = int(nelx/3)
    volfrac = 0.4
    rmin = 2.4#0.04*nelx  # 5.4
    penal = 5.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    #main(nelx=nelx, nely=int(nelx/3), volfrac=volfrac, penal=penal, rmin=rmin, 
    # ft=ft, passive=False,pde=True,solver="oc",expansion=1.,
    # nouteriter=100,display=False,
    # file="folding_"+"-".join([str(nelx),str(rmin),
    #                            ["sensitivity","density"][ft],
    #                            "oc"]))
    postprocess()
    
    
