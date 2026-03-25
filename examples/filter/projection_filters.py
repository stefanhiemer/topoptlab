# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import matplotlib.pyplot as plt

from topoptlab.filter.haeviside_projection import find_eta, eta_projection

if __name__ == "__main__":
    #
    n=10
    volfrac = 0.5
    beta = 100
    # 
    x = np.linspace(0,1,1001)[:,None]
    #
    
    #
    x_eta = eta_projection(xTilde=x,
                           eta=find_eta(xTilde=x,
                                        beta=beta,
                                        eta0=0.5,volfrac=volfrac),
                           beta=beta)
    #
    fig, ax = plt.subplots()
    # original
    ax.plot(x,x,label="x")
    # plot volume conservation
    ax.plot(x,eta_projection(xTilde=x,
                           eta=find_eta(xTilde=x,
                                        beta=beta,
                                        eta0=0.5,volfrac=volfrac),
                           beta=beta),
            label="eta")
    ax.legend()
    plt.show()
    
