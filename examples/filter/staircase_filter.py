# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
from scipy.special import hyp2f1
import matplotlib.pyplot as plt

from topoptlab.filter.haeviside_projection import find_eta, eta_projection

def staircase1(x : np.ndarray, 
              eps : float = 1.05, 
              freq: float = 10.) -> np.ndarray:
    
    return x - 0.5 + (1/np.pi) * np.arctan2(np.sin(2*np.pi*freq*x), 
                                            eps - np.cos(2*np.pi*freq*x))

def staircase2(x : np.ndarray, 
               nsteps: float = 10.) -> np.ndarray:
    steps = np.linspace(0,1,nsteps+2)[1:-1]
    return 

if __name__ == "__main__":
    #
    n=10
    volfrac = 0.5
    beta = 100
    # 
    x = np.linspace(0,1,1001)[:,None]
    #
    #
    fig, ax = plt.subplots()
    # original
    ax.plot(x,x,label="x")
    # plot volume conservation
    ax.plot(x,
            staircase2(x),
            label="eta")
    ax.legend()
    plt.show()
    
