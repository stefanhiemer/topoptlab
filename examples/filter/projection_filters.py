# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import matplotlib.pyplot as plt

from topoptlab.filter.haeviside_projection import find_eta, eta_projection,\
                                                   find_multieta, multieta_projection

if __name__ == "__main__":
    #
    volfrac = 0.5
    beta = 32
    weights = np.array([0.7, 0.3])
    #
    x = np.linspace(0,1,1001)[:,None]
    #
    eta = find_eta(xTilde=x, beta=beta, eta0=0.5, volfrac=volfrac)
    x_eta = eta_projection(xTilde=x, eta=eta, beta=beta)
    #
    etas = np.array([0.3, 0.7])
    x_multieta = multieta_projection(etas=etas, xTilde=x, beta=beta,
                                     weights=weights)
    #
    fig, ax = plt.subplots(figsize=(6, 4))
    #ax.plot(x, x,
    #        color="gray", linestyle="--", linewidth=1.2,
    #        label=r"$\tilde{x}$")
    ax.plot(x, x_eta,
            color="#1f77b4", linewidth=2.0,
            label=rf"single-$\eta$  ($\eta={eta:.3f}$)")
    ax.plot(x, x_multieta,
            color="#d62728", linewidth=2.0,
            label=(rf"multi-$\eta$  "
                   rf"($\eta=[{etas[0]:.2f},\,{etas[1]:.2f}]$, "
                   rf"$w=[0.7,\,0.3]$)"))
    ax.set_xlabel(r"$\tilde{x}$", fontsize=12)
    ax.set_ylabel(r"$\bar{x}$", fontsize=12)
    #
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    #ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    plt.savefig("projection.pdf", 
                format="pdf")
    plt.show()
