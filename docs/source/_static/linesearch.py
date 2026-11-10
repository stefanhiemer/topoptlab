# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

import matplotlib.pyplot as plt

# 1D restriction of the objective along the search direction d_k:
# phi(alpha) = f(x_k + alpha d_k). Single minimum, so the acceptable
# step sizes form one interval.
def phi(a):
    return 1. - 1.2*a + 0.5*a**2

def dphi(a):
    return -1.2 + 1.0*a

# exaggerated constants for visibility (typical: c1=1e-4, c2=0.9)
c1 = 0.25
c2 = 0.5
amax = 3.2

def intervals(mask, a):
    # contiguous True-intervals of a boolean mask
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return [(a[s], a[e]) for s, e in zip(starts, ends)]

def setup_axes(ax, ymin, ymax):
    ax.annotate("", xy=(amax+0.25, ymin), xytext=(-0.05, ymin),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    ax.annotate("", xy=(0, ymax), xytext=(0, ymin-0.02),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    ax.text(amax+0.3, ymin, r"$\alpha$", fontsize=20, va="center")
    ax.text(0.08, ymax, r"$f(\mathbf{x}_k+\alpha\mathbf{d}_k)$",
            fontsize=18, va="top")
    ax.set_xlim(-0.1, amax+0.6)
    ax.set_ylim(ymin-0.08, ymax+0.02)
    ax.axis("off")
    return

def shade(ax, ivals, ymin, color, label):
    for i, (lo, hi) in enumerate(ivals):
        ax.axvspan(lo, hi, color=color, alpha=0.2, lw=0,
                   label=label if i == 0 else None)
        ax.plot([lo, hi], [ymin, ymin], color=color, lw=6,
                solid_capstyle="butt")
    return

def armijo():
    #
    a = np.linspace(0, amax, 2001)
    ymin, ymax = phi(a).min() - 0.2, 2.0
    #
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(a, phi(a), linestyle="-", color="k", linewidth=4)
    # tangent: predicted decrease
    at = np.linspace(0, 0.9, 101)
    ax.plot(at, phi(0) + at*dphi(0), linestyle="--", color="b",
            linewidth=2, label=r"tangent $\alpha\nabla f^\top\mathbf{d}_k$")
    # start point alpha = 0, i.e. x_k
    ax.plot(0, phi(0), marker="o", markersize=12, color="tab:red",
            zorder=5, clip_on=False)
    ax.annotate(r"$\mathbf{x}_k$", (0, phi(0)),
                textcoords="offset points", xytext=(12, 6),
                fontsize=18)
    #
    setup_axes(ax, ymin, ymax)
    ax.legend(frameon=True, facecolor="white", edgecolor="none",
              framealpha=1., loc="upper center", fontsize=14)
    plt.savefig("linesearch-1.png", format="png",
                bbox_inches="tight")
    plt.show()
    plt.close()
    return

def wolfe():
    #
    a = np.linspace(0, amax, 2001)
    ymin, ymax = phi(a).min() - 0.2, 2.0
    armijo_ok = phi(a) <= phi(0) + c1*a*dphi(0)
    curv_ok = dphi(a) >= c2*dphi(0)
    #
    fig, ax = plt.subplots(figsize=(8,5))
    shade(ax, intervals(armijo_ok & curv_ok, a), ymin, "tab:green",
          "Wolfe satisfied")
    ax.plot(a, phi(a), linestyle="-", color="k", linewidth=4)
    ax.plot(a, phi(0) + c1*a*dphi(0), linestyle="-.", color="b",
            linewidth=2, label="Armijo")
    # tangent at alpha=0 and "desired slope" c2 * phi'(0)
    at = np.linspace(0, 0.9, 101)
    ax.plot(at, phi(0) + at*dphi(0), linestyle="--", color="b",
            linewidth=2)
    # points where the slope has flattened to c2 * phi'(0)
    edges = np.where(np.diff(curv_ok.astype(int)) == 1)[0]
    for i, e in enumerate(edges):
        a0 = a[e]
        seg = np.linspace(max(a0-0.45, 0.), a0+0.45, 51)
        ax.plot(seg, phi(a0) + c2*dphi(0)*(seg-a0), linestyle=":",
                color="b", linewidth=2.5,
                label=r"slope $c_2\nabla f^\top\mathbf{d}_k$"
                      if i == 0 else None)
    #
    setup_axes(ax, ymin, ymax)
    ax.legend(frameon=True, facecolor="white", edgecolor="none",
              framealpha=1., loc="upper center", fontsize=14)
    plt.savefig("linesearch-2.png", format="png",
                bbox_inches="tight")
    plt.show()
    plt.close()
    return

if __name__ == "__main__":
    armijo()
    wolfe()
