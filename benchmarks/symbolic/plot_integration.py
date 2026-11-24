# SPDX-License-Identifier: GPL-3.0-or-later
from itertools import product
from json import load
from pathlib import Path

import matplotlib.pyplot as plt

# categorical colors, one per integration method (fixed order)
colors = {"integrate serial": "#2a78d6",
          "integrate parallel": "#eb6834",
          "separated serial": "#1baf7a",
          "separated parallel": "#eda100"}
# backend encoded by line style and marker
styles = {"sympy": {"linestyle": "-"},
          "symint": {"linestyle": "--"}}
ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"

def plot_integration(infile: Path, outfile: Path) -> None:
    """
    Plot wall clock time per integrated matrix entry vs. number of dimensions
    of the integration benchmark, one column per element and one row per
    element order. Color encodes the
    method, line style and marker the backend. Timeouts are drawn as open
    triangles at the timeout per entry (lower bound).

    Parameters
    ----------
    infile : pathlib.Path
        json file written by integration.py.
    outfile : pathlib.Path
        image file.

    Returns
    -------
    None

    """
    with open(infile) as f:
        data = load(f)
    timeout, results = data["timeout"], data["results"]
    # records without order are order 1 from older runs
    for r in results:
        r.setdefault("order", 1)
    elements = list(dict.fromkeys(r["element"] for r in results))
    orders = sorted(set(r["order"] for r in results))
    #
    fig, axs = plt.subplots(len(orders), len(elements),
                            figsize=(3.2*len(elements), 0.6+3*len(orders)),
                            sharey=True, sharex=True,
                            squeeze=False)
    for (i, order), (j, element) in product(enumerate(orders),
                                            enumerate(elements)):
        ax = axs[i][j]
        for backend, style in styles.items():
            rows = sorted((r for r in results if r["element"] == element
                           and r["backend"] == backend
                           and r["order"] == order),
                          key=lambda r: r["ndim"])
            for method, color in colors.items():
                # skip ndims where the method was not run
                rows_m = [r for r in rows if method in r["times"]]
                ndims = [r["ndim"] for r in rows_m]
                # normalize by number of integrated matrix entries
                times = [None if r["times"][method] is None
                         else r["times"][method]/r["entries"] for r in rows_m]
                touts = [timeout/r["entries"] for r in rows_m]
                # finished runs
                ax.plot([d for d, t in zip(ndims, times) if t is not None],
                        [t for t in times if t is not None],
                        color=color, linewidth=2, markersize=6,
                        **style)
                # timeouts
                tout = [(d, to) for d, t, to in zip(ndims, times, touts)
                        if t is None]
                ax.plot([d for d, _ in tout], [to for _, to in tout],
                        linestyle="none",
                        marker="^", markersize=8, markerfacecolor="none",
                        markeredgecolor=color, markeredgewidth=1.5)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 3])
        # margin so that ticks/markers of neighbouring panels do not touch
        ax.set_xlim(0.7, 3.3)
        if i == 0:
            ax.set_title(element, color=ink)
        if i == len(orders)-1:
            ax.set_xlabel("ndim", color=ink)
        if j == 0:
            ax.set_ylabel(f"order {order}\ntime per entry (s)", color=ink)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(muted)
        ax.tick_params(colors=muted, labelcolor=ink)
    # legend: methods (color) and backends (line style), timeout marker
    handles = [plt.Line2D([], [], color=c, linewidth=2, label=m)
               for m, c in colors.items()]
    handles += [plt.Line2D([], [], color=muted, linewidth=2, label=b, **s)
                for b, s in styles.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize="small")
    fig.tight_layout(rect=(0, 0.1/len(orders), 1, 1))
    # no space between panels
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(outfile, dpi=150)
    return

if __name__ == "__main__":
    here = Path(__file__).parent
    plot_integration(infile=here / "integration.json",
                     outfile=here / "integration.png")
