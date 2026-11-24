# SPDX-License-Identifier: GPL-3.0-or-later
from json import dump, load
from pathlib import Path
from signal import signal, alarm, SIGALRM
from time import perf_counter

from symfem.functions import MatrixFunction
from symfem.references import Reference
from symfem.symbols import x, t

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.operators import aniso_laplacian
from topoptlab.symbolic.lin_elasticity import stiffness_matrix
from topoptlab.symbolic.mass_matrix import mass
from topoptlab.symbolic.buckling import buckling_matrix
from topoptlab.symbolic.matrix_utils import integrate, integrate_separated
from topoptlab.symbolic.utils import is_equal, separate_terms

# element integrands via return_integrand
elements = {"poisson": lambda ndim, order: aniso_laplacian(ndim=ndim,
                                                           order=order,
                                                           return_integrand=True),
            "elasticity": lambda ndim, order: stiffness_matrix(ndim=ndim, c=None,
                                                               order=order,
                                                               return_integrand=True),
            "mass": lambda ndim, order: mass(scalarfield=False, ndim=ndim,
                                             order=order,
                                             return_integrand=True), #"buckling": lambda ndim, order: buckling_matrix(ndim=ndim,order=order,return_integrand=True)
            }

# wall clock limit in seconds for each single integration
TIMEOUT = 300

class IntegrationTimeout(Exception):
    pass

def _raise_timeout(signum, frame):
    raise IntegrationTimeout

def benchmark(integrand: MatrixFunction,
              ref: Reference,
              check: bool = True,
              backend: str = "sympy",
              timeout: int = TIMEOUT,
              methods: None | list = None) -> dict:
    """
    Time integrate and integrate_separated (each serial and parallel) on the
    same symmetric integrand. Simplification is not included.

    Parameters
    ----------
    integrand : symfem.functions.MatrixFunction
        square, symmetric integrand.
    ref : symfem.references.Reference
        reference cell.
    check : bool
        if True, check that both methods give identical results.
    backend : str
        integration backend, either "sympy" or "symint".
    timeout : int
        wall clock limit in seconds for each integration. Integrations that
        take longer are aborted and their time is None.
    methods : None or list
        names of the methods to run, out of "integrate serial",
        "integrate parallel", "separated serial" and "separated parallel".
        If None, all are run. Methods not run are missing in the results.

    Returns
    -------
    results : dict
        wall clock times in seconds, number of total and distinct terms and
        result of the check of integrate vs. separated, serial if both were
        run, otherwise parallel (missing if no pair finished).

    """
    n = integrand.shape[0]
    results = {}
    for name, func, parallel in [("integrate serial", integrate, False),
                                 ("integrate parallel", integrate, True),
                                 ("separated serial", integrate_separated, False),
                                 ("separated parallel", integrate_separated, True)]:
        if methods is not None and name not in methods:
            continue
        signal(SIGALRM, _raise_timeout)
        alarm(timeout)
        t0 = perf_counter()
        try:
            results[name] = func(M=integrand, domain=ref, variables=x,
                                 dummy_vars=t, parallel=parallel,
                                 symmetry=True, backend=backend)
            results[name+" time"] = perf_counter() - t0
        except IntegrationTimeout:
            results[name] = None
            results[name+" time"] = None
        finally:
            alarm(0)
    # number of terms in the upper triangle
    separated = [separate_terms(integrand[i][j].as_sympy(), list(x)) \
                 for i in range(n) for j in range(i, n)]
    results["terms total"] = sum(len(s) for s in separated)
    results["terms distinct"] = len(set(m for s in separated for m in s))
    # compare integrate and separated, serial if available, else parallel
    A, B = None, None
    for mode in ["serial", "parallel"]:
        A = results.get("integrate "+mode)
        B = results.get("separated "+mode)
        if A is not None and B is not None:
            break
    if check and A is not None and B is not None:
        results["identical"] = all(is_equal(A[i][j].as_sympy(),
                                            B[i][j].as_sympy()) \
                                   for i in range(n) for j in range(i, n))
    return results

def _fmt(res: dict, name: str) -> str:
    if name+" time" not in res:
        return "-"
    time = res[name+" time"]
    return f"> {TIMEOUT} s" if time is None else f"{time:.2f} s"

if __name__ == "__main__":
    #
    names = ["integrate serial", "integrate parallel",
             "separated serial", "separated parallel"]
    backends = ["sympy", "symint"]
    # methods per element order. Serial runs of order 2 take too long.
    order_methods = {1: names,
                     2: ["integrate parallel", "separated parallel"]}
    # timing results are written to this file after each row. Results already
    # in the file are kept and not recomputed (records without order are
    # order 1 from older runs).
    outfile = Path(__file__).with_name("integration.json")
    records = []
    if outfile.exists():
        with open(outfile) as f:
            records = load(f)["results"]
        for r in records:
            r.setdefault("order", 1)
    done = {(r["element"], r["ndim"], r["order"], r["backend"])
            for r in records}
    #
    print("| element | ndim | order | backend | entries | "
          "terms total -> distinct | " + " | ".join(names) + " | identical |")
    print("|" + "---|"*(len(names)+7))
    for order in [1, 2]:
        for element in elements:
            for ndim in [1, 2, 3]:
                todo = [b for b in backends
                        if (element, ndim, order, b) not in done]
                if not todo:
                    continue
                integrand = elements[element](ndim, order)
                ref = base_cell(ndim, order=order)[2]
                n = integrand.shape[0]
                for backend in todo:
                    res = benchmark(integrand=integrand, ref=ref,
                                    check=True, backend=backend,
                                    methods=order_methods[order])
                    print(f"| {element} | {ndim} | {order} | {backend} | "
                          f"{n*(n+1)//2} | "
                          f"{res['terms total']} -> {res['terms distinct']} | "
                          + " | ".join(_fmt(res, name) for name in names)
                          + f" | {res.get('identical', '-')} |", flush=True)
                    # times of None mean timeout, methods not run are
                    # missing
                    records.append({"element": element,
                                    "ndim": ndim,
                                    "order": order,
                                    "backend": backend,
                                    "entries": n*(n+1)//2,
                                    "terms total": res["terms total"],
                                    "terms distinct": res["terms distinct"],
                                    "times": {name: res[name+" time"]
                                              for name in names
                                              if name+" time" in res},
                                    "identical": res.get("identical")})
                    with open(outfile, "w") as f:
                        dump({"timeout": TIMEOUT, "results": records}, f,
                             indent=2)
