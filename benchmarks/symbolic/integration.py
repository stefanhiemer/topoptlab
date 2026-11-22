# SPDX-License-Identifier: GPL-3.0-or-later
from argparse import ArgumentParser
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
elements = {"poisson": lambda ndim: aniso_laplacian(ndim=ndim,
                                                    return_integrand=True),
            "elasticity": lambda ndim: stiffness_matrix(ndim=ndim, c=None,
                                                        return_integrand=True),
            "mass": lambda ndim: mass(scalarfield=False, ndim=ndim,
                                      return_integrand=True),
            "buckling": lambda ndim: buckling_matrix(ndim=ndim,
                                                     return_integrand=True)}

def benchmark(integrand: MatrixFunction,
              ref: Reference,
              check: bool = True) -> dict:
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

    Returns
    -------
    results : dict
        wall clock times in seconds, number of total and distinct terms and
        result of the check.

    """
    n = integrand.shape[0]
    results = {}
    for name, func, parallel in [("integrate serial", integrate, False),
                                 ("integrate parallel", integrate, True),
                                 ("separated serial", integrate_separated, False),
                                 ("separated parallel", integrate_separated, True)]:
        t0 = perf_counter()
        results[name] = func(M=integrand, domain=ref, variables=x,
                             dummy_vars=t, parallel=parallel, symmetry=True)
        results[name+" time"] = perf_counter() - t0
    # number of terms in the upper triangle
    separated = [separate_terms(integrand[i][j].as_sympy(), list(x)) \
                 for i in range(n) for j in range(i, n)]
    results["terms total"] = sum(len(s) for s in separated)
    results["terms distinct"] = len(set(m for s in separated for m in s))
    if check:
        A, B = results["integrate serial"], results["separated serial"]
        results["identical"] = all(is_equal(A[i][j].as_sympy(),
                                            B[i][j].as_sympy()) \
                                   for i in range(n) for j in range(i, n))
    return results

if __name__ == "__main__":
    #
    parser = ArgumentParser(description="Compare element-wise and separated "
                                        "symbolic integration of element "
                                        "matrices.")
    parser.add_argument("--elements", nargs="+",
                        default=["poisson", "elasticity"],
                        choices=list(elements))
    parser.add_argument("--dims", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--no-check", action="store_true",
                        help="skip check that both methods agree.")
    args = parser.parse_args()
    #
    names = ["integrate serial", "integrate parallel",
             "separated serial", "separated parallel"]
    #
    print("| element | ndim | entries | terms total -> distinct | "
          + " | ".join(names) + " | identical |")
    print("|" + "---|"*(len(names)+5))
    for element in args.elements:
        for ndim in args.dims:
            integrand = elements[element](ndim)
            ref = base_cell(ndim)[2]
            n = integrand.shape[0]
            res = benchmark(integrand=integrand, ref=ref,
                            check=not args.no_check)
            print(f"| {element} | {ndim} | {n*(n+1)//2} | "
                  f"{res['terms total']} -> {res['terms distinct']} | "
                  + " | ".join(f"{res[name+' time']:.2f} s" for name in names)
                  + f" | {res.get('identical', '-')} |", flush=True)
