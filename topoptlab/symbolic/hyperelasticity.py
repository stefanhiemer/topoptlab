# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

from sympy import Rational
from sympy.core.symbol import Symbol
from symfem.functions import ScalarFunction,MatrixFunction

from topoptlab.symbolic.matrix_utils import generate_constMatrix, is_voigt,\
                                            to_voigt
from topoptlab.symbolic.stress_conversions import pk2_to_pk1

def _strain_symbols(E : None | MatrixFunction,
                    ndim: int | None) -> tuple[list, list]:
    """
    Return the independent symbols of the Lagrangian strain E in Voigt order
    and the weights that map derivatives with respect to these symbols to
    derivatives with respect to the engineering Voigt strain

        e = [E_11, E_22, (E_33,) (2*E_23, 2*E_13,) 2*E_12].

    Differentiating with respect to e makes the gradient of the energy density
    the 2. PK stress S in Voigt notation and its Hessian the constitutive
    tensor in Voigt notation, consistent with the engineering convention used
    for the strain variation matrices.

    If E is given in Voigt notation, its entries are interpreted as the
    engineering strains e (weights 1). If E is given as symmetric matrix, the
    shear symbols E_ij appear twice in the energy density, so

        d e / d e_ij = 1/2 * d e / d E_ij

    (weight 1/2 for shear components).

    Parameters
    ----------
    E : None or MatrixFunction
        Lagrangian strain either as symmetric matrix (ndim,ndim) or in
        engineering Voigt notation (ndim*(ndim+1)/2,1). Entries must be
        symbols. If None, engineering Voigt symbols E1, E2, ... are generated.
    ndim : None or int
        number of spatial dimensions. Only needed if E is None.

    Returns
    -------
    symbs : list
        sympy symbols of the strain in Voigt order.
    weights : list
        weights for derivatives with respect to symbs.
    """
    #
    if E is None and ndim is None:
        raise ValueError("E and ndim are none. "
                         "Either ndim must be an integer or E must be given.")
    #
    if E is None:
        E = generate_constMatrix(ncol=1,
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="E")
    #
    # infer ndim from E (overrides the argument)
    if E.shape[0] == E.shape[1]:
        ndim = E.shape[0]
    elif E.shape[1] == 1 and E.shape[0] in (1, 3, 6):
        ndim = {1: 1, 3: 2, 6: 3}[E.shape[0]]
    else:
        raise ValueError("E must be of shape (ndim,ndim) or "
                         "(ndim*(ndim+1)/2,1): ", E.shape)
    #
    nvoigt = int((ndim**2 + ndim) /2)
    if is_voigt(M=E, ndim=ndim):
        weights = [1 for i in range(nvoigt)]
    else:
        for i in range(ndim):
            for j in range(i+1, ndim):
                if E[i,j].as_sympy() != E[j,i].as_sympy():
                    raise ValueError("E in matrix form must be symmetric with "
                                     "identical symbols for E_ij and E_ji.")
        E = to_voigt(E, eng_conv=False)
        weights = [1 for i in range(ndim)] + \
                  [Rational(1,2) for i in range(nvoigt-ndim)]
    #
    symbs = [E[i,0].as_sympy() for i in range(nvoigt)]
    if not all(isinstance(s, Symbol) for s in symbs):
        raise ValueError("Entries of E must be sympy symbols. Expressions "
                         "(e.g. E calculated from F) cannot be differentiated "
                         "with respect to. Express the energy density in "
                         "symbolic strains and substitute afterwards.")
    return symbs, weights

def calculate_1pk(eng_density: ScalarFunction,
                  F : None | MatrixFunction = None,
                  E : None | MatrixFunction = None,
                  ndim: int | None = 3,
                  **kwargs: Any) -> MatrixFunction:
    """
    Returns 1. Piola-Kirchhoff stress (1PK) P of a given elastic energy density
    e by first calculating the 2. Piola-Kirchhoff stress (2PK) S

        S_i = d e(E) / d E_i

    and then converting it via

        P = F@S

    where F is the deformation gradient. The energy density must be a
    function of the Lagrangian strain E (see calculate_2pk). P thus contains
    the strain symbols of E and the symbols of F; to express P in terms of F
    only, E must be substituted by E(F).

    Parameters
    ----------
    eng_density : symfem.functions.ScalarFunction
        strain energy density.
    F : None or symfem.functions.MatrixFunction
        symbolic deformation gradient of shape (ndim,ndim). If None, generated
        from symbols F11, F12, ...
    E : None or MatrixFunction
        Lagrangian strain either as symmetric matrix or in engineering Voigt
        notation. See calculate_2pk.
    ndim : None or int
        number of spatial dimensions. Only needed if F or E is None.

    Returns
    -------
    P : symfem.functions.MatrixFunction
        symbolic 1. Piola-Kirchhoff stress (1PK) of shape (ndim,ndim).

    """
    #
    if F is None:
        if ndim is None:
            raise ValueError("F and ndim are none. "
                             "Either ndim must be an integer or F must be given.")
        F = generate_constMatrix(ncol=ndim,
                                 nrow=ndim,
                                 name="F")
    #
    S = calculate_2pk(eng_density=eng_density,
                      E=E,
                      ndim=F.shape[0])
    return pk2_to_pk1(S=S, F=F)

def calculate_2pk(eng_density: ScalarFunction,
                  E : None | MatrixFunction = None,
                  ndim: int | None = 3,
                  **kwargs: Any) -> MatrixFunction:
    """
    Returns 2. Piola-Kirchhoff stress (2PK) S of a given elastic energy density
    e by

        S_i = d e(E) / d e_i

    where e is the Lagrangian strain E in Voigt notation with engineering
    convention, i. e. shear components are scaled by two

        e = [E_11, E_22, (E_33,) (2*E_23, 2*E_13,) 2*E_12].

    We assume that the energy density is a function of E only (no other
    strain/deformation measures). If E is given in matrix form, the factor
    1/2 for the shear components is taken care of internally.

    Parameters
    ----------
    eng_density : symfem.functions.ScalarFunction
        strain energy density.
    E : None or MatrixFunction
        Lagrangian strain either as symmetric matrix of symbols (ndim,ndim) or
        in engineering Voigt notation (ndim*(ndim+1)/2,1). If None, Voigt
        symbols E1, E2, ... are generated.
    ndim : None or int
        number of spatial dimensions. Only needed if E is None.

    Returns
    -------
    S : symfem.functions.MatrixFunction
        symbolic 2. Piola-Kirchhoff stress (2PK) in Voigt notation of shape
        (ndim*(ndim+1)/2,1).

    """
    #
    symbs, weights = _strain_symbols(E=E, ndim=ndim)
    #
    return MatrixFunction([[eng_density.diff(variable=s).as_sympy()*w] \
                           for s,w in zip(symbs, weights)])

def calculate_consttensor_2pk(eng_density: ScalarFunction,
                              E : None | MatrixFunction = None,
                              ndim: int | None = 3,
                              **kwargs: Any) -> MatrixFunction:
    """
    Returns the constitutive tensor of the 2. Piola-Kirchhoff stress (2PK) in
    Voigt notation

        c_ij = d S_i / d e_j = d^2 e(E) / d e_i d e_j

    where e is the Lagrangian strain in Voigt notation with engineering
    convention (see calculate_2pk).

    Parameters
    ----------
    eng_density : symfem.functions.ScalarFunction
        strain energy density.
    E : None or MatrixFunction
        Lagrangian strain either as symmetric matrix of symbols (ndim,ndim) or
        in engineering Voigt notation (ndim*(ndim+1)/2,1). If None, Voigt
        symbols E1, E2, ... are generated.
    ndim : None or int
        number of spatial dimensions. Only needed if E is None.

    Returns
    -------
    c : symfem.functions.MatrixFunction
        symbolic constitutive tensor of shape
        (ndim*(ndim+1)/2,ndim*(ndim+1)/2).

    """
    #
    symbs, weights = _strain_symbols(E=E, ndim=ndim)
    #
    grad = [eng_density.diff(variable=s) for s in symbs]
    return MatrixFunction([[ (gi.diff(variable=sj).as_sympy()*wi*wj) \
                             for sj,wj in zip(symbs, weights)] \
                           for gi,wi in zip(grad, weights)])
