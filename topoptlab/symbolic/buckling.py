# SPDX-License-Identifier: GPL-3.0-or-later

from symfem.symbols import x,t
from symfem.functions import MatrixFunction

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.parametric_map import jacobian
from topoptlab.symbolic.matrix_utils import eye, simplify_matrix,\
                                            generate_constMatrix,\
                                            from_voigt, kron,\
                                            integrate
from topoptlab.symbolic.strain_measures import dispgrad_matrix,\
                                               small_strain_matrix

def buckling_matrix(ndim : int,
                    sigma : None | MatrixFunction = None,
                    u : None | MatrixFunction = None,
                    c : None | MatrixFunction = None,
                    element_type : str ="Lagrange",
                    order : int = 1,
                    return_integrand : bool = False) -> MatrixFunction:
    """
    Symbolically compute the stress stiffness (geometric stiffness) matrix
    for linearized buckling

        G = int B_h^T (I kron sigma) B_h dV

    where B_h is the matrix to compute the displacement gradient and sigma
    the (Cauchy) stress of the linear prebuckling state. If sigma is None,
    the stress is calculated from the displacements via sigma = c@B@u.
    Otherwise the given stress is assumed constant within the element (e. g.
    evaluated at the element center), so G is linear in the stress
    components.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    sigma : None or symfem.functions.MatrixFunction
        constant stress in Voigt notation. If None, stress is calculated from
        the displacements.
    u : None or symfem.functions.MatrixFunction
        nodal displacements of the prebuckling state. Only used if sigma is
        None. If None, generic displacements are assumed.
    c : None or symfem.functions.MatrixFunction
        stiffness tensor in Voigt notation. Only used if sigma is None. If
        None, generic stiffness tensor is assumed.
    element_type : str
        type of element.
    order : int
        order of element.
    return_integrand : bool
        if True, return the integrand (including the determinant of the
        jacobian) without integrating it over the element.

    Returns
    -------
    Ge : symfem.functions.MatrixFunction
        symbolic stress stiffness matrix.

    """
    #
    vertices, nd_inds, ref, basis = base_cell(ndim)
    # stress in Voigt notation
    if sigma is None:
        #
        if u is None:
            u = generate_constMatrix(ncol=1,
                                     nrow=len(nd_inds)*ndim,
                                     name="u")
        # anisotropic stiffness tensor or equivalent in Voigt notation
        if c is None:
            c = generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                     nrow=int((ndim**2 + ndim) /2),
                                     name="c")
        # small strain matrix
        b = small_strain_matrix(ndim=ndim,
                                nd_inds=nd_inds,
                                basis=basis,
                                isoparam_kws={"element_type": element_type,
                                              "order": order})
        sigma = c@b@u
    # calculate matrix for displacement gradient
    b_h = dispgrad_matrix(ndim=ndim,
                          nd_inds=nd_inds,
                          basis=basis,
                          isoparam_kws={"element_type": element_type,
                                        "order": order})
    sigma = kron(eye(ndim),from_voigt(sigma, eng_conv=False))
    # create full integral and multiply with determinant
    Jdet = jacobian(ndim=ndim,
                    element_type=element_type,
                    order=order,
                    return_J=False,
                    return_inv=False,
                    return_det=True)
    integrand = b_h.transpose()@sigma@b_h * Jdet
    if return_integrand:
        return integrand
    return simplify_matrix(M=integrate(M=integrand,
                                       domain=ref,
                                       variables=x,
                                       dummy_vars=t,
                                       parallel=None,
                                       symmetry=True),
                           symmetry=True)