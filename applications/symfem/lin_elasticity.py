from symfem.symbols import x

from topoptlab.symfem_utils import base_cell, generate_constMatrix
from topoptlab.symfem_utils import small_strain_matrix,stifftens_isotropic
from topoptlab.symfem_utils import convert_to_code, jacobian, simplify_matrix

def linelast(iso, plane_stress,
             ndim,
             element_type="Lagrange",
             order=1):
    """
    Symbolically compute the stiffness matrix for linear elasticity.

    Parameters
    ----------
    iso : bool
        if True, isotropic elasticity is assumed.
    plane_stress : bool
        if True, plane_stress is assumed. Only relevant for 2D at the moment.
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    stiffness_matrix : list
        symbolic stiffness matrix as list of lists .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # anisotropic stiffness tensor or equivalent in Voigt notation
    if iso:
        c = stifftens_isotropic(ndim,plane_stress=plane_stress)
    else:
        c = generate_constMatrix(int((ndim**2 + ndim) /2),
                                 int((ndim**2 + ndim) /2),
                                 "c")
    #
    b = small_strain_matrix(ndim=ndim,
                            nd_inds=nd_inds,
                            basis=basis,
                            isoparam_kws={"element_type": element_type,
                                          "order": order})
    # create full integral and multiply with determinant
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = b.transpose()@c@b * Jdet
    return simplify_matrix( integrand.integral(ref,x) )

def strainforces(iso, plane_stress, 
                 ndim,
                 element_type="Lagrange",
                 order=1):
    """
    Symbolically compute the nodal forces due to a uniform strain for an 
    isotropic linear elastic material.

    Parameters
    ----------
    iso : bool
        if True, isotropic elasticity is assumed.
    plane_stress : bool
        if True, plane_stress is assumed. Only relevant for 2D at the moment.
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    nodal_forces : list
        symbolic stiffness matrix as list of lists .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # anisotropic stiffness tensor or equivalent in Voigt notation
    if iso:
        c = stifftens_isotropic(ndim,plane_stress=plane_stress)
    else:
        c = generate_constMatrix(int((ndim**2 + ndim) /2),
                                 int((ndim**2 + ndim) /2),
                                 "c")
    # strains
    eps = generate_constMatrix(ncol=1,nrow=int((ndim**2 + ndim) /2),
                               name="eps")
    #
    b = small_strain_matrix(ndim=ndim,
                            nd_inds=nd_inds,
                            basis=basis,
                            isoparam_kws={"element_type": element_type,
                                          "order": order})
    # create full integral and multiply with determinant
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = b.transpose()@c@eps * Jdet
    return simplify_matrix( integrand.integral(ref,x) )

if __name__ == "__main__":
    
    # stiffness matrix isotropic linear elasticity
    #for dim in range(1,4):
    #    print(convert_to_code(linelast(iso=True, plane_stress=True, ndim=dim),
    #                          matrices=["c"],vectors=["l","g"]),"\n")
    # stiffness matrix anisotropic linear elasticity
    #for dim in range(3,4):
    #    print(str(dim)+"D")
    #    print(convert_to_code(linelast(ndim = dim, iso=False, plane_stress=True),
    #                          matrices=["c"],vectors=["l","g"]),"\n")
    #
    for dim in range(1,3):
        print(str(dim)+"D")
        print(convert_to_code(strainforces(iso=True, plane_stress=False,ndim = dim),
                              matrices=["c"],vectors=["l","g", "eps"]),"\n")
    import sys 
    sys.exit()
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(strainforces(iso=False, plane_stress=True,ndim = dim),
                              matrices=["c"],vectors=["l","g", "eps"]),"\n")