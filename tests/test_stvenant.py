import numpy as np
from numpy.testing import assert_allclose
import pytest
from topoptlab.material_models.stvenant import stress_2pk, stvenant_matmodel
from topoptlab.material_models.stvenant import stvenant_thermo_matmodel,stvenant_thermo_dmatmodel

@pytest.mark.parametrize("F, S_expect, C0",
    [pytest.param(
            np.eye(2)[None, None, :, :],
            np.array([0.0, 0.0, 0.0]),
            np.diag([10.0, 20.0, 5.0]),
            id="zero_deformation",
        ),
    pytest.param(
            np.array([[[[1.0, 0.2],
                        [0.0, 1.0]]]]),
            np.array([0.0, 0.4, 1.0]),
            np.diag([10.0, 20.0, 5.0]),
            id="simple_shear",
        )])

def test_stvenant_matmodel(F, S_expect, C0):
    """
    Test mechanical STVK model for prescribed deformation gradients.

    Each case provides:
        F   : deformation gradient
        S_expect : analytically expected 2PK stress in Voigt notation
        C0        : material stiffness matrix
    """

    C = C0[None, :, :]  # shape: (nel, nstress, nstress)

    S, Ctan = stvenant_matmodel(F=F, c=C)

    assert_allclose(S[0, 0, :], S_expect)

    # STVK material tangent is constant: dS / dE = C
    assert_allclose(Ctan[0, 0, :, :], C0)

@pytest.mark.parametrize("E, S_expect, C0",
    [pytest.param(
            np.array([[[0.0, 0.0, 0.0]]]),
            np.array([0.0, 0.0, 0.0]),
            np.diag([10.0, 20.0, 5.0]),
            id="zero_strain",
        ),
     pytest.param(
            np.array([[[0.0, 0.02, 0.2]]]),
            np.array([0.0, 0.4, 1.0]),
            np.diag([10.0, 20.0, 5.0]),
            id="given_E_simple_shear",
        )])

def test_stress_2pk(E, S_expect, C0):
    """
    Test direct 2PK stress calculation with prescribed Green-Lagrange strain.
    
    This checks:
        S = C @ E
    """
    F_dummy = np.eye(2)[None, None, :, :]
    C = C0[None, :, :]

    S = stress_2pk(F=F_dummy, E=E, c=C)

    assert_allclose(S[0, 0, :], S_expect)

def test_stvenant_thermo_matmodel():
    """
    Test thermo-STVK free expansion.

    F is chosen so that:
        E = DeltaT * alpha

    Therefore:
        E_mech = E - DeltaT * alpha = 0
        S = 0
    """

    a = 0.01
    DeltaT = np.array([[2.0]])

    C = np.diag([10.0, 20.0, 5.0])[None, :, :]
    alpha = np.array([[[a, 0.0],
                       [0.0, a]]])

    # Choose stretch such that:
    # 1/2 * (F.T @ F - I) = DeltaT * alpha
    stretch = np.sqrt(1.0 + 2.0 * a * DeltaT[0, 0])

    F = np.array([[[[stretch, 0.0],
                    [0.0, stretch]]]])

    S, Ctan = stvenant_thermo_matmodel(
        F=F,
        c=C,
        alpha=alpha,
        DeltaT=DeltaT,
    )

    assert_allclose(S, 0.0, atol=1e-13, err_msg="Failed thermo-STVK case: free expansion should give zero stress")
    assert_allclose(Ctan, C[:, None, :, :], err_msg="Failed tangent check for thermo-STVK free expansion")

def test_stvenant_thermo_dmatmodel_fd():
    """
    Test thermo-STVK directional derivative by central finite difference.

    The analytical derivative dS is compared with:
        dS_fd = [S(C + eps*dC, alpha + eps*dalpha)
               - S(C - eps*dC, alpha - eps*dalpha)] / (2 eps)
    """

    eps = 1e-6
    DeltaT = np.array([[3.0]])

    F = np.array([[[[1.1, 0.2],
                    [0.0, 0.95]]]])

    C = np.diag([10.0, 20.0, 5.0])[None, :, :]

    dC = np.array([[
        [0.1, 0.2, 0.0],
        [0.2, -0.1, 0.0],
        [0.0, 0.0, 0.05],
    ]])

    alpha = np.array([[[0.01, 0.002],
                       [0.002, 0.02]]])

    dalpha = np.array([[[0.001, 0.0002],
                        [0.0002, -0.0003]]])

    dS, dCout = stvenant_thermo_dmatmodel(
        F=F,
        c=C,
        dc=dC,
        alpha=alpha,
        dalpha=dalpha,
        DeltaT=DeltaT,
    )

    S_plus, _ = stvenant_thermo_matmodel(
        F=F,
        c=C + eps * dC,
        alpha=alpha + eps * dalpha,
        DeltaT=DeltaT,
    )

    S_minus, _ = stvenant_thermo_matmodel(
        F=F,
        c=C - eps * dC,
        alpha=alpha - eps * dalpha,
        DeltaT=DeltaT,
    )

    dS_fd = (S_plus - S_minus) / (2.0 * eps)

    assert_allclose(dS, dS_fd, rtol=1e-7, atol=1e-9, err_msg="Failed thermo-STVK derivative finite-difference check")
    assert_allclose(dCout, dC[:, None, :, :], err_msg="Failed returned dCout check in thermo-STVK derivative test")