import numpy as np
from numpy.testing import assert_allclose

from topoptlab.material_models.stvenant import stress_2pk, stvenant_matmodel
from topoptlab.material_models.stvenant import stvenant_thermo_matmodel,stvenant_thermo_dmatmodel

def test_stvenant_matmodel():

    # zero stress
    F = np.eye(2)[None, None, :, :]          # shape: (nel, nq, 2, 2)
    C = np.diag([10.0, 20.0, 5.0])[None, :, :]    # shape: (nel, 3, 3)

    S, Ctan = stvenant_matmodel(F=F, c=C)

    assert_allclose(S, 0.0, atol=1e-14)
    assert_allclose(Ctan, C[:, None, :, :])

    # simple shear
    gamma = 0.2
    F = np.array([[[[1.0, gamma],
                    [0.0, 1.0]]]])          # shape: (1, 1, 2, 2)
    S, Ctan = stvenant_matmodel(F=F, c=C)
    # For F = [[1, gamma], [0, 1]]:
    # E = 1/2 * (F.T @ F - I)
    # In engineering Voigt notation:
    # E_voigt = [E_xx, E_yy, 2 E_xy]
    #         = [0, gamma^2 / 2, gamma]
    E_expected = np.array([0.0, 0.5 * gamma**2, gamma])
    S_expected = np.diag([10.0, 20.0, 5.0]) @ E_expected
    assert_allclose(S[0, 0], S_expected)
    assert_allclose(Ctan[0, 0],np.diag([10.0, 20.0, 5.0]))

    # Direct 2PK stress check with explicit E
    E = np.array([[[0.0, 0.5 * gamma**2, gamma]]])
    S = stress_2pk(F=F, E=E, c=C)
    S_expected = np.diag([10.0, 20.0, 5.0]) @ E[0, 0]
    assert_allclose(S[0, 0], S_expected)

def test_stvenant_thermo_matmodel():
    a = 0.01
    DeltaT = np.array([[2.0]])

    C = np.diag([10.0, 20.0, 5.0])[None, :, :]
    alpha = np.array([[[a, 0.0],
                       [0.0, a]]])

    # Need finite-strain free expansion:
    # E = DeltaT * alpha
    # 1/2 * (F.T F - I) = DeltaT * alpha
    stretch = np.sqrt(1.0 + 2.0 * a * DeltaT[0, 0])

    F = np.array([[[[stretch, 0.0],
                    [0.0, stretch]]]])

    S, Ctan = stvenant_thermo_matmodel(
        F=F,
        c=C,
        alpha=alpha,
        DeltaT=DeltaT,
    )

    assert_allclose(S, 0.0, atol=1e-13)
    assert_allclose(Ctan, C[:, None, :, :])

def test_stvenant_thermo_dmatmodel_fd():
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

    assert_allclose(dS, dS_fd, rtol=1e-7, atol=1e-9)
    assert_allclose(dCout, dC[:, None, :, :])