from numpy.testing import assert_almost_equal
from numpy import loadtxt
from examples.fem.third_medium_contact.cshape import Cshape
def test_third_medium_contact():
    """
    Small test for the 2D third_medium_contact / HuHu benchmark.
    """
    nelx = 6
    nely = 6
    u = Cshape(
        nelx=nelx,
        nely=nely,
        nelz=None,
        export=False,
        nsteps=2,
        newton_maxit=4,
        rtol=1e-6,
    )
    u_ref = loadtxt("./tests/test_files/third_medium_contact_u.csv", delimiter=",")
    assert_almost_equal(u, u_ref)

