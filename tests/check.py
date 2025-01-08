import numpy as np

from topopt import main


def test_Helmholtz_sensitivity_filtering():
    """
    Tests Fig. 4 top row and Section 4.2.

    Returns
    -------
    None.

    """
    # Default input parameters
    volfrac = 0.5
    penal = 3.0
    ft = 0
    # check for different geometries
    for nelx,nely,csol in zip([60,150,300],[20,50,100],[218.79,217.88,219.44]):
        _, c = main(nelx, nely, volfrac, penal, 0.04 * nelx, ft, pde=True,
                    verbose=False)
        assert np.isclose(c, csol,atol=1e-2)
    
    print("Helmholtz sensitivity filter test passed.")
    return

def test_Helmholtz_density_filtering():
    """
    Tests Fig. 4 bottom row and Section 4.2.

    Returns
    -------
    None.

    """
    # Default input parameters
    volfrac = 0.5
    penal = 3.0
    ft = 1
    # check for different geometries
    for nelx,nely,csol in zip([60,150,300],[20,50,100],[237.60,235.36,236.62]):
        _, c = main(nelx, nely, volfrac, penal, 0.04 * nelx, ft, pde=True,
                    verbose=False)
        assert np.isclose(c, csol,atol=1e-2)
    print("Helmholtz density filter test passed.")
    return

def test_sensitivity_filtering():
    """
    Tests Fig. 3 top row and Section 3.4.

    Returns
    -------
    None.

    """
    # Default input parameters
    volfrac = 0.5
    penal = 3.0
    ft = 0
    # check for different geometries
    for nelx,nely,csol in zip([60,150,300],[20,50,100],[216.81,219.52,222.29]):
        _, c = main(nelx, nely, volfrac, penal, 0.04 * nelx, ft, pde=False,
                    verbose=False)
        assert np.isclose(c, csol,atol=1e-2)
    
    print("Sensitivity filter test passed.")
    return

def test_density_filtering():
    """
    Tests Fig. 3 bottom row and Section 3.4.

    Returns
    -------
    None.

    """
    # Default input parameters
    volfrac = 0.5
    penal = 3.0
    ft = 1
    # check for different geometries
    for nelx,nely,csol in zip([60,150,300],[20,50,100],[233.71,235.73,238.31]):
        _, c = main(nelx, nely, volfrac, penal, 0.04 * nelx, ft, pde=False,
                    verbose=False)
        assert np.isclose(c, csol,atol=1e-2)
    print("Density filter test passed.")
    return

# The real main driver
if __name__ == "__main__":
    
    test_sensitivity_filtering()
    test_density_filtering()