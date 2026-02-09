import pytest 


from topoptlab.voigt import voigt_index, voigt_pair  


@pytest.mark.parametrize("ndim", [2,3])
def test_voigt_inverse_consistency(ndim):
    for i in range(ndim):
        for j in range(ndim):
            alpha = voigt_index(i, j, ndim)
            ii, jj = voigt_pair(alpha, ndim)

            ii = int(ii)
            jj = int(jj)

            assert set((i, j)) == set((ii, jj))
    return
