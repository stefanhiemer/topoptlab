from pathlib import Path
from subprocess import run
from numpy import loadtxt
from numpy.testing import assert_almost_equal
import pytest
import sys


@pytest.mark.parametrize(
    "example_file, params",
    [
        ("cshape.py", "6 6 2 15 1e-6 0"),
    ],
)
def test_third_medium_contact(tmp_path, example_file, params):
    """
    Small test for the 2D third_medium_contact / HuHu benchmark.

    The example is run as a script, writes the displacement vector to a CSV
    file in tmp_path, and the result is compared against a reference CSV.
    """
    test_path = Path(__file__).resolve().parent
    file_path = test_path.parent / "examples" / "fem" / "third_medium_contact" / example_file
    cmd = [sys.executable, str(file_path)] + params.split(" ")
    run(cmd, cwd=tmp_path, shell=False, check=True)
    u = loadtxt(tmp_path / "third_medium_contact_u.csv", delimiter=",")[:, None]
    u_ref = loadtxt(test_path / "test_files" / "third_medium_contact_u.csv", delimiter=",")[:, None]

    assert_almost_equal(u, u_ref)