# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path
from subprocess import run
from numpy import loadtxt
from numpy.testing import assert_almost_equal
import pytest
import sys


@pytest.mark.parametrize(
    "example_file, params",
    [
        ("Lbracket.py", "100 100 0.3 5.0 3 5 3 0 0 0"),
    ],
)
def test_stress_lbracket(tmp_path, example_file, params):
    """
    Small test for the 2D stress Lbracket example.

    The example is run as a script, writes u_bw and rhs_adj to CSV files in
    tmp_path, and the results are compared against reference CSVs.
    """
    test_path = Path(__file__).resolve().parent
    file_path = (
        test_path.parent
        / "examples"
        / "topology_optimization"
        / "stress_constraint"
        / example_file
    )

    cmd = [sys.executable, str(file_path)] + params.split(" ")
    run(cmd, cwd=tmp_path, shell=False, check=True)

    u_bw = loadtxt(tmp_path / "stress_lbracket_u_bw.csv", delimiter=",")[:, None]
    rhs_adj = loadtxt(tmp_path / "stress_lbracket_rhs_adj.csv", delimiter=",")[:, None]

    u_bw_ref = loadtxt(
        test_path / "test_files" / "stress_lbracket_u_bw.csv",
        delimiter=",",
    )[:, None]
    rhs_adj_ref = loadtxt(
        test_path / "test_files" / "stress_lbracket_rhs_adj.csv",
        delimiter=",",
    )[:, None]

    assert_almost_equal(u_bw, u_bw_ref)
    assert_almost_equal(rhs_adj, rhs_adj_ref)