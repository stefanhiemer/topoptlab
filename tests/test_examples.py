from subprocess import run
import os

import pytest

@pytest.mark.parametrize('example_directory, params',
                         [("compliance_minimization","8 4 0.3 2.4 3.0 0 0 0"),
                          ("compliant_mechanisms","8 4 0.3 2.4 3.0 0 0 0"),
                          ])

def test_topology_optimization2d(tmp_path,example_directory,params):
    """
    Just checks whether they run, not their accuracy.
    """
    #
    test_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(test_path, '..', 
                        'applications/topology_optimization', 
                        example_directory)
    applications = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.py')])
    #
    cmds = [['python', file, ] + params.split(" ") for file in applications \
            if "2d" in file]
    #
    for cmd in cmds:
        run(cmd, shell=False, check=True)
    return 

@pytest.mark.parametrize('example_directory, params',
                         [("compliance_minimization","8 4 2 0.3 2.4 3.0 0 0 0"),
                          ("compliant_mechanisms","8 4 2 0.3 2.4 3.0 0 0 0"),
                          ])

def test_topology_optimization3d(tmp_path,example_directory,params):
    """
    Just checks whether they run, not their accuracy.
    """
    #
    test_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(test_path, '..', 
                        'applications/topology_optimization', 
                        example_directory)
    applications = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.py')])
    #
    cmds = [['python', file, ] + params.split(" ") for file in applications \
            if "3d" in file]
    #
    for cmd in cmds:
        run(cmd, shell=False, check=True)
    return 