# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Dict,Tuple
from pstats import Stats

import numpy as np

def display_performance(filename: str,
                        n: int = 5) -> None:
    """
    Load a cProfile dump and print the top n functions sorted by cumulative time.

    Parameters
    ----------
    filename : str
        path to the cProfile stats file written by cProfile.dump_stats().
    n : int
        number of top entries to print.

    Returns
    -------
    None
    """
    #
    stats = Stats(filename).strip_dirs().sort_stats("cumtime").print_stats(int(n))
    return 