# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List

from topoptlab.filter.sensitivity_filter import SensitivityFilter
from topoptlab.filter.density_filter import DensityFilter
from topoptlab.filter.haeviside_projectors import HaevisideProjectorGuest2004,\
                                                  HaevisideProjectorSigmund2007,\
                                                  EtaProjectorXu2010

def fetch_filters(ft : int,
                  filter_args : List) -> List:
    """
    Collect list of filters based on the integer ft and a list containing 
    dictionaries with the information necessary to initialize the filters.
    Currently these integer codes correspond to the following filters:
        
        0: sensitivity filter
        1: density filter
        2: density filter + Guest Haeviside projection
        3: density filter + Sigmund Haeviside projection
        4: density filter + eta projection with fixed eta (volume conserving 
                            eta depends on filter_args)
        -1: not filter

    Parameters
    ----------
    ft : int
        integer code for filters to collect.
    filter_args : list
        list of dictionaries containing information to initialize filters

    Returns
    -------
    filters : list
        list of collected, initialized TOfilters.
    """
    filters = []
    if ft == 0:
        filters.append(SensitivityFilter(**filter_args[0]))
    elif ft == 1:
        filters.append(DensityFilter(**filter_args[0]))
    elif ft == 2:
        filters.append(DensityFilter(**filter_args[0]))
        filters.append(HaevisideProjectorGuest2004(**filter_args[1]))
    elif ft == 3:
        filters.append(DensityFilter(**filter_args[0]))
        filters.append(HaevisideProjectorSigmund2007(**filter_args[1]))
    elif ft == 4:
        filters.append(DensityFilter(**filter_args[0]))
        filters.append(EtaProjectorXu2010(**filter_args[1]))
    return filters