# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List

from topoptlab.filter.sensitivity_filter import SensitivityFilter
from topoptlab.filter.density_filter import DensityFilter
from topoptlab.filter.haeviside_projectors import HaevisideProjectorGuest2004,\
                                                  HaevisideProjectorSigmund2007,\
                                                  EtaProjectorXu2010
from topoptlab.filter.amfilter_langelaar import LangelaarFilter

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
        4: density filter + eta projection (volume conserving 
                            eta depends on filter_args)
        5: density filter + langelaar filter + eta projection
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
    
    if ft == 0:
        filters = [SensitivityFilter]
    elif ft == 1:
        filters = [DensityFilter]
    elif ft == 2:
        filters = [DensityFilter,
                   HaevisideProjectorGuest2004]
    elif ft == 3:
        filters = [DensityFilter, 
                   HaevisideProjectorSigmund2007]
    elif ft == 4:
        filters = [DensityFilter, 
                   EtaProjectorXu2010]
    elif ft == 5:
        filters = [DensityFilter,
                   LangelaarFilter,
                   EtaProjectorXu2010]
    else:
        raise NotImplementedError("Unknown ft code: ", ft)
    #
    if len(filters) > 1 and len(filter_args) == 1:
        filter_args = len(filters)*filter_args
    #
    filters = [f(**filter_args[i]) for i,f in enumerate(filters)]
    return filters