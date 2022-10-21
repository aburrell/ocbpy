#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB & GC
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Hold, manipulate, and load the open-closed field line boundary data.

References
----------
.. [2] Angeline Burrell, Christer van der Meeren, & Karl M. Laundal. (2020).
   aburrell/aacgmv2 (All Versions). Zenodo. doi:10.5281/zenodo.1212694.

.. [3] Shepherd, S. G. (2014), Altitude‐adjusted corrected geomagnetic
   coordinates: Definition and functional approximations, Journal of
   Geophysical Research: Space Physics, 119, 7501–7521,
   doi:10.1002/2014JA020264.

"""

import datetime as dt
import numpy as np
import types
import warnings

import aacgmv2

import ocbpy
import ocbpy.ocb_correction as ocbcor
from ocbpy import ocb_time
from ocbpy.boundaries.files import get_default_file


class OCBoundary(ocbpy._boundary.OCBoundary):
    """Object containing open-closed field-line boundary (OCB) data.

    Parameters
    ----------
    filename : str or NoneType
        File containing the required open-closed boundary data sorted by time.
        If NoneType, no file is loaded.  If 'default',
        `ocbpy.boundaries.files.get_default_file` is called. (default='default')
    instrument : str
        Instrument providing the OCBoundaries.  Requires 'image', 'ampere', or
        'dmsp-ssj' if a file is provided.  If using filename='default', also
        accepts 'amp', 'si12', 'si13', 'wic', and ''.  (default='')
    hemisphere : int
        Integer (+/- 1) denoting northern/southern hemisphere (default=1)
    boundary_lat : float
        Typical OCBoundary latitude in AACGM coordinates.  Hemisphere will
        give this boundary the desired sign.  (default=74.0)
    stime : dt.datetime or NoneType
        First time to load data or beginning of file.  If specifying time, be
        sure to start before the time of the data to allow the best match within
        the allowable time tolerance to be found. (default=None)
    etime : dt.datetime or NoneType
        Last time to load data or ending of file.  If specifying time, be sure
        to end after the last data point you wish to match to, to ensure the
        best match within the allowable time tolerance is made. (default=None)
    rfunc : numpy.ndarray, function, or NoneType
        OCB radius correction function. If None, will use the instrument
        default. Function must have AACGM MLT (in hours) as argument input.
        To allow the boundary shape to change with univeral time, each temporal
        instance may have a different function (array input). If a single
        function is provided, will recast as an array that specifies this
        function for all times. (default=None)
    rfunc_kwargs : numpy.ndarray, dict, or NoneType
        Optional keyword arguements for `rfunc`. If None is specified,
        uses function defaults.  If dict is specified, recasts as an array
        of this dict for all times.  Array must be an array of dicts.
        (default=None)

    Attributes
    ----------
    records : int
        Number of OCB records (default=0)
    rec_ind : int
        Current OCB record index (default=0; initialised=-1)
    dtime : numpy.ndarray or NoneType
        Numpy array of OCB datetimes (default=None)
    phi_cent : numpy.ndarray or NoneType
        Numpy array of floats that give the angle from AACGM midnight
        of the OCB pole in degrees (default=None)
    r_cent : numpy.ndarray or NoneType
        Numpy array of floats that give the AACGM co-latitude of the OCB
        pole in degrees (default=None)
    r : numpy.ndarray or NoneType
        Numpy array of floats that give the radius of the OCBoundary
        in degrees (default=None)
    min_fom : float
        Minimum acceptable figure of merit for data (default=0)
    x, y, j_mag, etc. : numpy.ndarray or NoneType
        Numpy array of floats that hold the remaining values held in `filename`

    Raises
    ------
    ValueError
        Incorrect or incompatible input

    Warnings
    --------
    DeprecationWarning
        Class moved to `ocbpy._boundary` sub-module; use `ocbpy.OCBoundary`

    """

    def __init__(self, filename="default", instrument='', hemisphere=1,
                 boundary_lat=74.0, stime=None, etime=None, rfunc=None,
                 rfunc_kwargs=None):
        warnings.warn("".join(["Class moved to `ocbpy._boundary` sub-module. ",
                               "It will be removed in version 0.3.1+."]),
                      DeprecationWarning, stacklevel=2)

        ocbpy.OCBoundary.__init__(self, filename=filename,
                                  instrument=instrument, hemisphere=hemisphere,
                                  boundary_lat=boundary_lat, stime=stime,
                                  etime=etime, rfunc=rfunc,
                                  rfunc_kwargs=rfunc_kwargs)
        return


def retrieve_all_good_indices(ocb, **kwargs):
    """Retrieve all good indices from the ocb structure.

    Parameters
    ----------
    ocb : ocbpy.OCBoundary
        Class containing the open-close field line boundary data
    **kwargs : dict
        Optional kwargs

    Returns
    -------
    good_ind : list
        List of indices containing good OCBs

    Warnings
    --------
    DeprecationWarning
        Function moved to `ocbpy.cycle_boundary` sub-module

    See Also
    --------
    ocbpy.cycle_boundary.retrieve_all_good_indices

    """

    warnings.warn("".join(["Function moved to `ocbpy.cycle_boundary` ",
                           "sub-module. It will be removed in version",
                           " 0.3.1+."]),
                  DeprecationWarning, stacklevel=2)

    good_ind = ocbpy.cycle_boundary.retrieve_all_good_indices(ocb, **kwargs)

    # Return the good indices
    return good_ind


def match_data_ocb(ocb, dat_dtime, idat=0, max_tol=600, **kwargs):
    """Match data records with OCB records.

    Parameters
    ----------
    ocb : ocbpy.OCBoundary
        Class containing the open-close field line boundary data
    dat_dtime : (list or numpy array of datetime objects)
        Times where data exists
    idat : int
        Current data index (default=0)
    max_tol : int
        Maximum seconds between OCB and data record in sec (default=600)
    **kwargs : dict
        Optional keyword arguments

    Returns
    -------
    idat : int or NoneType
        Data index for match value, None if all of the data have been searched

    Notes
    -----
    Updates OCBoundary.rec_ind for matched value. None if all of the
    boundaries have been searched.

    Warnings
    --------
    DeprecationWarning
        Function moved to `ocbpy.cycle_boundary` sub-module

    See Also
    --------
    ocbpy.cycle_boundary.match_data_ocb

    """

    warnings.warn("".join(["Function moved to `ocbpy.cycle_boundary` ",
                           "sub-module. It will be removed in version",
                           " 0.3.1+."]),
                  DeprecationWarning, stacklevel=2)

    idat = ocbpy.cycle_boundary.match_data_ocb(ocb, dat_dtime, idat=idat,
                                               max_tol=max_tol, **kwargs)

    return idat
