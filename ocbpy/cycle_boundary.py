#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Routines to match and cycle through the OCboundary class records."""

import datetime as dt
import numpy as np
import warnings

from ocbpy import logger
from ocbpy import ocb_time


def retrieve_all_good_indices(ocb, min_merit=None, max_merit=None, **kwargs):
    """Retrieve all good indices from the OCBoundary class.

    Parameters
    ----------
    ocb : ocbpy.OCBoundary or ocbpy.EABoundary
        Class containing the open-close field line or equatorward auroral
        boundary data
    min_merit : float or NoneType
        Minimum value for the default figure of merit or None to not apply a
        custom minimum (default=None)
    max_merit : float or NoneTye
        Maximum value for the default figure of merit or None to not apply a
        custom maximum (default=None)
    kwargs : dict
        Dict with optional selection criteria.  The key should correspond to a
        data attribute and the value must be a tuple with the first value
        specifying 'max', 'min', 'maxeq', 'mineq', or 'equal' and the second
        value specifying the value to use in the comparison.

    Returns
    -------
    good_ind : list
        List of indices containing good OCBs

    """

    # Save the current record index
    icurrent = ocb.rec_ind

    # Set the record index to allow us to cycle through the entire data set
    ocb.rec_ind = -1

    # Initialize the output data
    good_ind = list()

    # Cycle through all records
    while ocb.rec_ind < ocb.records:
        ocb.get_next_good_ocb_ind(min_merit=min_merit, max_merit=max_merit,
                                  **kwargs)
        if ocb.rec_ind < ocb.records:
            good_ind.append(int(ocb.rec_ind))

    # Reset the record index
    ocb.rec_ind = icurrent

    # Return the good indices
    return good_ind


def match_data_ocb(ocb, dat_dtime, idat=0, max_tol=60, min_merit=None,
                   max_merit=None, **kwargs):
    """Match data records with OCB records.

    Parameters
    ----------
    ocb : ocbpy.OCBoundary, ocbpy.EABoundary, or ocbpy.DualBoundary
        Class containing the open-close field line, equatorial auroral
        boundary, or dual-boundary data
    dat_dtime : list-like
        List or array of datetime objects where data exists
    idat : int
        Current data index (default=0)
    max_tol : int
        maximum seconds between OCB and data record in sec (default=60)
    min_merit : float or NoneType
        Minimum value for the default figure of merit or None to not apply a
        custom minimum (default=None)
    max_merit : float or NoneTye
        Maximum value for the default figure of merit or None to not apply a
        custom maximum (default=None)
    kwargs : dict
        Dict with optional selection criteria.  The key should correspond to a
        data attribute and the value must be a tuple with the first value
        specifying 'max', 'min', 'maxeq', 'mineq', or 'equal' and the second
        value specifying the value to use in the comparison.
    min_sectors : int
        Minimum number of MLT sectors required for good OCB. Deprecated, will
        be removed in version 0.3.1+ (default=7)
    rcent_dev : float
        Maximum number of degrees between the new centre and the AACGM pole.
        Deprecated, will be removed in version 0.3.1+ (default=8.0)
    max_r : float
        Maximum radius for open-closed field line boundary in degrees
        Deprecated, will be removed in version 0.3.1+ (default=23.0)
    min_r : float
        Minimum radius for open-closed field line boundary in degrees
        Deprecated, will be removed in version 0.3.1+ (default=10.0)

    Returns
    -------
    idat : int or NoneType
        Data index for match value, None if all of the data have been searched

    Raises
    ------
    ValueError
        If the input boundary class has an unknown cycling method name

    Notes
    -----
    Updates `ocb.rec_ind` for matched value. This attribute is set to None if
    all of the boundaries have been searched.

    """

    # Add check for deprecated and custom kwargs
    dep_comp = {'min_sectors': ['num_sectors', ('mineq', 7)],
                'rcent_dev': ['r_cent', ('maxeq', 8.0)],
                'max_r': ['r', ('maxeq', 23.0)],
                'min_r': ['r', ('mineq', 10.0)]}
    cust_keys = list(kwargs.keys())

    for ckey in cust_keys:
        if ckey in dep_comp.keys():
            warnings.warn("".join(["Deprecated kwarg will be removed in ",
                                   "version 0.3.1+. To replecate behaviour",
                                   ", use {", dep_comp[ckey][0], ": ",
                                   repr(dep_comp[ckey][1]), "}"]),
                          DeprecationWarning, stacklevel=2)
            del kwargs[ckey]

            if hasattr(ocb, dep_comp[ckey][0]):
                kwargs[dep_comp[ckey][0]] = dep_comp[ckey][1]

    # Initalise the data record limit
    dat_records = len(dat_dtime)

    # Ensure that the indices are good
    if idat >= dat_records:
        return idat
    if ocb.rec_ind >= ocb.records:
        return idat

    # Get the boundary class cycle method
    if hasattr(ocb, "get_next_good_ocb_ind"):
        cycle_method = getattr(ocb, "get_next_good_ocb_ind")
        cycle_kwargs = dict(kwargs)
        cycle_kwargs["min_merit"] = min_merit
        cycle_kwargs["max_merit"] = max_merit
    elif hasattr(ocb, "get_next_good_ind"):
        cycle_method = getattr(ocb, "get_next_good_ind")
        cycle_kwargs = {}

        # If the selection method differs from the default, re-select the
        # good indices
        if(min_merit is not None or max_merit is not None
           or len(kwargs.keys()) > 0):
            logger.info("updating DualBoundary good index pairs")
            ocb.set_good_ind(ocb_min_merit=min_merit, ocb_max_merit=max_merit,
                             ocb_kwargs=kwargs, eab_min_merit=min_merit,
                             eab_max_merit=max_merit, eab_kwargs=kwargs)

            # Re-evaluate record index
            if ocb.rec_ind >= ocb.records:
                logger.error("".join(["after updating selection criteria, ",
                                      "unable to find a good OCB record"]))
                return idat

    else:
        raise ValueError("boundary class missing index cycling method")

    # Get the first reliable boundary estimate if none was provided
    if ocb.rec_ind < 0:
        cycle_method(**cycle_kwargs)
        if ocb.rec_ind >= ocb.records:
            logger.error("unable to find a good OCB record")
            return idat
        else:
            logger.info("".join(["found first good OCB record at ",
                                 "{:}".format(ocb.dtime[ocb.rec_ind])]))

        # Cycle past data occuring before the specified OC boundary point
        first_ocb = ocb.dtime[ocb.rec_ind] - dt.timedelta(seconds=max_tol)
        while dat_dtime[idat] < first_ocb:
            idat += 1

            if idat >= dat_records:
                logger.error("".join(["no input data close enough to the ",
                                      "first record"]))
                return None

    # If the times match, return
    if ocb.dtime[ocb.rec_ind] == dat_dtime[idat]:
        return idat

    # If the times don't match, cycle through both datasets until they do
    while idat < dat_records and ocb.rec_ind < ocb.records:
        # Increase the OCB index until one lies within the desired boundary
        sdiff = (ocb.dtime[ocb.rec_ind] - dat_dtime[idat]).total_seconds()

        if sdiff < -max_tol:
            # Cycle to the next OCB value since the lowest vorticity value
            # is in the future
            cycle_method(**cycle_kwargs)
        elif sdiff > max_tol:
            # Cycle to the next value if no OCB values were close enough
            logger.info("".join(["no OCB data available within [",
                                 "{:d} s] of input measurement".format(max_tol),
                                 " at [{:}]".format(dat_dtime[idat])]))
            idat += 1
        else:
            # Make sure this is the OCB value closest to the input record
            last_sdiff = sdiff
            last_iocb = ocb.rec_ind
            cycle_method(**cycle_kwargs)

            if ocb.rec_ind < ocb.records:
                sdiff = (ocb.dtime[ocb.rec_ind]
                         - dat_dtime[idat]).total_seconds()

                while abs(sdiff) < abs(last_sdiff):
                    last_sdiff = sdiff
                    last_iocb = ocb.rec_ind
                    cycle_method(**cycle_kwargs)
                    if ocb.rec_ind < ocb.records:
                        sdiff = (ocb.dtime[ocb.rec_ind]
                                 - dat_dtime[idat]).total_seconds()

            sdiff = last_sdiff

            # Set the output boundary index
            ocb.rec_ind = last_iocb

            # Return the requested indices
            return idat

    # Return from the last loop
    if idat == 0 and abs(sdiff) > max_tol:
        logger.info("".join(["no OCB data available within ",
                             "[{:d} s] of first measurement ".format(max_tol),
                             "[{:}]".format(dat_dtime[idat])]))
        idat = None

    return idat


def satellite_track(lat, mlt, x1, y1, x2, y2, hemisphere, del_x=1.0, del_y=1.0,
                    past_bound=5.0):
    """Determine whether or not a point lies along the satellite track.

    Parameters
    ----------
    lat : array-like
        AACGM latitude in degrees
    mlt : array-like
        AACGM local time in hours
    x1 : float
        Cartesian x-coordinate of the first boundary location in AACGM degrees
        along the Dawn-Dusk axis
    y1 : float
        Cartesian y-coordinate of the first boundary location in AACGM degrees
        along the Noon-Midnight axis
    x2 : float
        Cartesian x-coordinate of the second boundary location in AACGM degrees
        along the Dawn-Dusk axis
    y2 : float
        Cartesian y-coordinate of the second boundary location in AACGM degrees
        along the Noon-Midnight axis
    hemisphere : int
        Integer (+/- 1) denoting northern/southern hemisphere
    del_x : float
        Allowable distance from the track in AACGM degrees along the x-axis
        (default=1.0)
    del_y : float
        Allowable distance from the track in AACGM degrees along the y-axis
        (default=1.0)
    past_bound : float
        Allowable distance equatorward from the boundary in AACGM degrees
        (default=5.0)

    Returns
    -------
    good : array-like
         Array of booleans that are True if location is along the track and
         False if the location falls outside of the track

    Raises
    ------
    ValueError
        If the boundary values are negative or if an unknown hemisphere is
        specified

    """
    # Evaluate the boundary input
    if del_x < 0.0 or del_y < 0.0:
        raise ValueError("x- and y-axis allowable difference must be positive")

    if past_bound < 0.0:
        raise ValueError("equatorward buffer for track must be positive")

    if hemisphere not in [1.0, -1.0]:
        raise ValueError("hemisphere expecting +/- 1")

    # Ensure the input is array-like
    lat = np.asarray(lat)
    mlt = np.asarray(mlt)

    # Get the equation of the line defining the upper and lower bounds
    slope = (y1 - y2) / (x1 - x2)
    high_int = y1 + del_y - slope * (x1 + del_x)
    low_int = y1 - del_y - slope * (x1 - del_x)

    # Determine the Cartesian coordinates of the input point
    rad_in = ocb_time.hr2rad(mlt - 6.0)
    x_in = (90.0 - hemisphere * lat) * np.cos(rad_in)
    y_in = (90.0 - hemisphere * lat) * np.sin(rad_in)

    # Calculate the upper and lower limits for each of the x-inputs
    y_low = slope * x_in + low_int
    y_high = slope * x_in + high_int

    good = (y_in >= y_low) & (y_in <= y_high)

    # Ensure the latitude is not too low
    if np.any(good):
        r1 = np.sqrt((x1 - x_in)**2 + (y1 - y_in)**2)
        r2 = np.sqrt((x2 - x_in)**2 + (y2 - y_in)**2)

        # Find which boundary is closest to the good points
        ione = np.where(good & (r1 <= r2))
        itwo = np.where(good & (r2 < r1))

        # Evaluate the closest lower boundary limit
        if len(lat[ione]) > 0:
            lat_bound = 90.0 - np.sqrt(x1**2 + y1**2) - past_bound
            ipast = np.where(abs(lat[ione]) < lat_bound)
            good[ione[0][ipast]] = False

        if len(lat[itwo]) > 0:
            lat_bound = 90.0 - np.sqrt(x2**2 + y2**2) - past_bound
            ipast = np.where(abs(lat[itwo]) < lat_bound)
            good[itwo[0][ipast]] = False

    # Return the boolean mask
    return good
