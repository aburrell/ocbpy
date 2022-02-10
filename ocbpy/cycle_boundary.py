#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Routines to cycle through the OCboundary class records."""

import datetime as dt
import warnings

from ocbpy import logger


def retrieve_all_good_indices(ocb, min_merit=None, max_merit=None, **kwargs):
    """Retrieve all good indices from the OCBoundary class.

    Parameters
    ----------
    ocb : ocbpy.OCBoundary
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


def match_data_ocb(ocb, dat_dtime, idat=0, max_tol=600, min_merit=None,
                   max_merit=None, **kwargs):
    """Match data records with OCB records.

    Parameters
    ----------
    ocb : ocbpy.OCBoundary
        Class containing the open-close field line or equatorial auroral
        boundary data
    dat_dtime : list-like
        List or array of datetime objects where data exists
    idat : int
        Current data index (default=0)
    max_tol : int
        maximum seconds between OCB and data record in sec (default=600)
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

    # Get the first reliable boundary estimate if none was provided
    if ocb.rec_ind < 0:
        ocb.get_next_good_ocb_ind(min_merit=min_merit, max_merit=max_merit,
                                  **kwargs)
        if ocb.rec_ind >= ocb.records:
            logger.error("".join(["unable to find a good OCB record in ",
                                  ocb.filename]))
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
            ocb.get_next_good_ocb_ind(min_merit=min_merit, max_merit=max_merit,
                                      **kwargs)
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
            ocb.get_next_good_ocb_ind(min_merit=min_merit, max_merit=max_merit,
                                      **kwargs)

            if ocb.rec_ind < ocb.records:
                sdiff = (ocb.dtime[ocb.rec_ind]
                         - dat_dtime[idat]).total_seconds()

                while abs(sdiff) < abs(last_sdiff):
                    last_sdiff = sdiff
                    last_iocb = ocb.rec_ind
                    ocb.get_next_good_ocb_ind(min_merit=min_merit,
                                              max_merit=max_merit, **kwargs)
                    if ocb.rec_ind < ocb.records:
                        sdiff = (ocb.dtime[ocb.rec_ind]
                                 - dat_dtime[idat]).total_seconds()

            sdiff = last_sdiff
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
