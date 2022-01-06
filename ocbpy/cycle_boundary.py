#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Routines to cycle through the OCboundary class records."""

from ocbpy import logger


def retrieve_all_good_indices(ocb):
    """Retrieve all good indices from the OCBoundary class.

    Parameters
    ----------
    ocb : ocbpy.OCBoundary
        Class containing the open-close field line or equatorward auroral
        boundary data

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
        ocb.get_next_good_ocb_ind()
        if ocb.rec_ind < ocb.records:
            good_ind.append(int(ocb.rec_ind))

    # Reset the record index
    ocb.rec_ind = icurrent

    # Return the good indices
    return good_ind


def match_data_ocb(ocb, dat_dtime, idat=0, max_tol=600, min_sectors=7,
                   rcent_dev=8.0, max_r=23.0, min_r=10.0):
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
    min_sectors : int
        Minimum number of MLT sectors required for good OCB. (default=7)
    rcent_dev : float
        Maximum number of degrees between the new centre and the AACGM pole
        (default=8.0)
    max_r : float
        Maximum radius for open-closed field line boundary in degrees
        (default=23.0)
    min_r : float
        Minimum radius for open-closed field line boundary in degrees
        (default=10.0)

    Returns
    -------
    idat : int or NoneType
        Data index for match value, None if all of the data have been searched

    Notes
    -----
    Updates `ocb.rec_ind` for matched value. This attribute is set to None if
    all of the boundaries have been searched.

    """

    dat_records = len(dat_dtime)

    # Ensure that the indices are good
    if idat >= dat_records:
        return idat
    if ocb.rec_ind >= ocb.records:
        return idat

    # Get the first reliable circle boundary estimate if none was provided
    if ocb.rec_ind < 0:
        ocb.get_next_good_ocb_ind(min_sectors=min_sectors, rcent_dev=rcent_dev,
                                  max_r=max_r, min_r=min_r)
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
            ocb.get_next_good_ocb_ind(min_sectors=min_sectors,
                                      rcent_dev=rcent_dev, max_r=max_r,
                                      min_r=min_r)
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
            ocb.get_next_good_ocb_ind(min_sectors=min_sectors,
                                      rcent_dev=rcent_dev, max_r=max_r,
                                      min_r=min_r)

            if ocb.rec_ind < ocb.records:
                sdiff = (ocb.dtime[ocb.rec_ind]
                         - dat_dtime[idat]).total_seconds()

                while abs(sdiff) < abs(last_sdiff):
                    last_sdiff = sdiff
                    last_iocb = ocb.rec_ind
                    ocb.get_next_good_ocb_ind(min_sectors=min_sectors,
                                              rcent_dev=rcent_dev, max_r=max_r,
                                              min_r=min_r)
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
                             "[{:}]".format(dat_dtime[idat])])

    return idat
