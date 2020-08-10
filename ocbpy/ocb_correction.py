#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# ----------------------------------------------------------------------------
""" Functions that specify the boundary location as a function of MLT

Functions
---------
circular(mlt, [r_add])
    Return a circular boundary correction for a specified offset
elliptical(mlt, [instrument, method])
    Return the ellptical boundary correction for a data set and method
harmonic(mlt, [instrument, method])
    Return the harmonic boundary correction for a data set and method

References
----------
Burrell, A. G. et al.: AMPERE Polar Cap Boundaries, Ann. Geophys., 38, 481-490,
doi:10.5194/angeo-38-481-2020, 2020.

"""

import numpy as np

from ocbpy.ocb_time import hr2rad


def circular(mlt, r_add=0.0):
    """Return a circular boundary correction

    Parameters
    ----------
    mlt : (float or array-like)
        Magnetic local time in hours (not actually used)
    r_add : (float)
        Offset added to default radius in degrees.  Positive values shift the
        boundary equatorward, whilst negative values shift the boundary
        poleward.  (default=0.0)

    Returns
    -------
    r_corr : (float or array-like)
        Radius correction in degrees at this MLT

    """
    mlt = np.asarray(mlt)
    r_corr = np.full(shape=mlt.shape, fill_value=r_add)

    return r_corr


def elliptical(mlt, instrument='ampere', method='median'):
    """ Return the results of an elliptical correction to the data boundary

    Parameters
    ----------
    mlt : (float or array-like)
        Magnetic local time in hours
    instrument : (str)
        Data set's instrument name (default='ampere')
    method : (str)
        Method used to calculate the elliptical correction, accepts
        'median' or 'gaussian'. (default='median')

    Returns
    -------
    r_corr : (float or array-like)
        Radius correction in degrees at this MLT

    References
    ----------
    Burrell, A. G. et al.: AMPERE Polar Cap Boundaries, Ann. Geophys., 38,
    481-490, doi:10.5194/angeo-38-481-2020, 2020.

    """

    if instrument.lower() != 'ampere':
        raise ValueError("no elliptical correction for {:}".format(instrument))

    method = method.lower()
    coeff = {"median": {"a": 4.01, "e": 0.55, "t": -0.92},
             "gaussian": {"a": 4.41, "e": 0.51, "t": -0.95}}

    if method not in coeff.keys():
        raise ValueError("unknown coefficient computation method")

    mlt_rad = hr2rad(mlt)
    r_corr = (coeff[method]["a"] * (1.0-coeff[method]["e"]**2) /
              (1.0 + coeff[method]["e"]*np.cos(mlt_rad-coeff[method]["t"])))

    # Because this is a poleward correction, return the negative
    return -r_corr


def harmonic(mlt, instrument='ampere', method='median'):
    """Return the results of a harmonic fit correction to the data boundary

    Parameters
    ----------
    mlt : (float or array-like)
        Magnetic local time in hours
    instrument : (str)
        Data set's instrument name (default='ampere')
    method : (str)
        Method used to determine coefficients; accepts 'median' or
        'gaussian' (default='median')

    Returns
    -------
    r_corr : (float or array-like)
        Radius correction in degrees at this MLT

    References
    ----------
    Burrell, A. G. et al.: AMPERE Polar Cap Boundaries, Ann. Geophys., 38,
    481-490, doi:10.5194/angeo-38-481-2020, 2020.

    """
    if instrument.lower() != 'ampere':
        raise ValueError("no harmonic correction for {:}".format(instrument))

    method = method.lower()
    coeff = {'median': [3.31000535, -0.5452934, -1.24389141, 2.42619653,
                        -0.66677988, -1.03467488, -0.30763009, 0.52426756,
                        0.04359299, 0.60201848, 0.50618522, 1.04360529,
                        0.25186405],
             'gaussian': [3.80100827, 0.98555723, -3.43760943, 1.85084271,
                          -0.36730751, -0.81975654, -1.02823832, 1.30637288,
                          -0.53599218, 0.40380183, -1.22462708, -1.2733629,
                          -0.62743381]}

    if method not in coeff.keys():
        raise ValueError("unknown coefficient computation method")

    rad_mlt = hr2rad(mlt)
    r_corr = coeff[method][0] \
        + coeff[method][1] * np.cos(rad_mlt+coeff[method][2]) \
        + coeff[method][3] * np.sin(rad_mlt+coeff[method][4]) \
        + coeff[method][5] * np.cos(2.0 * (rad_mlt+coeff[method][6])) \
        + coeff[method][7] * np.sin(2.0 * (rad_mlt+coeff[method][8])) \
        + coeff[method][9] * np.cos(3.0 * (rad_mlt+coeff[method][10])) \
        + coeff[method][11] * np.sin(3.0 * (rad_mlt+coeff[method][12]))

    # Because this is a poleward shift, return the negative of the correction
    return -r_corr
