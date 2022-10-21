#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# ----------------------------------------------------------------------------
""" Functions that specify the boundary location as a function of MLT

References
----------
.. [4] Burrell, A. G. et al.: AMPERE Polar Cap Boundaries, Ann. Geophys., 38,
   481-490, doi:10.5194/angeo-38-481-2020, 2020.
.. [6] Chisham, G. et al.: Ionospheric Boundaries Derived from Auroral Images,
    in prep, 2022.

"""

import numpy as np

from ocbpy.ocb_time import hr2rad


def circular(mlt, r_add=0.0):
    """Return a circular boundary correction.

    Parameters
    ----------
    mlt : float or array-like
        Magnetic local time in hours (not actually used)
    r_add : float
        Offset added to default radius in degrees.  Positive values shift the
        boundary equatorward, whilst negative values shift the boundary
        poleward.  (default=0.0)

    Returns
    -------
    r_corr : float or array-like
        Radius correction in degrees at this MLT

    """
    mlt = np.asarray(mlt)
    r_corr = np.full(shape=mlt.shape, fill_value=r_add)

    return r_corr


def elliptical(mlt, instrument='ampere', method='median'):
    """Return the results of an elliptical correction to the data boundary.

    Parameters
    ----------
    mlt : float or array-like
        Magnetic local time in hours
    instrument : str
        Data set's instrument name (default='ampere')
    method : str
        Method used to calculate the elliptical correction, accepts
        'median' or 'gaussian'. (default='median')

    Returns
    -------
    r_corr : float or array-like
        Radius correction in degrees at this MLT

    References
    ----------
    Prefered AMPERE boundary correction validated in [4]_.

    """

    if instrument.lower() != 'ampere':
        raise ValueError("no elliptical correction for {:}".format(instrument))

    method = method.lower()
    coeff = {"median": {"a": 4.01, "e": 0.55, "t": -0.92},
             "gaussian": {"a": 4.41, "e": 0.51, "t": -0.95}}

    if method not in coeff.keys():
        raise ValueError("unknown coefficient computation method")

    mlt_rad = hr2rad(mlt)
    r_corr = (coeff[method]["a"] * (1.0 - coeff[method]["e"]**2)
              / (1.0 + coeff[method]["e"] * np.cos(mlt_rad
                                                   - coeff[method]["t"])))

    # Because this is a poleward correction, return the negative
    return -r_corr


def harmonic(mlt, instrument='ampere', method='median'):
    """Return the results of a harmonic fit correction to the data boundary.

    Parameters
    ----------
    mlt : float or array-like
        Magnetic local time in hours
    instrument : str
        Data set's instrument name (default='ampere')
    method : str
        Method used to determine coefficients; accepts 'median' or
        'gaussian' when `instrument` is 'ampere'.  Otherwise, accepts 'eab' or
        'ocb'. (default='median')

    Returns
    -------
    r_corr : float or array-like
        Radius correction in degrees at this MLT

    References
    ----------
    AMPERE boundaries obtained from [4]_. IMAGE boundaries obtained from [6]_.

    """
    # Define the harmonic coefficients for each instrument and method/location
    coeff = {'ampere': {'median': [3.31000535, -0.5452934, -1.24389141,
                                   2.42619653, -0.66677988, -1.03467488,
                                   -0.30763009, 0.52426756, 0.04359299,
                                   0.60201848, 0.50618522, 1.04360529,
                                   0.25186405],
                        'gaussian': [3.80100827, 0.98555723, -3.43760943,
                                     1.85084271, -0.36730751, -0.81975654,
                                     -1.02823832, 1.30637288, -0.53599218,
                                     0.40380183, -1.22462708, -1.2733629,
                                     -0.62743381]},
             'si12': {'ocb': [0.0405, -1.5078, 1.0, 0.5095, 1.0, 0.9371, 1.0,
                              0.0708, 1.0, 0.0, 1.0, 0.0, 1.0],
                      'eab': [-0.1447, -1.9779, 1.0, 2.6799, 1.0, 0.5778, 1.0,
                              -1.2297, 1.0, 0.0, 1.0, 0.0, 1.0]},
             'si13': {'ocb': [0.5797, -0.6837, 1.0, -0.5020, 1.0, 0.2971, 1.0,
                              -0.4173, 1.0, 0.0, 1.0, 0.0, 1.0],
                      'eab': [0.2500, -2.9931, 1.0, 0.8818, 1.0, 0.8511, 1.0,
                              -0.6300, 1.0, 0.0, 1.0, 0.0, 1.0]},
             'wic': {'ocb': [1.0298, -1.1249, 1.0, -0.7380, 1.0, 0.1838, 1.0,
                             -0.6171, 1.0, 0.0, 1.0, 0.0, 1.0],
                     'eab': [-0.4935, -2.1186, 1.0, 0.3188, 1.0, 0.5749, 1.0,
                             -0.3118, 1.0, 0.0, 1.0, 0.0, 1.0]}}

    # Check the inputs
    inst = instrument.lower()
    if inst not in coeff.keys():
        raise ValueError("no harmonic correction for {:}".format(instrument))

    method = method.lower()
    if method not in coeff[inst].keys():
        raise ValueError("".join(["unknown coefficient computation method, ",
                                  "expects one of: {:}".format(
                                      coeff[inst].keys())]))

    # Calculate the offset
    rad_mlt = hr2rad(mlt)
    r_corr = coeff[inst][method][0] \
        + coeff[inst][method][1] * np.cos(rad_mlt + coeff[inst][method][2]) \
        + coeff[inst][method][3] * np.sin(rad_mlt + coeff[inst][method][4]) \
        + coeff[inst][method][5] * np.cos(2.0 * (
            rad_mlt + coeff[inst][method][6])) \
        + coeff[inst][method][7] * np.sin(2.0 * (
            rad_mlt + coeff[inst][method][8])) \
        + coeff[inst][method][9] * np.cos(3.0 * (
            rad_mlt + coeff[inst][method][10])) \
        + coeff[inst][method][11] * np.sin(3.0 * (
            rad_mlt + coeff[inst][method][12]))

    # Because this is a poleward shift, return the negative of the correction
    return -r_corr
