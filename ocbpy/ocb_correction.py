#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Functions that specify the boundary location as a function of MLT

Functions
-------------------------------------------------------------------------------
circular(ocb, mlt, r_add)
    Return a circular boundary
ampere_harmonic(ocb, mlt)
    Return the results of a 3rd order harmonic fit to AMPERE/DMSP differences

References
-------------------------------------------------------------------------------
Burrell paper in prep

"""
import numpy as np

def circular(mlt, r_add=0.0):
    """Return a circular boundary correction

    Parameters
    ----------
    mlt : (float)
        Magnetic local time in hours (not actually used)
    r_add : (float)
        Offset added to default radius in degrees.  Positive values shift the
        boundary equatorward, whilst negative values shift the boundary
        poleward.  (default=0.0)

    Returns
    -------
    r_corr : (float)
        Radius correction in degrees at this MLT

    """

    return r_add


def ampere_harmonic(mlt, method='median'):
    """Return the results of a 3rd order harmonic fit to AMPERE/DMSP differences

    Parameters
    ----------
    mlt : (float or array-like)
        Magnetic local time in hours
    method : (str)
        Method used to determine coefficients; accepts median or
        smoothed_gaussian (default='median')

    Returns
    -------
    r_corr : (float)
        Radius correction in degrees at this MLT

    """
    from ocbpy.ocb_time import hr2rad

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
