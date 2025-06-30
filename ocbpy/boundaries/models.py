#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DOI: 10.5281/zenodo.1179230
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Functions that provide boundary locations through a mathematical model.

References
----------
.. [8] Starkov, G. V. (1994) Mathematical model of the auroral boundaries,
   Geomagnetism and Aeronomy, English Translation, 34(3), 331-336.

"""

import numpy as np


def starkov_auroral_boundary(mlt, al=-1, bnd='ocb'):
    """Calculate the location of the auroral boundaries.

    Parameters
    ----------
    mlt : float or array-like
        Magnetic local time in hours
    al : float or int
        AL geomagnetic index, Auroral Electrojet Lower envelope in nT
        (default=-1)
    bnd : str
        Boundary to calculate, expects one of 'ocb', 'eab', or 'diffuse'
        (default='ocb')

    Returns
    -------
    bnd_lat : float or array-like
        Location of the boundary in degrees away from the pole in corrected
        geomagnetic coordinates for the specified magnetic local times.

    References
    ----------
    [8]_

    """

    # Calculate the AL dependence of the coefficients
    A0 = starkov_coefficient_values(al, "A0", bnd)
    A1 = starkov_coefficient_values(al, "A1", bnd)
    alpha1 = starkov_coefficient_values(al, "alpha1", bnd)
    A2 = starkov_coefficient_values(al, "A2", bnd)
    alpha2 = starkov_coefficient_values(al, "alpha2", bnd)
    A3 = starkov_coefficient_values(al, "A3", bnd)
    alpha3 = starkov_coefficient_values(al, "alpha3", bnd)

    # Calculate the angular inputs in radians
    rad1 = np.radians(15.0 * (mlt + alpha1))
    rad2 = np.radians(15.0 * (2.0 * mlt + alpha2))
    rad3 = np.radians(15.0 * (3.0 * mlt + alpha3))

    # Calculate the boundary location in degrees latitude
    bnd_lat = A0 + A1 * np.cos(rad1) + A2 * np.cos(rad2) + A3 * np.cos(rad3)

    # Ensure all co-latitudes are positive or zero
    if np.asarray(bnd_lat).shape == ():
        if bnd_lat < 0:
            bnd_lat = 0.0
    else:
        bnd_lat[bnd_lat < 0] = 0.0

    return bnd_lat


def starkov_coefficient_values(al, coeff_name, bnd):
    """Calculate the Starkov auroral model coefficient values.

    Parameters
    ----------
    al : float or int
        AL geomagnetic index, Auroral Electrojet Lower envelope in nT
    coeff_name : str
        Coefficient name, expects one of 'A0', 'A1', 'A2', 'A3', 'alpha1',
        'alpha2', or 'alpha3'
    bnd : str
        Boundary to calculate, expects one of 'ocb', 'eab', or 'diffuse'

    Returns
    -------
    coeff : float
        Coefficient value in hours (alpha) or degrees latitude (A)

    References
    ----------
    [8]_

    """
    # Define the model coefficients for each type of boundary
    coeff_terms = {'A0': {'ocb': [-.07, 24.54, -12.53, 2.15],
                          'eab': [1.16, 23.21, -10.97, 2.03],
                          'diffuse': [3.44, 29.77, -16.38, 3.35]},
                   'A1': {'ocb': [-10.06, 19.83, -9.33, 1.24],
                          'eab': [-9.59, 17.78, -7.20, 0.96],
                          'diffuse': [-2.41, 7.89, -4.32, 0.87]},
                   'alpha1': {'ocb': [-6.61, 10.17, -5.80, 1.19],
                              'eab': [-2.22, 1.50, -0.58, 0.08],
                              'diffuse': [-1.68, -2.48, 1.58, -0.28]},
                   'A2': {'ocb': [-4.44, 7.47, -3.01, 0.25],
                          'eab': [-12.07, 17.49, -7.96, 1.15],
                          'diffuse': [-0.74, 3.94, -3.09, 0.72]},
                   'alpha2': {'ocb': [6.37, -1.10, 0.34, -0.38],
                              'eab': [-23.98, 42.79, -26.96, 5.56],
                              'diffuse': [8.69, -20.73, 13.03, -2.14]},
                   'A3': {'ocb': [-3.77, 7.90, -4.73, 0.91],
                          'eab': [-6.56, 11.44, -6.73, 1.31],
                          'diffuse': [-2.12, 3.24, -1.67, 0.31]},
                   'alpha3': {'ocb': [-4.48, 10.16, -5.87, 0.98],
                              'eab': [-20.07, 36.67, -24.20, 5.11],
                              'diffuse': [8.61, -5.34, -1.36, 0.76]}}

    # Calculate the desired coefficient
    log_al = np.log10(abs(al))
    coeff = coeff_terms[coeff_name][bnd][0] + coeff_terms[coeff_name][bnd][
        1] * log_al + coeff_terms[coeff_name][bnd][2] * (
            log_al**2) + coeff_terms[coeff_name][bnd][3] * (log_al**3)

    return coeff
