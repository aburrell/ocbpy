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
.. [9] Gussenhoven, M. S. et al. (1983) Systematics of the Equatorward Diffuse
   Auroral Boundary, J. Geophys. Res, 88(A7), 5692-5708.
.. [10] Umbach and Jones (2003) A Few Methods for Fitting Circles to Data,
   IEEE Transactions on Instruments and Measurements, 52(6), pp 1881-1885

"""

import numpy as np

from ocbpy import ocb_time


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


def gussenhoven_equatorward_auroral_boundary(mlt, kp=0, model='circle'):
    """Calculate the location of the diffuse EAB.

    Parameters
    ----------
    mlt : float or array-like
        Magnetic local time in hours
    kp : float or int
        Decimal Kp geomagnetic index, ranges from 0 to 9 with "+" translating
        to plus one third and "-" translating to minus one third (default=0)
    model : str
        Model flavour, with 'binned' using the value from the available hourly
        solutions or NaN if that hour is absent, 'closest' to use the closest
        hourly bin, or 'circle' to use the value of the circle fit to the
        available hourly solutions (default='circle')

    Returns
    -------
    bnd_lat : float or array-like
        Location of the boundary in degrees away from the pole in corrected
        geomagnetic coordinates for the specified magnetic local times.

    Raises
    ------
    ValueError
        If an unknown model method is requested.

    References
    ----------
    [9]_

    """
    closest = True if model.lower() == 'closest' else False
    
    # If desired, calculate the integer hour
    if model.lower() in ['binned', 'closest']:
        imlt = np.floor(mlt).astype(int)

        if imlt.shape == ():
            imlt = np.array([imlt])
    else:
        imlt = None

    # Get the desired latitude boundaries
    colats, mlts = gussenhoven_colatitudes(kp, mlt_inds=imlt, closest=closest)

    # If desired, fit the co-latitude boundaries and return the locations
    # at the exact MLT values
    if model.lower() in ['binned', 'closest']:
        bnd_lat = colats
    elif model.lower() == "circle":
        # Fit a circle to the boundaries at this Kp
        phi_cent, r_cent, radius, _ = circle_fit(mlts, colats)
        
        # Calculate the circle location at the desired MLTs. Use the positive
        # solution of the quadratic equation, as radius >= r_cent
        theta = ocb_time.hr2rad(mlt) - phi_cent
        bnd_lat = r_cent * np.cos(theta) + np.sqrt(
            radius**2 - r_cent**2 * (np.sin(theta))**2)
    else:
        raise ValueError('unknown model method: {:}'.format(model))

    return bnd_lat


def gussenhoven_colatitudes(kp, mlt_inds=None, closest=False):
    """Calculate the Gussenhoven EAB model co-latitude values.

    Parameters
    ----------
    kp : float or int
        Decimal Kp geomagnetic index, ranges from 0 to 9 with "+" translating
        to plus one third and "-" translating to minus one third
    mlt_inds : array-like or NoneType
        MLT hourly bins at which boundary locations will be returned or None
        to return all available values (default=None)
    closest : bool
        If False, will return NaN for the request for an MLT value that does not
        exist. If True, will find the closest binned value and return that
        instead (default=False)

    Returns
    -------
    colats : array-like
        Co-latitude values in degrees away from the pole for the requested
        MLT bins
    mlts : array-like
        MLT bin values for each co-latitude

    References
    ----------
    [9]_

    """
    # Set the function values for each MLT bin
    offset = {0: 66.1, 1: 65.1, 4: 67.7, 5: 67.8, 6: 68.2, 7: 68.9, 8: 69.3,
              9: 69.5, 10: 69.6, 11: 70.1, 12: 69.4, 15: 70.9, 16: 71.6,
              17: 71.1, 18: 71.2, 19: 70.4, 20: 69.4, 21: 68.6, 22: 67.9,
              23: 67.8}
    slope = {0: -1.99, 1: -1.55, 4: -1.48, 5: -1.87, 6: -1.90, 7: -1.91,
             8: -1.87, 9: -1.67, 10: -1.41, 11: -1.25, 12: -0.84, 15: -0.81,
             16: -1.28, 17: -1.31, 18: -1.74, 19: -1.83, 20: -1.89, 21: -1.86,
             22: -1.78, 23: -2.07}

    # Ensure the MLT indices are set
    mlt_keys = np.array(list(offset.keys()))
    if mlt_inds is None:
        mlt_inds = mlt_keys

    # Initialize the output
    colats = np.full(shape=mlt_inds.shape, fill_value=np.nan)
    mlts = np.asarray(mlt_inds)

    # Cycle through each MLT index
    for i, imlt in enumerate(mlt_inds):
        if imlt in mlt_keys:
            # Set the calculation key
            colat_key = imlt
        elif closest:
            # Find the closest time with a boundary solution
            iclose = abs(imlt - mlt_keys).argmin()
            colat_key = mlt_keys[iclose]
            mlts[i] = colat_key
        else:
            colat_key = None

        if colat_key is not None:
            # Find the closest co-latitude solution
            colat = 90 - (offset[colat_key] + slope[colat_key] * kp)

            # Ensure there are no values outside of the allowed range
            if colat < 0.0:
                colat = 0.0
            elif colat > 90.0:
                colat = 90.0

            # Save the output
            colats[i] = colat

    return colats, mlts


def circle_fit(mlt, colat):
    """Fit a circle to boundary estimates.

    Parameters
    ----------
    mlt : array-like
        Array of MLT values in hours for the boundary locations
    colat : array-like
        Array of magnetic co-latitude values in degrees denoting the boundary
        locations, paired to the `mlt` inputs

    Returns
    -------
    phi_cent : float
        MLT location of the circle centers in radians
    r_cent : float
        Co-latitude of the circle center in degrees
    radius : float
        Radius of the circle in degrees latitude
    r_err : float
        RMS error of the fit

    Raises
    ------
    ValueError
        If the inputs are the wrong shape

    References
    ----------
    Section D in [10]_

    """
    # Ensure the arrays are of equal length
    phi = ocb_time.hr2rad(np.asarray(mlt))
    colat = np.asarray(colat)

    if phi.shape != colat.shape:
        raise ValueError("input MLT and MLat arrays must be the same shape")

    if len(phi.shape) != 1:
        raise ValueError('routine only supports 1D arrays')

    # Get the cartesian values of the boundary points.  Define the x-axis to
    # lie along 0 MLT and the y-axis to lie along 6 MLT, with the origin at the
    # magnetic pole.
    x_bound = colat * np.cos(phi)
    y_bound = colat * np.sin(phi)

    # Apply the circle fitting algorithm, a modified least-squares method,
    # from Umbach and Jones section D
    sum_x = np.sum(x_bound)
    sum_y = np.sum(y_bound)
    sum_x2 = np.sum(x_bound**2)
    sum_y2 = np.sum(y_bound**2)
    sum_xy = np.sum((x_bound * y_bound))
    sum_xy2 = np.sum(x_bound * (y_bound**2))
    sum_yx2 = np.sum(y_bound * (x_bound**2))
    sum_x3 = np.sum(x_bound * x_bound * x_bound)
    sum_y3 = np.sum(y_bound * y_bound * y_bound)

    fit_a = (colat.shape[0] * sum_x2) - sum_x**2
    fit_b = (colat.shape[0] * sum_xy) - (sum_x * sum_y)
    fit_c = (colat.shape[0] * sum_y2) - sum_y**2
    fit_d = 0.5 * ((colat.shape[0] * sum_xy2) - (sum_x * sum_y2)
                   + (colat.shape[0] * sum_x3) - (sum_x * sum_x2))
    fit_e = 0.5 * ((colat.shape[0] * sum_yx2) - (sum_y * sum_x2)
                   + (colat.shape[0] * sum_y3) - (sum_y * sum_y2))

    # Determine the centre of the fitted circle in Cartesian coordinates
    circle_denom = (fit_a * fit_c) - fit_b**2
    major_a = ((fit_d * fit_c) - (fit_b * fit_e)) / circle_denom
    minor_b = ((fit_a * fit_e) - (fit_b * fit_d)) / circle_denom

    # Convert the circle centre to co-latitude in degrees
    r_cent = np.sqrt(major_a**2 + minor_b**2)
    phi_cent = np.arctan2(minor_b, major_a)

    # Determine the radius of the fitted circle
    rad_bound = np.sqrt((x_bound - major_a)**2 + (y_bound - minor_b)**2)
    radius = rad_bound.mean()

    # Calculate the error of the radius
    r_err = np.sqrt(np.mean((rad_bound - radius)**2))

    return phi_cent, r_cent, radius, r_err
