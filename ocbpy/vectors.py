#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Functions for performing vector transformations."""

import numpy as np

from ocbpy import ocb_time


def get_pole_loc(phi_cent, r_cent):
    """Convert a second coordinate system's pole location to lt/lat.

    Parameters
    ----------
    phi_cent : float or array-like
        The angle from midnight in the base coordinate system of the destination
        coordinate system pole in degrees
    r_cent : float or array-like
        The co-latitude in the base coordinate system of the destination
        coordinate system pole in degrees

    Returns
    -------
    pole_lt : float or array-like
        The base coordinate system LT of the destination coordinate system pole
        location in hours; float if both inputs are float-like
    pole_lat : float or array-like
        The base coordinate system latitude of the destination coordinate
        system pole location in degrees; float if both inputs are float-like

    """
    pole_lt = ocb_time.deg2hr(np.asarray(phi_cent))
    pole_lat = 90.0 - np.asarray(r_cent)

    if len(pole_lt.shape) == 0 and len(pole_lat.shape) == 0:
        pole_lt = float(pole_lt)
        pole_lat = float(pole_lat)

    return pole_lt, pole_lat


def calc_vec_pole_angle(data_lt, data_lat, pole_lt, pole_lat):
    """Find the angle between the base pole, data, and the other pole.

    Parameters
    ----------
    data_lt : float or array-like
        Local time of data in the base coordinate system in hours
    data_lat : float or array-like
        Latitude of data in the base coordinate system in degrees
    pole_lt : float or array-like
        Local time of destination pole location in the base coordinate system
        in hours
    pole_lat : float or array-like
        Latitude of destination pole location in the base coordinate system in
        degrees

    Returns
    -------
    pole_angle : float or array-like
        Angle between the base coordinate system pole, data location, and the
        destination coordinate sytem pole (as located in the base coordinate
        system) in degrees

    Notes
    -----
    The data coordinates must be in the base coordinate system. Finds the
    `pole_angle` using spherical trigonometry.

    """
    # Convert the local time values to radians, after calculating the
    # difference between the destination pole and the data. Restrict data
    # from -pi to pi
    del_long = ocb_time.hr2rad(np.asarray(pole_lt) - np.asarray(data_lt),
                               max_range=np.pi)

    # Initalize the output
    pole_angle = np.full(shape=del_long.shape, fill_value=np.nan)

    # Ensure the latitudes are array-like
    data_lat = np.asarray(data_lat)
    pole_lat = np.asarray(pole_lat)

    # Assign the extreme values
    if len(del_long.shape) == 0:
        if del_long in [-np.pi, 0.0, np.pi]:
            if abs(data_lat) > abs(pole_lat):
                pole_angle = 180.0
            else:
                pole_angle = 0.0
            return pole_angle
    else:
        flat_mask = (((del_long == 0) | (abs(del_long) == np.pi))
                     & np.greater(abs(data_lat), abs(pole_lat),
                                  where=~np.isnan(del_long)))
        zero_mask = (((del_long == 0) | (abs(del_long) == np.pi))
                     & np.less_equal(abs(data_lat), abs(pole_lat),
                                     where=~np.isnan(del_long)))

        pole_angle[flat_mask] = 180.0
        pole_angle[zero_mask] = 0.0
        update_mask = (~zero_mask & ~flat_mask)

        if not np.any(update_mask):
            return pole_angle

    # Find the distance in radians between the two poles
    hemisphere = np.sign(pole_lat)
    rad_pole = hemisphere * np.pi * 0.5
    del_pole = hemisphere * (rad_pole - np.radians(pole_lat))

    # Get the distance in radians between the base pole and the data point
    del_vect = hemisphere * (rad_pole - np.radians(data_lat))

    # Use the Vincenty formula for a sphere
    del_dest = np.arctan2(
        np.sqrt((np.cos(np.radians(pole_lat)) * np.sin(del_long))**2
                + (np.cos(np.radians(data_lat)) * np.sin(np.radians(pole_lat))
                   - np.sin(np.radians(data_lat)) * np.cos(np.radians(pole_lat))
                   * np.cos(del_long))**2),
        np.sin(np.radians(data_lat)) * np.sin(np.radians(pole_lat))
        + np.cos(np.radians(data_lat)) * np.cos(np.radians(pole_lat))
        * np.cos(del_long))

    # Use the half-angle formula to get the pole angle
    sum_sides = 0.5 * (del_vect + del_dest + del_pole)
    half_angle = np.sqrt(np.sin(sum_sides) * np.sin(sum_sides - del_pole)
                         / (np.sin(del_vect) * np.sin(del_dest)))

    if pole_angle.shape == ():
        pole_angle = np.degrees(2.0 * np.arccos(half_angle))
    else:
        pole_angle[update_mask] = np.degrees(2.0 * np.arccos(
            half_angle[update_mask]))

    return pole_angle


def define_pole_quadrants(data_lt, pole_lt, pole_angle):
    """Define LT quadrants for the pole of the another coordinate system.

    Parameters
    ----------
    data_lt : float or array-like
        Local time of data in the base coordinate system in hours
    pole_lt : float or array-like
        Local time of destination pole location in the base coordinate system
        in hours
    pole_angle : float or array-like
        Angle between the base coordinate system pole, data location, and the
        destination coordinate sytem pole (as located in the base coordinate
        system) in degrees

    Returns
    -------
    pole_quad : float or array-like
        Specifies the base coordinate system LT quadrant for each pole/data pair

    Notes
    -----
    North (N) and East (E) are defined by the base coordinate system directions
    centred on the the data vector location, assuming vertical is positive
    downwards. Quadrants: 1 [N, E]; 2 [N, W]; 3 [S, W]; 4 [S, E]; 0 [undefined]

    """
    # Determine where the destination pole is relative to the data vector
    del_lt = np.asarray(pole_lt) - np.asarray(data_lt)

    # Initalize the output
    pole_quad = np.zeros(shape=np.asarray(del_lt).shape)

    # Determine which differences need to be
    neg_mask = np.less(del_lt, 0.0, where=~np.isnan(del_lt)) & ~np.isnan(del_lt)
    while np.any(neg_mask):
        if len(del_lt.shape) == 0:
            del_lt += 24.0
            neg_mask = np.less(del_lt, 0.0)  # Has one finite value
        else:
            del_lt[neg_mask] += 24.0
            neg_mask = np.less(del_lt, 0.0,
                               where=~np.isnan(del_lt)) & ~np.isnan(del_lt)

    large_mask = np.greater_equal(abs(del_lt), 24.0,
                                  where=~np.isnan(del_lt)) & ~np.isnan(del_lt)
    while np.any(large_mask):
        if len(del_lt.shape) == 0:
            del_lt -= 24.0 * np.sign(del_lt)
            large_mask = np.greater_equal(abs(del_lt), 24.0)  # One finite value
        else:
            del_lt[large_mask] -= 24.0 * np.sign(del_lt[large_mask])
            large_mask = np.greater_equal(abs(del_lt), 24.0,
                                          where=~np.isnan(del_lt)) & ~np.isnan(
                                              del_lt)

    # Find the quadrant in which the OCB pole lies
    nan_mask = ~np.isnan(pole_angle) & ~np.isnan(del_lt)
    quad1_mask = np.less(pole_angle, 90.0, where=nan_mask) & np.less(
        del_lt, 12.0, where=nan_mask) & nan_mask
    quad2_mask = np.less(pole_angle, 90.0, where=nan_mask) & np.greater_equal(
        del_lt, 12.0, where=nan_mask) & nan_mask
    quad3_mask = np.greater_equal(
        pole_angle, 90.0, where=nan_mask) & np.greater_equal(
            del_lt, 12.0, where=nan_mask) & nan_mask
    quad4_mask = np.greater_equal(pole_angle, 90.0, where=nan_mask) & np.less(
        del_lt, 12.0, where=nan_mask) & nan_mask

    if len(pole_quad.shape) == 0:
        if np.all(quad1_mask):
            pole_quad = np.asarray(1)
        elif np.all(quad2_mask):
            pole_quad = np.asarray(2)
        elif np.all(quad3_mask):
            pole_quad = np.asarray(3)
        elif np.all(quad4_mask):
            pole_quad = np.asarray(4)
    else:
        pole_quad[quad1_mask] = 1
        pole_quad[quad2_mask] = 2
        pole_quad[quad3_mask] = 3
        pole_quad[quad4_mask] = 4

    return pole_quad


def define_vect_quadrants(vect_n, vect_e):
    """Define LT quadrants for the data vectors.

    Parameters
    ----------
    vect_n : float or array-like
        North component of data vector in the base coordinate system in units
        of degrees latitude
    vect_e : float or array-like
        East component of data vector in the base coordinate system in units of
        degrees latitude

    Returns
    -------
    vect_quad : float or array-like
        Specifies the base coordinate system LT quadrant for each data vector

    Notes
    -----
    North (N) and East (E) are defined by the base coordinate system directions
    centred on the data vector location, assuming vertical is positive downwards
    Quadrants: 1 [N, E]; 2 [N, W]; 3 [S, W]; 4 [S, E]; 0 [undefined]

    """
    # Get the masks for non-fill values in each quadrant
    nan_mask = ~np.isnan(vect_n) & ~np.isnan(vect_e)
    quad1_mask = np.greater_equal(
        vect_n, 0.0, where=nan_mask) & np.greater_equal(
            vect_e, 0.0, where=nan_mask) & nan_mask
    quad2_mask = np.greater_equal(vect_n, 0.0, where=nan_mask) & np.less(
        vect_e, 0.0, where=nan_mask) & nan_mask
    quad3_mask = np.less(vect_n, 0.0, where=nan_mask) & np.less(
        vect_e, 0.0, where=nan_mask) & nan_mask
    quad4_mask = np.less(vect_n, 0.0, where=nan_mask) & np.greater_equal(
        vect_e, 0.0, where=nan_mask) & nan_mask

    # Initialize the output
    vect_quad = np.zeros(shape=nan_mask.shape)

    # Assign the quandrants to the output
    if len(vect_quad.shape) == 0:
        if np.all(quad1_mask):
            vect_quad = np.asarray(1)
        elif np.all(quad2_mask):
            vect_quad = np.asarray(2)
        elif np.all(quad3_mask):
            vect_quad = np.asarray(3)
        elif np.all(quad4_mask):
            vect_quad = np.asarray(4)
    else:
        vect_quad[quad1_mask] = 1
        vect_quad[quad2_mask] = 2
        vect_quad[quad3_mask] = 3
        vect_quad[quad4_mask] = 4

    return vect_quad


def calc_dest_polar_angle(pole_quad, vect_quad, base_naz_angle, pole_angle):
    """Calcuate the North azimuth angle for the destination pole.

    Parameters
    ----------
    pole_quad : float or array-like
        Specifies the base coordinate system LT quadrant for each pole/data pair
    vect_quad : float or array-like
        Specifies the base coordinate system LT quadrant for each data vector
    base_naz_angle : float or array-like
        North azimuth angle for the base coordinate system pole in degrees
    pole_angle: float or array-like
        Angle between the base coordinate system pole, data location, and the
        destination coordinate sytem pole (as located in the base coordinate
        system) in degrees

    Returns
    -------
    dest_naz_angle : float or array-like
        North azimuth angle for the destination coordinate system pole in
        degrees

    Raises
    ------
    ValueError
        If the input quadrant data is undefined

    """
    quad_range = np.arange(1, 5)

    # Test input
    if not np.isin(pole_quad, quad_range).any():
        raise ValueError("destination coordinate pole quadrant is undefined")

    if not np.isin(vect_quad, quad_range).any():
        raise ValueError("data vector quadrant is undefined")

    # Cast the necessary variables
    base_naz_angle = np.asarray(base_naz_angle)
    pole_angle = np.asarray(pole_angle)

    # Initialise the quadrant dictionary
    nan_mask = ~np.isnan(base_naz_angle) & ~np.isnan(pole_angle)
    quads = {oquad: {vquad:
                     (pole_quad == oquad) & (vect_quad == vquad) & nan_mask
                     for vquad in quad_range} for oquad in quad_range}

    # Create masks for the different quadrant combinations
    abs_mask = quads[1][1] | quads[2][2] | quads[3][3] | quads[4][4]
    add_mask = (quads[1][2] | quads[1][3] | quads[2][1] | quads[2][4]
                | quads[3][1] | quads[4][2])
    mpa_mask = quads[1][4] | quads[2][3]
    maa_mask = quads[3][2] | quads[4][1]
    cir_mask = quads[3][4] | quads[4][3]

    # Initialise the output
    dest_naz_angle = np.full(shape=(base_naz_angle + pole_angle
                                    + abs_mask).shape, fill_value=np.nan)

    # Calculate OCB polar angle based on the quadrants and other angles
    if np.any(abs_mask):
        if len(dest_naz_angle.shape) == 0:
            dest_naz_angle = abs(base_naz_angle - pole_angle)
        elif len(nan_mask.shape) == 0:
            dest_naz_angle[abs_mask] = abs(base_naz_angle - pole_angle)
        else:
            dest_naz_angle[abs_mask] = abs(base_naz_angle
                                           - pole_angle)[abs_mask]

    if np.any(add_mask):
        if len(dest_naz_angle.shape) == 0:
            dest_naz_angle = pole_angle + base_naz_angle
            if dest_naz_angle > 180.0:
                dest_naz_angle = 360.0 - dest_naz_angle
        elif len(nan_mask.shape) == 0:
            add_val = pole_angle + base_naz_angle
            dest_naz_angle[add_mask] = add_val
            if add_val > 180.0:
                dest_naz_angle[add_mask] = 360.0 - add_val
        else:
            dest_naz_angle[add_mask] = (pole_angle + base_naz_angle)[add_mask]
            lmask = (dest_naz_angle > 180.0) & add_mask
            if np.any(lmask):
                dest_naz_angle[lmask] = 360.0 - dest_naz_angle[lmask]

    if np.any(mpa_mask):
        if len(dest_naz_angle.shape) == 0:
            dest_naz_angle = base_naz_angle - pole_angle
        elif len(nan_mask.shape) == 0:
            dest_naz_angle[mpa_mask] = (base_naz_angle - pole_angle)
        else:
            dest_naz_angle[mpa_mask] = (base_naz_angle - pole_angle)[mpa_mask]

    if np.any(maa_mask):
        if len(dest_naz_angle.shape) == 0:
            dest_naz_angle = pole_angle - base_naz_angle
        elif len(nan_mask.shape) == 0:
            dest_naz_angle[maa_mask] = (pole_angle - base_naz_angle)
        else:
            dest_naz_angle[maa_mask] = (pole_angle - base_naz_angle)[maa_mask]

    if np.any(cir_mask):
        if len(dest_naz_angle.shape) == 0:
            dest_naz_angle = 360.0 - base_naz_angle - pole_angle
        elif len(nan_mask.shape) == 0:
            dest_naz_angle[cir_mask] = 360.0 - base_naz_angle - pole_angle
        else:
            dest_naz_angle[cir_mask] = (360.0 - base_naz_angle
                                        - pole_angle)[cir_mask]

    return dest_naz_angle


def calc_dest_vec_sign(pole_quad, vect_quad, base_naz_angle, pole_angle,
                       north=False, east=False, quads=None):
    """Calculate the sign of the North and East components.

    Parameters
    ----------
    pole_quad : float or array-like
        Specifies the base coordinate system LT quadrant for each pole/data pair
    vect_quad : float or array-like
        Specifies the base coordinate system LT quadrant for each data vector
    base_naz_angle : float or array-like
        North azimuth angle for the base coordinate system pole in degrees
    pole_angle: float or array-like
        Angle between the base coordinate system pole, data location, and the
        destination coordinate sytem pole (as located in the base coordinate
        system) in degrees
    north : bool
        Get the sign of the north component(s) (default=False)
    east : bool
        Get the sign of the east component(s) (default=False)
    quads : dict or NoneType
        Dictionary of boolean values or arrays of boolean values for the
        destination coordinate system pole and data vector quadrants
        (default=None)

    Returns
    -------
    vsigns : dict
        Dictionary with keys 'north' and 'east' containing the desired signs

    Raises
    ------
    ValueError
        If the input quadrant data is undefined

    """
    quad_range = np.arange(1, 5)

    # Cast the input
    base_naz_angle = np.asarray(base_naz_angle)
    pole_angle = np.asarray(pole_angle)

    # Test input
    if not np.any(np.isin(pole_quad, quad_range)):
        raise ValueError("destination coordinate pole quadrant is undefined")

    if not np.any(np.isin(vect_quad, quad_range)):
        raise ValueError("data vector quadrant is undefined")

    # If necessary, initialise quadrant dictionary
    nan_mask = ~np.isnan(base_naz_angle) & ~np.isnan(pole_angle)
    if quads is None or not np.all([kk in quads.keys() for kk in quad_range]):
        quads = {o: {v: (pole_quad == o) & (vect_quad == v)
                     & nan_mask for v in quad_range} for o in quad_range}

    # Initialise output
    vsigns = {"north": np.zeros(shape=quads[1][1].shape),
              "east": np.zeros(shape=quads[1][1].shape)}

    # Determine the desired vector signs
    if north:
        pole_minus = pole_angle - 90.0
        minus_pole = 90.0 - pole_angle
        pole_plus = pole_angle + 90.0

        pmask = (quads[1][1] | quads[2][2] | quads[3][3] | quads[4][4]
                 | ((quads[1][4] | quads[2][3]) & np.less_equal(
                     base_naz_angle, pole_plus, where=nan_mask))
                 | ((quads[1][2] | quads[2][1]) & np.less_equal(
                     base_naz_angle, minus_pole, where=nan_mask))
                 | ((quads[3][4] | quads[4][3]) & np.greater_equal(
                     base_naz_angle, 180.0 - pole_minus, where=nan_mask))
                 | ((quads[3][2] | quads[4][1]) & np.greater_equal(
                     base_naz_angle, pole_minus, where=nan_mask)))

        if np.any(pmask):
            if len(vsigns["north"].shape) == 0:
                vsigns["north"] = 1
            else:
                vsigns["north"][pmask] = 1

        if np.any(~pmask):
            if len(vsigns["north"].shape) == 0:
                vsigns["north"] = -1
            else:
                vsigns["north"][~pmask] = -1

    if east:
        minus_pole = 180.0 - pole_angle

        pmask = (quads[1][4] | quads[2][1] | quads[3][2] | quads[4][3]
                 | ((quads[1][1] | quads[4][4]) & np.greater_equal(
                     base_naz_angle, pole_angle, where=nan_mask))
                 | ((quads[3][1] | quads[2][4]) & np.less_equal(
                     base_naz_angle, minus_pole, where=nan_mask))
                 | ((quads[4][2] | quads[1][3]) & np.greater_equal(
                     base_naz_angle, minus_pole, where=nan_mask))
                 | ((quads[2][2] | quads[3][3]) & np.less_equal(
                     base_naz_angle, pole_angle, where=nan_mask)))

        if np.any(pmask):
            if len(vsigns["east"].shape) == 0:
                vsigns["east"] = 1
            else:
                vsigns["east"][pmask] = 1

        if np.any(~pmask):
            if len(vsigns["east"].shape) == 0:
                vsigns["east"] = -1
            else:
                vsigns["east"][~pmask] = -1

    return vsigns


def adjust_vector(vect_lt, vect_lat, vect_n, vect_e, vect_z, vect_quad,
                  pole_lt, pole_lat, pole_angle, pole_quad):
    """Adjust a vector from one coordinate system to another.

    Parameters
    ----------
    vect_lt : float or array-like
        Vector local time in base coordinate system in hours
    vect_lat : float or array-like
        Vector latitude in base coordinate system in degrees
    vect_n : float or array-like
        Vector North component in base coordinate system in degrees latitude
    vect_e : float or array-like
        Vector East component in base coordinate system in degrees latitude
    vect_z : float or array-like
        Vector vertical component in base coordinate system in degrees latitude
    vect_quad : float or array-like
        Specifies the base coordinate system LT quadrant for each data vector
    pole_lt : float or array-like
        Local time of destination pole location in the base coordinate system
        in hours
    pole_lat : float or array-like
        Latitude of destination pole location in the base coordinate system in
        degrees
    pole_angle : float or array-like
        Angle between the base coordinate system pole, data location, and the
        destination coordinate sytem pole (as located in the base coordinate
        system) in degrees
    pole_quad : float or array-like
        Specifies the base coordinate system LT quadrant for each pole/data pair

    Returns
    -------
    dest_n : float or array-like
        Vector North component in destination coordinate system in degrees
        latitude
    dest_e : float or array-like
        Vector East component in destination coordinate system in degrees
        latitude
    dest_z : float or array-like
        Vector vertical component in destination coordinate system in degrees
        latitude

    """
    # Ensure the input is array-like
    vect_lt = np.asarray(vect_lt)
    vect_lat = np.asarray(vect_lat)
    vect_n = np.asarray(vect_n)
    vect_e = np.asarray(vect_e)
    vect_z = np.asarray(vect_z)
    vect_quad = np.asarray(vect_quad)
    pole_lt = np.asarray(pole_lt)
    pole_lat = np.asarray(pole_lat)
    pole_angle = np.asarray(pole_angle)
    pole_quad = np.asarray(pole_quad)

    # Initialize the output, ensuring it is the same shape
    out_shape = (vect_lt + vect_lat + vect_n + vect_e + vect_z + vect_quad
                 + pole_lt + pole_lat + pole_angle + pole_quad).shape
    dest_n = np.full(shape=out_shape, fill_value=np.nan)
    dest_e = np.full(shape=out_shape, fill_value=np.nan)
    dest_z = np.full(shape=out_shape, fill_value=np.nan)

    # Determine the special case assignments
    zero_mask = (vect_n == 0.0) & (vect_e == 0.0)
    ns_mask = (pole_angle == 0.0) | (pole_angle == 180.0)
    norm_mask = ~(zero_mask + ns_mask)

    # There's nothing to adjust if there is no magnitude
    if np.any(zero_mask):
        if len(out_shape) == 0:
            dest_n = 0.0
            dest_e = 0.0
            dest_z = 0.0
        else:
            dest_n[zero_mask] = 0.0
            dest_e[zero_mask] = 0.0
            dest_z[zero_mask] = 0.0

    # The measurement is aligned with the base and destination poles
    if np.any(ns_mask):
        if len(out_shape) == 0:
            dest_n = float(vect_n)
            dest_e = float(vect_e)
            dest_z = float(vect_z)
        else:
            if len(vect_n.shape) == 0:
                dest_n[ns_mask] = vect_n
            else:
                dest_n[ns_mask] = vect_n[ns_mask]

            if len(vect_e.shape) == 0:
                dest_e[ns_mask] = vect_e
            else:
                dest_e[ns_mask] = vect_e[ns_mask]

            if len(vect_z.shape) == 0:
                dest_z[ns_mask] = vect_z
            else:
                dest_z[ns_mask] = vect_z[ns_mask]

        # Determine if the measurement is on or between the poles. This does
        # not affect the vertical direction
        sign_mask = (pole_angle == 0.0) & np.greater_equal(
            vect_lat, pole_lat, where=~np.isnan(vect_lat)) & ~np.isnan(vect_lat)

        if np.any(sign_mask):
            if len(out_shape) == 0:
                dest_n *= -1.0
                dest_e *= -1.0
            else:
                dest_n[sign_mask] *= -1.0
                dest_e[sign_mask] *= -1.0

    # If there are still undefined vectors, assign them using the typical case
    if np.any(norm_mask):
        # If not defined, get the pole and vector quadrants
        if len(vect_quad.shape) == 0 and vect_quad == 0 or (
                len(vect_quad.shape) > 0 and np.any(vect_quad[norm_mask] == 0)):
            vect_quad = define_vect_quadrants(vect_n, vect_e)

        if (len(pole_quad.shape) == 0 and pole_quad == 0) or (
                len(pole_quad.shape) > 0 and np.any(pole_quad[norm_mask] == 0)):
            pole_quad = define_pole_quadrants(vect_lt, pole_lt, pole_angle)

        # Get the unscaled 2D vector magnitude and calculate the AACGM north
        # azimuth in degrees
        if len(vect_n.shape) == 0 and len(vect_e.shape) == 0:
            vmag = np.sqrt(vect_n**2 + vect_e**2)
            base_naz_angle = np.degrees(np.arccos(vect_n / vmag))
        else:
            base_naz_angle = np.full(shape=norm_mask.shape, fill_value=np.nan)

            if len(vect_n.shape) == 0:
                vmag = np.sqrt(vect_n**2 + vect_e[norm_mask]**2)
                base_naz_angle[norm_mask] = np.degrees(np.arccos(vect_n / vmag))
            else:
                if len(vect_e.shape) == 0:
                    vmag = np.sqrt(vect_n[norm_mask]**2 + vect_e**2)
                else:
                    vmag = np.sqrt(vect_n[norm_mask]**2 + vect_e[norm_mask]**2)
                base_naz_angle[norm_mask] = np.degrees(
                    np.arccos(vect_n[norm_mask] / vmag))

        # Get the destination coordinate system north azimuth in radians
        dest_naz_angle = np.radians(calc_dest_polar_angle(
            pole_quad, vect_quad, base_naz_angle, pole_angle))

        # Get the sign of the North and East components
        vsigns = calc_dest_vec_sign(pole_quad, vect_quad, base_naz_angle,
                                    pole_angle, north=True, east=True)

        # Scale the vector along the OCB north
        if len(vect_z.shape) == 0:
            vz = vect_z
        else:
            vz = vect_z[norm_mask]
            nan_mask = np.isnan(vmag) | (
                np.isnan(dest_naz_angle) if len(dest_naz_angle.shape) == 0
                else np.isnan(dest_naz_angle[norm_mask]))
            vz[nan_mask] = np.nan

        # Restrict the OCB angle to result in positive sines and cosines
        lmask = dest_naz_angle > np.pi / 2.0
        if np.any(lmask):
            if len(dest_naz_angle.shape) == 0:
                dest_naz_angle = np.pi - dest_naz_angle
            else:
                dest_naz_angle[lmask] = np.pi - dest_naz_angle[lmask]

        # Calculate the vector components
        if len(vmag.shape) == 0:
            if len(dest_naz_angle.shape) == 0:
                dest_n = np.full(shape=out_shape, fill_value=(
                    vsigns['north'] * vmag * np.cos(dest_naz_angle)))
                dest_e = np.full(shape=out_shape, fill_value=(
                    vsigns['east'] * vmag * np.sin(dest_naz_angle)))
                dest_z = np.full(shape=out_shape, fill_value=vz)
            else:
                nval = vsigns['north'][norm_mask] * vmag * np.cos(
                    dest_naz_angle)[norm_mask]
                dest_n = np.full(shape=nval.shape, fill_value=nval)
                dest_e = np.full(shape=nval.shape, fill_value=(
                    vsigns['east'][norm_mask] * vmag * np.sin(
                        dest_naz_angle)[norm_mask]))
                dest_z = np.full(shape=nval.shape, fill_value=vz)
        else:
            dest_n[norm_mask] = vsigns['north'][norm_mask] * vmag * np.cos(
                dest_naz_angle)[norm_mask]
            dest_e[norm_mask] = vsigns['east'][norm_mask] * vmag * np.sin(
                dest_naz_angle)[norm_mask]
            dest_z[norm_mask] = vz

    return dest_n, dest_e, dest_z
