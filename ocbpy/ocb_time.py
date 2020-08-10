#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Routines to convert from different file timekeeping methods to datetime

Functions
---------
get_datetime_fmt_len(datetime_fmt)
    Gets the length of a string line needed to hold a specified datetime format
year_soy_to_datetime(yyyy, soy)
    Converts from seconds of year to datetime
yyddd_to_date(yyddd)
    Converts from years since 1900 and day of year to datetime
convert_time(kwargs)
    Convert to datetime from multiple time formats
deg2hr(lon)
    Convert from degrees to hours
hr2deg(lt)
    Convert from hours to degrees
rad2hr(lon)
    Convert from radians to hours
hr2rad(lt)
    Convert from hours to radians
datetime2hr(dtime)
    Calculate fractional hours of day from timestamp
slt2glon(slt, dtime)
    Convert from solar local time to geographic longitude
glon2slt(glon, dtime)
    Convert from geographic longitude to solar local time
fix_range(values, max_val, min_val)
    Ensure cyclic values lie within a specified range

Moduleauthor
------------
Angeline G. Burrell (AGB), 15 April 2017, University of Texas, Dallas

"""

import datetime as dt
import numpy as np


def get_datetime_fmt_len(datetime_fmt):
    """ Get the lenght of a string line needed for a specific datetime format

    Parameters
    ----------
    datetime_fmt : (str)
        Formatting string used to convert between datetime and string object

    Returns
    -------
    str_len : (int)
        Minimum length of a string needed to hold the specified data

    Notes
    -----
    See the datetime documentation for meanings of the datetime directives

    """

    # Start by setting the base length.  This accounts for any non-datetime
    # directives in the string length.
    str_len = len(datetime_fmt)

    # Each of the directives have character lengths that they fill.  Add the
    # appropriate number of spaces.
    add_len = {'%a': 1, '%A': 10, '%b': 1, '%B': 8, '%Y': 2, '%f': 4, '%z': 3,
               '%Z': 1, '%j': 1, '%c': 22, '%x': 8, '%X': 7}

    for dt_dir in add_len.keys():
        if datetime_fmt.find(dt_dir) >= 0:
            str_len += add_len[dt_dir]

    return str_len


def year_soy_to_datetime(yyyy, soy):
    """Converts year and soy to datetime

    Parameters
    ----------
    yyyy : (int)
        4 digit year
    soy : (float)
        seconds of year

    Returns
    -------
    dtime : (dt.datetime)
        datetime object

    """

    # Calcuate doy, hour, min, seconds of day
    ss = soy / 86400.0
    ddd = np.floor(ss)

    ss = (soy - ddd * 86400.0) / 3600.0
    hh = np.floor(ss)

    ss = (soy - ddd * 86400.0 - hh * 3600.0) / 60.0
    mm = np.floor(ss)

    ss = soy - ddd * 86400.0 - hh * 3600.0 - mm * 60.0

    # Define format
    stime = "{:d}-{:.0f}-{:.0f}-{:.0f}-{:.0f}".format(yyyy, ddd + 1, hh, mm,
                                                      ss)

    # Convert to datetime
    dtime = dt.datetime.strptime(stime, "%Y-%j-%H-%M-%S")

    return dtime


def yyddd_to_date(yyddd):
    """ Convert from years since 1900 and day of year to datetime

    Parameters
    ----------
    yyddd : (str)
        String containing years since 1900 and day of year
        (e.g. 100126 = 2000-05-5).

    Returns
    -------
    dtime : (dt.datetime)
        Datetime object containing date information

    """
    if not isinstance(yyddd, str):
        raise ValueError("YYDDD must be a string")

    # Remove any decimal data
    yyddd = yyddd.split(".")[0]

    # Select the year
    year = int(yyddd[:-3]) + 1900

    # Format the datetime string
    dtime = dt.datetime.strptime("{:d} {:s}".format(year, yyddd[-3:]), "%Y %j")

    return dtime


def convert_time(year=None, soy=None, yyddd=None, sod=None, date=None,
                 tod=None, datetime_fmt="%Y-%m-%d %H:%M:%S"):
    """ Convert to datetime from multiple time formats

    Parameters
    ----------
    year : (int or NoneType)
        Year or None if not in year-soy format (default=None)
    soy : (int or NoneType)
        Seconds of year or None if not in year-soy format (default=None)
    yyddd : (str or NoneType)
        String containing years since 1900 and 3-digit day of year
        (default=None)
    sod : (int,float or NoneType)
        Seconds of day or None if the time of day is not in this format
        (default=None)
    date : (str or NoneType)
        String containing date information or None if not in date-time format
        (default=None)
    tod : (str or NoneType)
        String containing time of day information or None if not in date-time
        format (default=None)
    datetime_fmt : (str)
        String with the date-time or date format (default='%Y-%m-%d %H:%M:%S')

    Returns
    -------
    dtime : (datetime)
        Datetime object

    """

    try:
        if year is not None and soy is not None:
            dtime = year_soy_to_datetime(year, soy)
        else:
            if yyddd is not None:
                ddate = yyddd_to_date(yyddd)
                date = ddate.strftime("%Y-%m-%d")

                # Ensure that the datetime format contains current date format
                if datetime_fmt.find("%Y-%m-%d") < 0:
                    ifmt = datetime_fmt.upper().find("YYDDD")
                    if ifmt >= 0:
                        old_fmt = datetime_fmt[ifmt:ifmt+5]
                        datetime_fmt = datetime_fmt.replace(old_fmt,
                                                            "%Y-%m-%d")
                    else:
                        datetime_fmt = "%Y-%m-%d {:s}".format(datetime_fmt)
            if tod is None:
                str_time = "{:}".format(date)

                # Ensure that the datetime format does not contain time
                for time_fmt in [" %H:%M:%S", " SOD"]:
                    time_loc = datetime_fmt.upper().find(time_fmt)
                    if time_loc > 0:
                        datetime_fmt = datetime_fmt[:time_loc]
            else:
                str_time = "{:s} {:s}".format(date, tod)

            dtime = dt.datetime.strptime(str_time, datetime_fmt)

            if sod is not None:
                # Add the seconds of day to dtime
                microsec, sec = np.modf(sod)
                dtime += dt.timedelta(seconds=int(sec))

                if microsec > 0.0:
                    # Add the microseconds to dtime
                    microsec = np.ceil(microsec * 1.0e6)
                    dtime += dt.timedelta(microseconds=int(microsec))

    except ValueError as verr:
        if(len(verr.args) > 0 and
           verr.args[0].startswith('unconverted data remains: ')):
            vsplit = verr.args[0].split(" ")
            dtime = dt.datetime.strptime(str_time[:-(len(vsplit[-1]))],
                                         datetime_fmt)
        else:
            raise ValueError(verr)

    return dtime


def deg2hr(lon):
    """ Convert from degrees to hours

    Parameters
    ----------
    lon : (float or array-like)
        Longitude-like value in degrees

    Returns
    -------
    lt : (float or array-like)
        Local time-like value in hours

    """

    lon = np.asarray(lon)
    lt = lon / 15.0  # 12 hr/180 deg = 1/15 hr/deg

    return lt


def hr2deg(lt):
    """ Convert from degrees to hours

    Parameters
    ----------
    lt : (float or array-like)
        Local time-like value in hours

    Returns
    -------
    lon : (float or array-like)
        Longitude-like value in degrees

    """

    lt = np.asarray(lt)
    lon = lt * 15.0  # 180 deg/12 hr = 15 deg/hr

    return lon


def hr2rad(lt):
    """ Convert from hours to radians

    Parameters
    ----------
    lt : (float or array-like)
        Local time-like value in hours

    Returns
    -------
    lon : (float or array-like)
        Longitude-like value in radians

    """

    lt = np.asarray(lt)
    lon = lt * np.pi / 12.0

    return lon


def rad2hr(lon):
    """ Convert from radians to hours

    Parameters
    ----------
    lon : (float or array-like)
        Longitude-like value in radians

    Returns
    -------
    lt : (float or array-like)
        Local time-like value in hours

    """

    lon = np.asarray(lon)
    lt = lon * 12.0 / np.pi

    return lt


def datetime2hr(dtime):
    """ Calculate hours of day from datetime

    Parameters
    ----------
    dtime : (dt.datetime)
        Universal time as a timestamp

    Returns
    -------
    uth : (float)
        Hours of day, includes fractional hours

    """

    uth = dtime.hour + dtime.minute / 60.0 \
        + (dtime.second + dtime.microsecond * 1.0e-6) / 3600.0

    return uth


def slt2glon(slt, dtime):
    """ Convert from solar local time to geographic longitude

    Parameters
    ----------
    slt : (float or array-like)
        Solar local time in hours
    dtime : (dt.datetime)
        Universal time as a timestamp

    Returns
    -------
    glon : (float or array-like)
        Geographic longitude in degrees

    """

    # Calculate universal time of day in hours
    uth = datetime2hr(dtime)

    # Calculate the longitude in degrees
    slt = np.asarray(slt)
    glon = hr2deg(slt - uth)

    # Ensure the longitude is not at or above 360 or at or below -180
    glon = fix_range(glon, -180.0, 360.0, 360.0)

    return glon


def glon2slt(glon, dtime):
    """ Convert from geographic longitude to solar local time

    Parameters
    ----------
    glon : (float or array-like)
        Geographic longitude in degrees
    dtime : (dt.datetime)
        Universal time as a timestamp

    Returns
    -------
    slt : (float or array-like)
        Solar local time in hours

    """

    # Calculate the longitude in degrees
    slt = deg2hr(glon) + datetime2hr(dtime)

    # Ensure the local time is between 0 and 24 h
    slt = fix_range(slt, 0.0, 24.0)

    return slt


def fix_range(values, min_val, max_val, val_range=None):
    """ Ensure cyclic values lie below the maximum and at or above the mininum

    Parameters
    ----------
    values : (int, float, or array-like)
        Values to adjust
    min_val : (int or float)
        Maximum that values may not meet or exceed
    max_val : (int or float)
        Minimum that values may not lie below
    val_range : (int, float, or NoneType)
        Value range or None to calculate from min and max (default=None)

    Returns
    -------
    fixed_vals : (int, float, or array-like)
        Values adjusted to lie min_val <= fixed_vals < max_val

    """

    # Cast output as array-like
    fixed_vals = np.asarray(values)

    # Test input to ensure the maximum is greater than the minimum
    if min_val >= max_val:
        raise ValueError('Minimum is not less than the maximum')

    # Determine the allowable range
    if val_range is None:
        val_range = max_val - min_val

    # Test input to ensure the value range is greater than zero
    if val_range <= 0.0:
        raise ValueError('Value range must be greater than zero')

    # Fix the values, allowing for deviations that are multiples of the
    # value range.  Also propagate NaNs
    ibad = (np.greater_equal(fixed_vals, max_val, where=~np.isnan(fixed_vals))
            & ~np.isnan(fixed_vals))
    while np.any(ibad):
        fixed_vals[ibad] -= val_range
        ibad = (np.greater_equal(fixed_vals, max_val,
                                 where=~np.isnan(fixed_vals))
                & ~np.isnan(fixed_vals))

    ibad = (np.less(fixed_vals, min_val, where=~np.isnan(fixed_vals))
            & ~np.isnan(fixed_vals))
    while np.any(ibad):
        fixed_vals[ibad] += val_range
        ibad = (np.less(fixed_vals, min_val, where=~np.isnan(fixed_vals))
                & ~np.isnan(fixed_vals))

    return fixed_vals
