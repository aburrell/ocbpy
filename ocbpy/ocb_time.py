#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
"""Routines to convert from different file timekeeping methods to datetime

Functions
-------------------------------------------------------------------------------
year_soy_to_datetime(yyyy, soy)
    Converts from seconds of year to datetime
yyddd_to_date(yyddd)
    Converts from years since 1900 and day of year to datetime
convert_time(kwargs)
    Convert to datetime from multiple time formats

Moduleauthor
-------------------------------------------------------------------------------
Angeline G. Burrell (AGB), 15 April 2017, University of Texas, Dallas (UTDallas)
"""

import logbook as logging
import datetime as dt

def year_soy_to_datetime(yyyy, soy):
    """Converts year and soy to datetime

    Parameters
    -----------
    yyyy : (int)
        4 digit year
    soy : (float)
        seconds of year

    Returns
    ---------
    dtime : (dt.datetime)
        datetime object
    """
    import numpy as np
                
    # Calcuate doy, hour, min, seconds of day
    ss = soy / 86400.0
    ddd = np.floor(ss)

    ss = (soy - ddd * 86400.0) / 3600.0
    hh = np.floor(ss)

    ss = (soy - ddd * 86400.0 - hh * 3600.0) / 60.0
    mm = np.floor(ss)

    ss = soy - ddd * 86400.0 - hh * 3600.0 - mm * 60.0
    
    # Define format
    stime = "{:d}-{:.0f}-{:.0f}-{:.0f}-{:.0f}".format(yyyy, ddd + 1, hh, mm, ss)

    # Convert to datetime
    dtime = dt.datetime.strptime(stime, "%Y-%j-%H-%M-%S")

    return dtime

def yyddd_to_date(yyddd):
    """ Convert from years since 1900 and day of year to datetime

    Parameters
    -----------
    yyddd : (str)
        String containing years since 1900 and day of year
        (e.g. 100126 = 2000-05-5).

    Returns
    -------
    dtime : (dt.datetime)
        Datetime object containing date information
    """
    assert isinstance(yyddd, str), logging.error("YYDDD must be a string")
    
    # Remove any decimal data
    yyddd = yyddd.split(".")[0]

    # Select the year
    year = int(yyddd[:-3]) + 1900

    # Format the datetime string
    dtime = dt.datetime.strptime("{:d} {:s}".format(year, yyddd[-3:]), "%Y %j")

    return dtime

def convert_time(year=None, soy=None, yyddd=None, sod=None, date=None, tod=None,
                 datetime_fmt="%Y-%m-%d %H:%M:%S"):
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
        String with the date-time or date format.  (default='%Y-%m-%d %H:%M:%S')

    Returns
    --------
    dtime : (datetime)
        Datetime object
    """
    import numpy as np

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
                        datetime_fmt = datetime_fmt.replace(old_fmt, "%Y-%m-%d")
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
                
    except ValueError as v:
        if(len(v.args) > 0 and
           v.args[0].startswith('unconverted data remains: ')):
            vsplit = v.args[0].split(" ")
            dtime = dt.datetime.strptime(str_time[:-(len(vsplit[-1]))],
                                         datetime_fmt)
        else:
            raise v

    return dtime
