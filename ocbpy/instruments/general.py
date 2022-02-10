#!/usr/bin/env python
# Copyright (C) 2017 AGB
# Full license can be found in LICENSE.txt
# ---------------------------------------------------------------------------
""" General loading routines for data files
"""

import numpy as np
from os import path

import ocbpy
import ocbpy.ocb_time as ocbt


def test_file(filename):
    """Test to ensure the file is small enough to read in

    Parameters
    ----------
    filename : str
        Filename to test

    Returns
    -------
    good_flag : bool
        True if good, bad if false

    Notes
    -----
    Python can only allocate 2GB of data without crashing

    """

    if not path.isfile(filename):
        ocbpy.logger.warning("name provided is not a file")
        return False

    fsize = path.getsize(filename)

    if(fsize > 2.0e9):
        ocbpy.logger.warning(
            "File size [{:.2f} GB > 2 GB]".format(fsize * 1e-9))
        return False
    elif(fsize == 0):
        ocbpy.logger.warning("empty file [{:s}]".format(filename))
        return False

    return True


def load_ascii_data(filename, hlines, gft_kwargs=dict(), hsplit=None,
                    datetime_cols=None, datetime_fmt=None, int_cols=None,
                    str_cols=None, max_str_length=50, header=None):
    """ Load an ascii data file into a dict of numpy array

    Parameters
    ----------
    filename : str
        data file name
    hlines : int
        number of lines in header.  If zero, must include header.
    gft_kwargs : dict
        Dictionary holding optional keyword arguments for the numpy genfromtxt
        routine (default=dict())
    hsplit : str, NoneType
        character seperating data labels in header.  None splits on all
        whitespace characters. (default=None)
    datetime_cols : list, NoneType
        If there are date strings or values that should be converted to a
        datetime object, list them in order here. Not processed as floats.
        NoneType produces an empty list. (default=None)
    datetime_fmt : str, NoneType
        Format needed to convert the datetime_cols entries into a datetime
        object.  Special formats permitted are: 'YEAR SOY', 'SOD'.
        'YEAR SOY' must be used together; 'SOD' indicates seconds of day, and
        may be used with any date format (default=None)
    int_cols : list, NoneType
        Data that should be processed as integers, not floats. NoneType
        produces an empty list. (default=None)
    str_cols : list, NoneType
        Data that should be processed as strings, not floats. NoneType produces
        an empty list. (default=None)
    max_str_length : int
        Maximum allowed string length. (default=50)
    header : list, NoneType
        Header string(s) where the last line contains whitespace separated data
        names. NoneType produces an empty list. (default=None)

    Returns
    -------
    header : list of strings
        Contains all specified header lines
    out : dict of numpy.arrays
        The dict keys are specified by the header data line, the data
        for each key are stored in the numpy array

    Notes
    -----
    Data is assumed to be float unless otherwise stated.

    """
    # Initialize the empty lists
    if datetime_cols is None:
        datetime_cols = list()

    if int_cols is None:
        int_cols = list()

    if str_cols is None:
        str_cols = list()

    if header is None:
        header = list()

    # Test to ensure the file is small enough to read in.  Python can only
    # allocate 2GB of data.  If you load something larger, python will crash
    if not test_file(filename):
        return header, dict()

    # Initialize the convert_time input dictionary
    dfmt_parts = list() if datetime_fmt is None else datetime_fmt.split(" ")
    time_formats = ["H", "I", "p", "M", "S", "f", "z", "Z"]

    # Make sure the max_str_length is long enough to read datetime and that
    # the time data will be cast in the correct format
    if datetime_fmt is not None:
        dt_str_len = ocbt.get_datetime_fmt_len(datetime_fmt)
        if max_str_length < dt_str_len:
            max_str_length = dt_str_len

        if datetime_fmt.upper().find("YEAR") >= 0:
            ipart = datetime_fmt.upper().find("YEAR")
            case_part = datetime_fmt[ipart:ipart + 4]
            int_cols.append(dfmt_parts.index(case_part))
        if datetime_fmt.upper().find("SOY") >= 0:
            ipart = datetime_fmt.upper().find("SOY")
            case_part = datetime_fmt[ipart:ipart + 3]
            int_cols.append(dfmt_parts.index(case_part))

    # Open the data file and read the header rows
    with open(filename, "r") as fin:
        in_header = str(header[-1]) if len(header) > 0 else None

        for hind in range(hlines):
            header.append(fin.readline().strip())

    # Create the output dictionary keylist
    if len(header) == 0:
        estr = "unable to find header of [{:d}] lines".format(hlines)
        ocbpy.logger.error(estr)
        return header, dict()

    keyheader = in_header if in_header is not None else header[-1]

    if 'comments' in gft_kwargs.keys() and gft_kwargs['comments'] is not None:
        keyheader = keyheader.split(gft_kwargs['comments'])[0]

    keyheader = keyheader.replace("#", "").strip()
    keylist = [okey for okey in keyheader.split(hsplit) if len(okey) > 0]
    nhead = len(keylist)
    out = {okey: list() for okey in keylist}

    # Build the dtype list
    ldtype = [float for i in range(nhead)]

    for icol in int_cols:
        ldtype[icol] = int

    for icol in str_cols:
        ldtype[icol] = '|U{:d}'.format(max_str_length)

    # Build and add the datetime objects to the output dictionary
    dt_keys = ['datetime', 'DATETIME', 'DT', 'dt']
    if len(datetime_cols) > 0 and datetime_fmt is not None:
        idt = 0
        while dt_keys[idt] in out.keys():
            idt += 1

        if idt < len(dt_keys):
            keylist.append(dt_keys[idt])
            out[dt_keys[idt]] = list()

        # Change the datetime column input from float to string, if it is not
        # supposed to be an integer
        for i, icol in enumerate(datetime_cols):
            if(icol not in int_cols
               and dfmt_parts[i].upper().find("SOD") < 0):
                ldtype[icol] = '|U{:d}'.format(max_str_length)
    else:
        idt = len(dt_keys)

    # Open the datafile and read the data rows
    temp = np.genfromtxt(filename, skip_header=hlines, dtype=str, **gft_kwargs)

    if len(temp) > 0:
        # When dtype is specified, output comes as a void np.array
        #
        # Moved type specification for numpy 1.19.0, which throws a TypeError.
        # Also accounted for possibility of line variable being a scalar (but
        # not when calculating a time value)
        for iline, line in enumerate(temp):
            # Each line may have times that need to be combined and converted
            convert_time_input = {"year": None, "soy": None, "yyddd": None,
                                  "date": None, "tod": None,
                                  "datetime_fmt": datetime_fmt}

            # Cycle through each of the columns in this data row
            for num, name in enumerate(keylist):
                if idt < len(dt_keys) and name == dt_keys[idt]:
                    # Build the convert_time input
                    for icol, dcol in enumerate(datetime_cols):
                        line_val = line[dcol].astype(ldtype[dcol])

                        if dfmt_parts[icol].find("%") == 0:
                            if dfmt_parts[icol][1] in time_formats:
                                ckey = "tod"
                            else:
                                ckey = "date"
                        else:
                            ckey = dfmt_parts[icol].lower()
                            if ckey in ['year', 'soy']:
                                line_val = int(line_val)
                            elif ckey == 'sod':
                                line_val = float(line_val)

                        if ckey not in convert_time_input.keys():
                            convert_time_input[ckey] = line_val
                        else:
                            if convert_time_input[ckey] is None:
                                convert_time_input[ckey] = line_val
                            else:
                                convert_time_input[ckey] = " ".join([
                                    convert_time_input[ckey], line_val])

                    # Convert the string into a datetime object
                    ftime = ocbt.convert_time(**convert_time_input)

                    # Save the output data
                    out[dt_keys[idt]].append(ftime)
                else:
                    # Save the output data without any manipulation
                    try:
                        out[name].append(line[num].astype(ldtype[num]))
                    except AttributeError:
                        out[name].append(line.astype(ldtype[num]))

    # Cast all lists as numpy arrays, if possible
    for k in out.keys():
        try:
            out[k] = np.array(out[k], dtype=type(out[k][0]))
        except TypeError:
            # Leave as a list if array casting doesn't work.  This was an
            # issue before, but may have been an old numpy bug that is fixed.
            pass

    return header, out


def satellite_track(lat, mlt, x1, y1, x2, y2, hemisphere, del_x=1.0, del_y=1.0,
                    past_bound=5.0):
    """Determine whether or not a point lies along the satellite track.

    Parameters
    ----------
    lat : array-like
        AACGM latitude in degrees
    mlt : array-like
        AACGM local time in hours
    x1 : float
        Cartesian x-coordinate of the first boundary location in AACGM degrees
        along the Dawn-Dusk axis
    y1 : float
        Cartesian y-coordinate of the first boundary location in AACGM degrees
        along the Noon-Midnight axis
    x2 : float
        Cartesian x-coordinate of the second boundary location in AACGM degrees
        along the Dawn-Dusk axis
    y2 : float
        Cartesian y-coordinate of the second boundary location in AACGM degrees
        along the Noon-Midnight axis
    hemisphere : int
        Integer (+/- 1) denoting northern/southern hemisphere
    del_x : float
        Allowable distance from the track in AACGM degrees along the x-axis
        (default=1.0)
    del_y : float
        Allowable distance from the track in AACGM degrees along the y-axis
        (default=1.0)
    past_bound : float
        Allowable distance equatorward from the boundary in AACGM degrees
        (default=5.0)

    Returns
    -------
    good : array-like
         Array of booleans that are True if location is along the track and
         False if the location falls outside of the track

    Raises
    ------
    ValueError
        If the boundary values are negative or if an unknown hemisphere is
        specified

    """
    # Evaluate the boundary input
    if del_x < 0.0 or del_y < 0.0:
        raise ValueError("x- and y-axis allowable difference must be positive")

    if past_bound < 0.0:
        raise ValueError("equatorward buffer for track must be positive")

    if hemisphere not in [1.0, -1.0]:
        raise ValueError("hemisphere expecting +/- 1")

    # Ensure the input is array-like
    lat = np.asarray(lat)
    mlt = np.asarray(mlt)

    # Get the equation of the line defining the upper and lower bounds
    slope = (y1 - y2) / (x1 - x2)
    high_int = y1 + del_y - slope * (x1 + del_x)
    low_int = y1 - del_y - slope * (x1 - del_x)

    # Determine the Cartesian coordinates of the input point
    rad_in = ocbpy.ocb_time.hr2rad(mlt - 6.0)
    x_in = (90.0 - hemisphere * lat) * np.cos(rad_in)
    y_in = (90.0 - hemisphere * lat) * np.sin(rad_in)

    # Calculate the upper and lower limits for each of the x-inputs
    y_low = slope * x_in + low_int
    y_high = slope * x_in + high_int

    good = (y_in >= y_low) & (y_in <= y_high)

    # Ensure the latitude is not too low
    if np.any(good):
        r1 = np.sqrt((x1 - x_in)**2 + (y1 - y_in)**2)
        r2 = np.sqrt((x2 - x_in)**2 + (y2 - y_in)**2)

        # Find which boundary is closest to the good points
        ione = np.where(good & (r1 <= r2))
        itwo = np.where(good & (r2 < r1))

        # Evaluate the closest lower boundary limit
        if len(lat[ione]) > 0:
            lat_bound = 90.0 - np.sqrt(x1**2 + y1**2) - past_bound
            ipast = np.where(abs(lat[ione]) < lat_bound)
            good[ione[0][ipast]] = False

        if len(lat[itwo]) > 0:
            lat_bound = 90.0 - np.sqrt(x2**2 + y2**2) - past_bound
            ipast = np.where(abs(lat[itwo]) < lat_bound)
            good[itwo[0][ipast]] = False

    # Return the boolean mask
    return good

