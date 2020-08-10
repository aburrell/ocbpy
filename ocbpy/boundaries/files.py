#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB & GC
# Full license can be found in License.md
# ----------------------------------------------------------------------------
""" Provide desired boundary file names

Functions
---------
get_boundary_dir()
    Get the OCBpy boundary directory
get_boundary_files()
    Return a dict with boundary filenames and their spatiotemporal ranges
get_default_file(stime, etime, hemisphere, [instrument])
    Return the name of the default boundary file

Moduleauthor
------------
Angeline G. Burrell (AGB), 25 September 2019, Naval Research Laboratory (NRL)

References
----------
Chisham, G. (2017), A new methodology for the development of high-latitude
 ionospheric climatologies and empirical models, Journal of Geophysical
 Research: Space Physics, 122, doi:10.1002/2016JA023235.
Milan, S. E., et al. (2015), Principal component analysis of
  Birkeland currents determined by the Active Magnetosphere and Planetary
  Electrodynamics Response Experiment, J. Geophys. Res. Space Physics, 120,
  10,415–10,424, doi:10.1002/2015JA021680.

"""
from __future__ import absolute_import, unicode_literals

import datetime as dt
import itertools
import os

import ocbpy


def get_boundary_directory():
    """ Get the OCBpy boundary directory

    Returns
    -------
    boundary_dir : (str)
        Directory holding the boundary files included in OCBpy

    """

    boundary_dir = os.path.join(os.path.dirname(ocbpy.__file__), "boundaries")

    if not os.path.isdir(boundary_dir):
        raise OSError("can't find the OCBpy boundary file directory")

    return boundary_dir


def get_boundary_files(bound='ocb'):
    """ Get boundary filenames and their spatiotemporal ranges

    Parameters
    ----------
    bound : (str)
        String specifying which boundary is desired (OCB or EAB)
        (default='ocb')

    Returns
    -------
    boundary_files : (dict)
        Dict with keys of boundary files containing dicts specifying the
        hemisphere, instrument, file start time, and file end time

    Notes
    -----
    IMAGE instruments are separated into WIC, SI12, and SI13
    Unknown instruments should have filenames of the format:
        instrument_hemisphere_%Y%m%d_%Y%m%d.boundary

    """
    hemi = {"north": 1, "south": -1}
    stime = {"amp": dt.datetime(2010, 1, 1),
             "si12": dt.datetime(2000, 5, 4),
             "si13": dt.datetime(2000, 5, 5),
             "wic": dt.datetime(2000, 5, 3),
             "dmsp-ssj": dt.datetime(2010, 1, 1)}
    etime = {"amp": dt.datetime(2017, 1, 1),
             "si12": dt.datetime(2002, 8, 23),
             "si13": dt.datetime(2002, 8, 23),
             "wic": dt.datetime(2002, 8, 22),
             "dmsp-ssj": dt.datetime.today().replace(hour=0, minute=0,
                                                     second=0, microsecond=0)
             + dt.timedelta(days=1)}

    # List all of the files in the OCBpy boundary directory
    boundary_dir = get_boundary_directory()
    file_list = os.listdir(boundary_dir)

    # Add the boundary files to the output dictionary
    boundary_files = dict()
    for bfile in file_list:
        if(os.path.isfile(os.path.join(boundary_dir, bfile))
           and bfile.find(".py") < 0 and bfile.find("README") < 0
           and bfile[-3:].lower() == bound.lower()):
            file_info = bfile.lower()[:-4].split("_")
            boundary_files[bfile] = {"instrument": file_info[0],
                                     "hemisphere": hemi[file_info[1]]}

            if file_info[0] in stime.keys():
                boundary_files[bfile]["stime"] = stime[file_info[0]]
                boundary_files[bfile]["etime"] = etime[file_info[0]]
            else:
                try:
                    boundary_files[bfile]["stime"] = dt.datetime.strptime(
                        file_info[2], '%Y%m%d')
                    boundary_files[bfile]["etime"] = dt.datetime.strptime(
                        file_info[3], '%Y%m%d')
                except ValueError as verr:
                    err = 'Unknown boundary file present: [{:s}]\n{:}'.format(
                        bfile, verr)
                    ocbpy.logger.warning(err)

    return boundary_files


def get_default_file(stime, etime, hemisphere, instrument='', bound='ocb'):
    """ Get the default file for a specified time and hemisphere

    Parameters
    ----------
    stime : (dt.datetime or NoneType)
        Starting time for which the file is desired; if None, will prioritize
        IMAGE data for the northern and AMERE data for the southern hemisphere
    etime : (dt.datetime or NoneType)
        Ending time for which the file is desired; if None, will prioritize
        IMAGE data for the northern and AMPERE data for the southern hemisphere
    hemisphere : (int)
        Hemisphere for which the file is desired (1=north, -1=south)
    instrument : (str)
        Instrument that provides the data.  This will override the starting
        and ending times.  Accepts 'ampere', 'amp', 'image', 'si12', 'si13',
        'wic', 'dmsp-ssj', and '' (to accept instrument defaults based on time
        range).  Will also accept the instrument name for any instrument whose
        boundary file follows the naming convention
        INST_HEMI_YYYYMMDD_YYYYMMDD_*.BBB, where:
            INST     = instrument name
            HEMI     = north or south
            YYYYMMDD = starting and ending year, month, day
            BBB      = ocb or eab
        (default='')
    bound : (str)
        String specifying which boundary is desired (OCB or EAB)
        (default='ocb')

    Returns
    -------
    default_file : (str or NoneType)
        Default filename with full path defined or None if no file was
        available for the specified input constraints
    instrument : (str)
        Instrument for the default file (either 'ampere', 'image',
        or 'dmsp-ssj')

    """

    # Get the boundary file information
    boundary_dir = get_boundary_directory()
    boundary_files = get_boundary_files(bound=bound)

    # Determine the list of acceptable instruments
    long_to_short = {"ampere": ["amp"], "image": ["si12", "si13", "wic"],
                     "dmsp-ssj": ["dmsp-ssj"]}
    if len(instrument) == 0:
        inst = list(itertools.chain.from_iterable(long_to_short.values()))
    elif instrument in long_to_short.keys():
        inst = long_to_short[instrument]
    else:
        inst = [instrument]

    # Make a list of appropriate boundary files
    good_files = list()
    for bfile, bdict in list(boundary_files.items()):
        # Select by hemisphere
        if bdict['hemisphere'] == hemisphere:
            # Select by instrument
            if bdict['instrument'] in inst:
                # Select by time
                if(stime is None or etime is None or
                   (stime <= bdict['etime'] and etime >= bdict['stime'])):
                    good_files.append(bfile)

    # Get the default file and instrument (returning at most one)
    short_to_long = {"amp": "ampere", "si12": "image", "si13": "image",
                     "wic": "image"}
    if len(good_files) == 0:
        estr = "".join(["no boundary file available for ", ", ".join(inst),
                        " northern" if hemisphere == 1 else " southern",
                        " hemisphere, {:} to {:}".format(stime, etime)])
        ocbpy.logger.info(estr)

        default_file = None
    elif len(good_files) == 1:
        default_file = os.path.join(boundary_dir, good_files[0])
        if boundary_files[good_files[0]]['instrument'] in short_to_long.keys():
            instrument = short_to_long[boundary_files[good_files[0]][
                'instrument']]
        else:
            instrument = boundary_files[good_files[0]]['instrument']
    else:
        # Rate files by instrument
        default_inst = ['si13', 'si12', 'wic', 'amp', 'dmsp-ssj']
        ordered_files = {default_inst.index(boundary_files[bb]['instrument']):
                         bb for bb in good_files}
        bfile = ordered_files[min(ordered_files.keys())]
        default_file = os.path.join(boundary_dir, bfile)
        if boundary_files[bfile]['instrument'] in short_to_long.keys():
            instrument = short_to_long[boundary_files[bfile]['instrument']]
        else:
            instrument = boundary_files[bfile]['instrument']

    return default_file, instrument
