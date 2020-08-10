# -*- coding: utf-8 -*-
# Copyright (C) 2017 AGB
# Full license can be found in LICENSE.txt
# ---------------------------------------------------------------------------
""" Perform OCB gridding for SuperMAG data

Functions
---------
supermag2ascii_ocb(smagfile, outfile, kwargs)
     Write and ASCII file with SuperMAG data and the OCB coordinates for each
     data point
load_supermag_ascii_data(filename)
     Load SuperMAG ASCII data files

Data
----
SuperMAG data available at: http://supermag.jhuapl.edu/

"""

from __future__ import absolute_import, unicode_literals
import datetime as dt
import numpy as np

import ocbpy
import ocbpy.ocb_scaling as ocbscal


def supermag2ascii_ocb(smagfile, outfile, hemisphere=0, ocb=None,
                       ocbfile='default', instrument='', max_sdiff=600,
                       min_sectors=7, rcent_dev=8.0, max_r=23.0, min_r=10.0):
    """ Coverts and scales the SuperMAG data into OCB coordinates

    Parameters
    ----------
    smagfile : (str)
        file containing the required SuperMAG file sorted by time
    outfile : (str)
        filename for the output data
    hemisphere : (int)
        Hemisphere to process (can only do one at a time).  1=Northern,
        -1=Southern, 0=Determine from data (default=0)
    ocb : (OCBoundary or NoneType)
        OCBoundary object with data loaded from an OC boundary data file.
        If None, looks to ocbfile
    ocbfile : (str)
        file containing the required OC boundary data sorted by time, or
        'default' to load default file for time and hemisphere.  Only used if
        no OCBoundary object is supplied (default='default')
    instrument : (str)
        Instrument providing the OCBoundaries.  Requires 'image' or 'ampere'
        if a file is provided.  If using filename='default', also accepts
        'amp', 'si12', 'si13', 'wic', and ''.  (default='')
    max_sdiff : (int)
        maximum seconds between OCB and data record in sec (default=600)
    min_sectors : (int)
        Minimum number of MLT sectors required for good OCB (default=7).
    rcent_dev : (float)
        Maximum number of degrees between the new centre and the AACGM pole
        (default=8.0)
    max_r : (float)
        Maximum radius for open-closed field line boundary in degrees
        default=23.0)
    min_r : (float)
        Minimum radius for open-closed field line boundary in degrees
        (default=10.0)

    Notes
    -----
    May only process one hemisphere at a time.  Scales the magnetic field
    observations using `ocbpy.ocb_scale.normal_curl_evar`.

    """

    if not ocbpy.instruments.test_file(smagfile):
        raise IOError("SuperMAG file cannot be opened [{:s}]".format(smagfile))

    if not isinstance(outfile, str):
        raise IOError("output filename is not a string [{:}]".format(outfile))

    # Read the superMAG data and calculate the magnetic field magnitude
    header, mdata = load_supermag_ascii_data(smagfile)

    # Load the OCB data for the SuperMAG data period
    if ocb is None or not isinstance(ocb, ocbpy.ocboundary.OCBoundary):
        mstart = mdata['DATETIME'][0] - dt.timedelta(seconds=max_sdiff+1)
        mend = mdata['DATETIME'][-1] + dt.timedelta(seconds=max_sdiff+1)

        # If hemisphere isn't specified, set it here
        if hemisphere == 0:
            hemisphere = np.sign(np.nanmax(mdata['MLAT']))

            # Ensure that all data is in the same hemisphere
            if hemisphere == 0:
                hemisphere = np.sign(np.nanmin(mdata['MLAT']))
            elif hemisphere != np.sign(np.nanmin(mdata['MLAT'])):
                raise ValueError("".join(["cannot process observations from "
                                          "both hemispheres at the same time;"
                                          "set hemisphere=+/-1 to choose."]))

        # Initialize the OCBoundary object
        ocb = ocbpy.OCBoundary(ocbfile, stime=mstart, etime=mend,
                               hemisphere=hemisphere, instrument=instrument)

    elif hemisphere == 0:
        # If the OCBoundary object is specified and hemisphere isn't use
        # the OCBoundary object to specify the hemisphere
        hemisphere = ocb.hemisphere

    # Test the OCB data
    if ocb.filename is None or ocb.records == 0:
        ocbpy.logger.error("no data in OCB file {:}".format(ocb.filename))
        return

    # Remove the data with NaNs/Inf and from the opposite hemisphere/equator
    igood = np.where((np.isfinite(mdata['MLT'])) & (np.isfinite(mdata['MLAT']))
                     & (np.isfinite(mdata['BE'])) & (np.isfinite(mdata['BN']))
                     & (np.isfinite(mdata['BZ']))
                     & (np.sign(mdata['MLAT']) == hemisphere))[0]

    if igood.shape != mdata['MLT'].shape:
        for k in mdata.keys():
            mdata[k] = mdata[k][igood]

        # Recalculate the number of stations if some data was removed
        for tt in np.unique(mdata['DATETIME']):
            itimes = np.where(mdata['DATETIME'] == tt)[0]
            mdata['NST'][itimes] = len(itimes)

    # Open and test the file to ensure it can be written
    with open(outfile, 'w') as fout:
        # Write the output line
        outline = "#DATE TIME NST STID "
        optional_keys = ["SML", "SMU", "SZA"]
        for okey in optional_keys:
            if okey in mdata.keys():
                outline = "{:s}{:s} ".format(outline, okey)

        outline = "".join([outline, "MLAT MLT BMAG BN BE BZ OCB_MLAT OCB_MLT ",
                           "OCB_BMAG OCB_BN OCB_BE OCB_BZ\n"])
        fout.write(outline)

        # Initialise the ocb and SuperMAG indices
        imag = 0
        nmag = mdata['DATETIME'].shape[0]

        # Cycle through the data, matching SuperMAG and OCB records
        while imag < nmag and ocb.rec_ind < ocb.records:
            imag = ocbpy.match_data_ocb(ocb, mdata['DATETIME'], idat=imag,
                                        max_tol=max_sdiff,
                                        min_sectors=min_sectors,
                                        rcent_dev=rcent_dev, max_r=max_r,
                                        min_r=min_r)

            if imag < nmag and ocb.rec_ind < ocb.records:
                # Set this value's AACGM vector values
                vdata = ocbscal.VectorData(imag, ocb.rec_ind,
                                           mdata['MLAT'][imag],
                                           mdata['MLT'][imag],
                                           aacgm_n=mdata['BN'][imag],
                                           aacgm_e=mdata['BE'][imag],
                                           aacgm_z=mdata['BZ'][imag],
                                           scale_func=ocbscal.normal_curl_evar)

                vdata.set_ocb(ocb)

                # Format the output line:
                #    DATE TIME NST [SML SMU] STID [SZA] MLAT MLT BMAG BN BE BZ
                #    OCB_MLAT OCB_MLT OCB_BMAG OCB_BN OCB_BE OCB_BZ
                outline = "{:} {:d} {:s} ".format(mdata['DATETIME'][imag],
                                                  mdata['NST'][imag],
                                                  mdata['STID'][imag])

                for okey in optional_keys:
                    if okey == "SZA":
                        outline = "{:s}{:.2f} ".format(outline,
                                                       mdata[okey][imag])
                    else:
                        outline = "{:s}{:d} ".format(outline,
                                                     mdata[okey][imag])

                outline = "".join([outline, "{:.2f} ".format(vdata.aacgm_lat),
                                   "{:.2f} {:.2f} ".format(vdata.aacgm_mlt,
                                                           vdata.aacgm_mag),
                                   "{:.2f} {:.2f} ".format(vdata.aacgm_n,
                                                           vdata.aacgm_e),
                                   "{:.2f} {:.2f} {:.2f} {:.2f}".format(
                                       vdata.aacgm_z, vdata.ocb_lat,
                                       vdata.ocb_mlt, vdata.ocb_mag),
                                   " {:.2f} {:.2f} {:.2f}\n".format(
                                       vdata.ocb_n, vdata.ocb_e, vdata.ocb_z)])
                fout.write(outline)

                # Move to next line
                imag += 1

    return


def load_supermag_ascii_data(filename):
    """Load a SuperMAG ASCII data file

    Parameters
    ----------
    filename : (str)
        SuperMAG ASCI data file name

    Returns
    -------
    out : (dict of numpy.arrays)
        The dict keys are specified by the header data line, the data
        for each key are stored in the numpy array

    """

    fill_val = 999999
    header = list()
    ind = {"SMU": fill_val, "SML": fill_val}
    out = {"YEAR": list(), "MONTH": list(), "DAY": list(), "HOUR": list(),
           "MIN": list(), "SEC": list(), "DATETIME": list(), "NST": list(),
           "SML": list(), "SMU": list(), "STID": list(), "BN": list(),
           "BE": list(), "BZ": list(), "MLT": list(), "MLAT": list(),
           "DEC": list(), "SZA": list()}

    if not ocbpy.instruments.test_file(filename):
        return header, dict()

    # Open the datafile and read the data
    with open(filename, "r") as f:
        hflag = True
        n = -1
        for line in f.readlines():
            if hflag:
                # Fill the header list
                header.append(line)
                if line.find("=========================================") >= 0:
                    hflag = False
            else:
                # Fill the output dictionary
                if n < 0:
                    # This is a date line
                    n = 0
                    lsplit = np.array(line.split(), dtype=int)
                    dtime = dt.datetime(lsplit[0], lsplit[1], lsplit[2],
                                        lsplit[3], lsplit[4], lsplit[5])
                    snum = lsplit[-1]
                else:
                    lsplit = line.split()

                    if len(lsplit) == 2:
                        # This is an index line
                        ind[lsplit[0]] = int(lsplit[1])
                    else:
                        # This is a station data line
                        out['YEAR'].append(dtime.year)
                        out['MONTH'].append(dtime.month)
                        out['DAY'].append(dtime.day)
                        out['HOUR'].append(dtime.hour)
                        out['MIN'].append(dtime.minute)
                        out['SEC'].append(dtime.second)
                        out['DATETIME'].append(dtime)
                        out['NST'].append(snum)

                        for k in ind.keys():
                            out[k].append(ind[k])

                        out['STID'].append(lsplit[0])
                        out['BN'].append(float(lsplit[1]))
                        out['BE'].append(float(lsplit[2]))
                        out['BZ'].append(float(lsplit[3]))
                        out['MLT'].append(float(lsplit[4]))
                        out['MLAT'].append(float(lsplit[5]))
                        out['DEC'].append(float(lsplit[6]))
                        out['SZA'].append(float(lsplit[7]))

                        n += 1

                        if n == snum:
                            n = -1
                            ind = {"SMU": fill_val, "SML": fill_val}

    # Recast data as numpy arrays and replace fill value with np.nan
    for k in out:
        if k == "STID":
            out[k] = np.array(out[k], dtype=str)
        else:
            out[k] = np.array(out[k])

            if k in ['BE', 'BN', 'DEC', 'SZA', 'MLT', 'BZ']:
                out[k][out[k] == fill_val] = np.nan

    return header, out
