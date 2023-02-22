# -*- coding: utf-8 -*-
# Copyright (C) 2017 AGB
# Full license can be found in LICENSE.txt
# ---------------------------------------------------------------------------
"""Perform OCB gridding for SuperMAG data.

Notes
-----
SuperMAG data available at: http://supermag.jhuapl.edu/

"""

import datetime as dt
import numpy as np
import warnings

import ocbpy
import ocbpy.ocb_scaling as ocbscal


def supermag2ascii_ocb(smagfile, outfile, hemisphere=0, ocb=None,
                       ocbfile='default', instrument='', max_sdiff=600,
                       min_merit=None, max_merit=None,
                       scale_func=ocbscal.normal_curl_evar, **kwargs):
    """Covert and scales the SuperMAG data into OCB coordinates.

    Parameters
    ----------
    smagfile : str
        File containing the required SuperMAG file sorted by time
    outfile : str
        Filename for the output data
    hemisphere : int
        Hemisphere to process (can only do one at a time).  1=Northern,
        -1=Southern, 0=Determine from data (default=0)
    ocb : ocbpy.OCBoundary, ocbpy.DualBoundary, or NoneType
        OCBoundary or DualBoundary object with data loaded already. If None,
        looks to `ocbfile` and creates an OCBoundary object. (default=None)
    ocbfile : str
        File containing the required OC Boundary data sorted by time, or
        'default' to load default file for time and hemisphere.  Only used if
        no OCBoundary object is supplied (default='default')
    instrument : str
        Instrument providing the OCBoundaries.  Requires 'image' or 'ampere'
        if a file is provided.  If using filename='default', also accepts
        'amp', 'si12', 'si13', 'wic', and ''.  (default='')
    max_sdiff : int
        Maximum seconds between OCB and data record in sec (default=60)
    min_merit : float or NoneType
        Minimum value for the default figure of merit or None to not apply a
        custom minimum (default=None)
    max_merit : float or NoneTye
        Maximum value for the default figure of merit or None to not apply a
        custom maximum (default=None)
    kwargs : dict
        Dict with optional selection criteria.  The key should correspond to a
        data attribute and the value must be a tuple with the first value
        specifying 'max', 'min', 'maxeq', 'mineq', or 'equal' and the second
        value specifying the value to use in the comparison.
    min_sectors : int
        Minimum number of MLT sectors required for good OCB. Deprecated, will
        be removed in version 0.3.1+ (default=7).
    rcent_dev : float
        Maximum number of degrees between the new centre and the AACGM pole.
        Deprecated, will be removed in version 0.3.1+ (default=8.0)
    max_r : float
        Maximum radius for open-closed field line boundary in degrees/
        Deprecated, will be removed in version 0.3.1+ (default=23.0)
    min_r : float
        Minimum radius for open-closed field line boundary in degrees.
        Deprecated, will be removed in version 0.3.1+ (default=10.0)
    scale_func : function or NoneType
        Scale the magnetic field observations unless None
        (default=ocbpy.ocb_scale.normal_curl_evar)

    Raises
    ------
    IOError
        If unable to open the input or output file

    Notes
    -----
    May only process one hemisphere at a time.

    See Also
    --------
    ocbpy.ocb_scale.normal_curl_evar

    """

    # Test inputs
    if not ocbpy.instruments.test_file(smagfile):
        raise IOError("SuperMAG file cannot be opened [{:s}]".format(smagfile))

    if not isinstance(outfile, str):
        raise IOError("output filename is not a string [{:}]".format(outfile))

    # Read the superMAG data and calculate the magnetic field magnitude
    header, mdata = load_supermag_ascii_data(smagfile)

    # Load the OCB data for the SuperMAG data period
    if ocb is None or (not isinstance(ocb, ocbpy.OCBoundary)
                       and not isinstance(ocb, ocbpy.DualBoundary)):
        mstart = mdata['DATETIME'][0] - dt.timedelta(seconds=max_sdiff + 1)
        mend = mdata['DATETIME'][-1] + dt.timedelta(seconds=max_sdiff + 1)

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
    if ocb.records == 0:
        ocbpy.logger.error("no data in the Boundary file(s)")
        return

    # Add check for deprecated and custom kwargs
    dep_comp = {'min_sectors': ['num_sectors', ('mineq', 7)],
                'rcent_dev': ['r_cent', ('maxeq', 8.0)],
                'max_r': ['r', ('maxeq', 23.0)],
                'min_r': ['r', ('mineq', 10.0)]}
    cust_keys = list(kwargs.keys())

    for ckey in cust_keys:
        if ckey in dep_comp.keys():
            warnings.warn("".join(["Deprecated kwarg will be removed in ",
                                   "version 0.3.1+. To replecate behaviour",
                                   ", use {", dep_comp[ckey][0], ": ",
                                   repr(dep_comp[ckey][1]), "}"]),
                          DeprecationWarning, stacklevel=2)
            del kwargs[ckey]

            if hasattr(ocb, dep_comp[ckey][0]):
                kwargs[dep_comp[ckey][0]] = dep_comp[ckey][1]

    # Remove the data with NaNs/Inf and from the opposite hemisphere/equator
    igood = np.where((np.isfinite(mdata['MLT'])) & (np.isfinite(mdata['MLAT']))
                     & (np.isfinite(mdata['BE'])) & (np.isfinite(mdata['BN']))
                     & (np.isfinite(mdata['BZ']))
                     & (np.sign(mdata['MLAT']) == hemisphere))[0]

    if igood.shape != mdata['MLT'].shape:
        for mkey in mdata.keys():
            mdata[mkey] = mdata[mkey][igood]

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
                                        max_tol=max_sdiff, min_merit=min_merit,
                                        max_merit=max_merit, **kwargs)

            if imag < nmag and ocb.rec_ind < ocb.records:
                # Get all of the points for this time pairing
                itime = np.where(mdata['DATETIME'] == mdata['DATETIME'][imag])

                # Set this value's AACGM vector values
                vdata = ocbscal.VectorData(
                    itime[0], ocb.rec_ind, mdata['MLAT'][itime],
                    mdata['MLT'][itime], aacgm_n=mdata['BN'][itime],
                    aacgm_e=mdata['BE'][itime], aacgm_z=mdata['BZ'][itime],
                    scale_func=scale_func)

                vdata.set_ocb(ocb)

                # Output one line for each time
                for tind, jmag in enumerate(itime[0]):
                    # Format the output line:
                    #    DATE TIME NST [SML SMU] STID [SZA] MLAT MLT BMAG BN BE
                    #    BZ OCB_MLAT OCB_MLT OCB_BMAG OCB_BN OCB_BE OCB_BZ
                    # Recall that NST is the number of stations at this time,
                    # so output the number of indices to be output at this time.
                    outline = "{:} {:d} {:s} ".format(mdata['DATETIME'][jmag],
                                                      len(itime[0]),
                                                      mdata['STID'][jmag])

                    for okey in optional_keys:
                        if okey == "SZA":
                            outline = "{:s}{:.2f} ".format(outline,
                                                           mdata[okey][jmag])
                        else:
                            outline = "{:s}{:d} ".format(outline,
                                                         mdata[okey][jmag])

                    outline = "".join([
                        outline, "{:.2f} ".format(vdata.aacgm_lat[tind]),
                        "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} ".format(
                            vdata.aacgm_mlt[tind], vdata.aacgm_mag[tind],
                            vdata.aacgm_n[tind], vdata.aacgm_e[tind],
                            vdata.aacgm_z[tind], vdata.ocb_lat[tind]),
                        "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                            vdata.ocb_mlt[tind], vdata.ocb_mag[tind],
                            vdata.ocb_n[tind], vdata.ocb_e[tind],
                            vdata.ocb_z[tind])])
                    fout.write(outline)

                # Move to next line
                imag = itime[0][-1] + 1

    return


def load_supermag_ascii_data(filename):
    """Load a SuperMAG ASCII data file.

    Parameters
    ----------
    filename : str
        SuperMAG ASCI data file name

    Returns
    -------
    out : dict
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
