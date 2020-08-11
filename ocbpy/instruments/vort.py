# -*- coding: utf-8 -*-
# Copyright (C) 2017 AGB
# Full license can be found in LICENSE.txt
# ---------------------------------------------------------------------------
""" Perform OCB gridding for SuperDARN vorticity data

Functions
---------
vort2ascii_ocb(vortfile, outfile, kwargs)
    Write and ASCII file with SuperDARN data and the OCB coordinates for each
    data point
load_vorticity_ascii_data(filename, save_all=False)
    Load vorticity block ASCII data files

Data
----
Specialised SuperDARN data product, available from: gchi@bas.ac.uk

"""
from __future__ import absolute_import, unicode_literals
import datetime as dt
import numpy as np

import ocbpy
import ocbpy.ocb_scaling as ocbscal


def vort2ascii_ocb(vortfile, outfile, hemisphere=0, ocb=None,
                   ocbfile='default', instrument='', max_sdiff=600,
                   save_all=False, min_sectors=7, rcent_dev=8.0, max_r=23.0,
                   min_r=10.0):
    """ Coverts the location of vorticity data from AACGM to OCB coordinates

    Parameters
    ----------
    vortfile : (str)
        file containing the required vorticity file sorted by time
    outfile : (str)
        filename for the output data
    hemisphere : (int)
        Hemisphere to process (can only do one at a time).  1=Northern,
        -1=Southern, 0=Determine from data (default=0)
    ocb : (ocbpy.ocboundary.OCBoundary or NoneType)
        Object containing open closed boundary data or None to load from file
    ocbfile : (str)
        file containing the required OC boundary data sorted by time, ignorned
        if OCBoundary object supplied. (default='default')
    instrument : (str)
        Instrument providing the OCBoundaries.  Requires 'image' or 'ampere'
        if a file is provided.  If using filename='default', also accepts
        'amp', 'si12', 'si13', 'wic', and ''.  (default='')
    max_sdiff : (int)
        maximum seconds between OCB and data record in sec (default=600)
    save_all : (bool)
        Save all data (True), or only that needed to calcuate OCB and vorticity
        (False). (default=False)
    min_sectors : (int)
        Minimum number of MLT sectors required for good OCB. (default=7)
    rcent_dev : (float)
        Maximum number of degrees between the new centre and the AACGM pole
        (default=8.0).
    max_r : (float)
        Maximum radius for open-closed field line boundary in degrees.
        (default=23.0)
    min_r : (float)
        Minimum radius for open-closed field line boundary in degrees
        (default=10.0)

    Notes
    -----
    Input header or col_names must include the names in the default string.

    """

    if not ocbpy.instruments.test_file(vortfile):
        raise IOError("vorticity file cannot be opened [{:s}]".format(
            vortfile))

    if not isinstance(outfile, str):
        raise IOError("output filename is not a string [{:}]".format(outfile))

    # Read the vorticity data
    vdata = load_vorticity_ascii_data(vortfile, save_all=save_all)
    need_keys = ["VORTICITY", "CENTRE_MLAT", "DATETIME", "MLT"]

    if vdata is None or not all([kk in vdata.keys() for kk in need_keys]):
        estr = "unable to load necessary data from [{:s}]".format(vortfile)
        raise ValueError(estr)

    # Load the OCB data
    if ocb is None or not isinstance(ocb, ocbpy.ocboundary.OCBoundary):
        vstart = vdata['DATETIME'][0] - dt.timedelta(seconds=max_sdiff + 1)
        vend = vdata['DATETIME'][-1] + dt.timedelta(seconds=max_sdiff + 1)

        # If hemisphere isn't specified, set it here
        if hemisphere == 0:
            hemisphere = np.sign(np.nanmax(vdata['CENTRE_MLAT']))

            # Ensure that all data is in the same hemisphere
            if hemisphere == 0:
                hemisphere = np.sign(np.nanmin(vdata['CENTRE_MLAT']))
            elif hemisphere != np.sign(np.nanmin(vdata['CENTRE_MLAT'])):
                raise ValueError("".join(["cannot process observations from ",
                                          "both hemispheres at the same time;",
                                          " set hemisphere=+/-1 to choose."]))

        # Initialize the OCBoundary object
        ocb = ocbpy.ocboundary.OCBoundary(ocbfile, stime=vstart, etime=vend,
                                          instrument=instrument,
                                          hemisphere=hemisphere)
    elif hemisphere == 0:
        # If the OCBoundary object is specified and hemisphere isn't use
        # the OCBoundary object to specify the hemisphere
        hemisphere = ocb.hemisphere

    # Test the OCB data
    if ocb.filename is None or ocb.records == 0:
        ocbpy.logger.error("no data in OCB file [{:}]".format(ocb.filename))
        return

    # Remove the data from the opposite hemisphere
    igood = np.where(np.sign(vdata['CENTRE_MLAT']) == hemisphere)[0]

    if igood.shape != vdata['CENTRE_MLAT'].shape:
        if len(igood) == 0:
            # Exit with warning if no vorticity data from this hemisphere
            ocbpy.logger.warning("".join(["No ", "north" if hemisphere == 1
                                          else "south",
                                          "ern hemisphere data in file: [",
                                          vortfile, "]"]))
            return

        # Downselect vorticity data
        for k in vdata.keys():
            vdata[k] = vdata[k][igood]

    # Set the reference radius
    ref_r = 90.0 - abs(ocb.boundary_lat)

    # Open the output file for writting
    with open(outfile, 'w') as fout:
        # Write header line
        outline = "#DATE TIME"

        if save_all:
            vkeys = [kk for kk in vdata.keys() if kk != "DATETIME"]
            outline = " ".join([outline, " ".join(vkeys)])

        outline = " ".join([outline, "OCB_LAT OCB_MLT NORM_VORT\n"])
        fout.write(outline)

        # Initialise the ocb and vorticity indices
        ivort = 0
        num_vort = vdata['DATETIME'].shape[0]

        # Cycle through the data, matching vorticity and OCB records
        while ivort < num_vort and ocb.rec_ind < ocb.records:
            ivort = ocbpy.match_data_ocb(ocb, vdata['DATETIME'], idat=ivort,
                                         max_tol=max_sdiff,
                                         min_sectors=min_sectors,
                                         rcent_dev=rcent_dev, max_r=max_r,
                                         min_r=min_r)

            if ivort < num_vort and ocb.rec_ind < ocb.records:
                # Use the indexed OCB to convert the AACGM grid coordinate to
                # one related to the OCB
                nlat, nmlt, ncor = ocb.normal_coord(
                    vdata['CENTRE_MLAT'][ivort], vdata['MLT'][ivort])
                nvort = ocbscal.normal_curl_evar(vdata['VORTICITY'][ivort],
                                                 ocb.r[ocb.rec_ind] + ncor,
                                                 ref_r)

                # Format the output line
                #    DATE TIME (SAVE_ALL) OCB_LAT OCB_MLT NORM_VORT
                outline = "{:} ".format(vdata['DATETIME'][ivort])

                if save_all:
                    for kk in vkeys:
                        outline += "{:} ".format(vdata[kk][ivort])

                outline += "{:.2f} {:.6f} {:.6f}\n".format(nlat, nmlt, nvort)
                fout.write(outline)

                # Move to next line
                ivort += 1

    return


def load_vorticity_ascii_data(vortfile, save_all=False):
    """Load SuperDARN vorticity data files.

    Parameters
    ----------
    vortfile : (str)
        SuperDARN vorticity file in block format
    save_all : (bool)
        Save all data from the file (True), or only data needed to calculate
        the OCB coordinates and normalised vorticity (False). (default=False)

    Returns
    -------
    vdata : (dict)
        Dictionary of numpy arrays

    """

    if not ocbpy.instruments.test_file(vortfile):
        return None

    # Open the data file
    with open(vortfile, "r") as fvort:
        # Initialise the output dictionary
        vkeys = ["YEAR", "MONTH", "DAY", "UTH", "VORTICITY", "MLT",
                 "CENTRE_MLAT", "DATETIME"]
        if save_all:
            vkeys.extend(["R1BM1", "R1BM2", "R2BM1", "R2BM2", "AREA",
                          "CENTRE_GLAT", "CENTRE_GLON", "C1_GLAT", "C1_GLON",
                          "C2_GLAT", "C2_GLON", "C3_GLAT", "C3_GLON",
                          "C4_GLAT", "C4_GLON", "CENTRE_MLON", "C1_MLAT",
                          "C1_MLON", "C2_MLAT", "C2_MLON", "C3_MLAT",
                          "C3_MLON", "C4_MLAT", "C4_MLON"])
        vdata = {vk: list() for vk in vkeys}
        vkeys = set(vkeys)

        # Set the data block keys
        bkeys = [["R1BM1", "R1BM2", "R2BM1", "R2BM2", "AREA", "VORTICITY",
                  "MLT"],
                 ["GFLG", "CENTRE_GLAT", "CENTRE_GLON", "C1_GLAT", "C1_GLON",
                  "C2_GLAT", "C2_GLON", "C3_GLAT", "C3_GLON", "C4_GLAT",
                  "C4_GLON"],
                 ["MFLG", "CENTRE_MLAT", "CENTRE_MLON", "C1_MLAT", "C1_MLON",
                  "C2_MLAT", "C2_MLON", "C3_MLAT", "C3_MLON", "C4_MLAT",
                  "C4_MLON"]]

        # Read the lines and assign data.  Recall that blank lines in file are
        # returned as '\n'
        vline = fvort.readline()
        vsplit = vline.split()
        vinc = 0

        while len(vline) > 0:
            if vinc == 0:
                # This is a date line
                if len(vsplit) != 4:
                    estr = "".join(["unexpected line encountered when date ",
                                    "line expected [{:s}]".format(vline)])
                    ocbpy.logger.error(estr)
                    fvort.close()
                    return None

                # Save the data in the format desired for the output dict
                yy = int(vsplit[0])
                mm = int(vsplit[1])
                dd = int(vsplit[2])
                hh = float(vsplit.pop())

                # Calculate and save the datetime
                stime = " ".join(vsplit)
                dtime = (dt.datetime.strptime(stime, "%Y %m %d") +
                         dt.timedelta(seconds=np.floor(hh * 3600.0)))
                vinc += 1
            elif vinc == 1:
                # This is a number of entries line
                if len(vsplit) != 1:
                    estr = "".join(["unexpected line encountered when number",
                                    " of entries line expected ",
                                    "[{:s}]".format(vline)])
                    ocbpy.logger.error(estr)
                    fvort.close()
                    return None

                # Save the number of entries
                nentries = int(vsplit[0])
                vinc += 1
            else:
                # This is an entry.  For each entry there are three lines
                ninc = 0
                while ninc < nentries:
                    # Save the time data
                    vdata['YEAR'].append(yy)
                    vdata['MONTH'].append(mm)
                    vdata['DAY'].append(dd)
                    vdata['UTH'].append(hh)
                    vdata['DATETIME'].append(dtime)

                    for bklist in bkeys:
                        # Test to see that this line has the right number of
                        # columns
                        if len(vsplit) != len(bklist):
                            estr = "".join(["unexpected line encountered ",
                                            "for a data block ",
                                            "[{:s}]".format(vline)])
                            ocbpy.logger.error(estr)
                            fvort.close()
                            return None

                        # Save all desired keys
                        gkeys = list(vkeys.intersection(bklist))

                        for gk in gkeys:
                            ik = bklist.index(gk)
                            vdata[gk].append(float(vsplit[ik]))

                        # Move to next line
                        vline = fvort.readline()
                        vsplit = vline.split()

                    # All data lines for this entry have been processed,
                    # block line incriment
                    ninc += 1

                # All entries in block have been processed, reset incriment
                vinc = 0

            # Move to next line
            vline = fvort.readline()
            vsplit = vline.split()

    # Recast lists as numpy arrays
    for k in vdata.keys():
        vdata[k] = np.array(vdata[k])

    return vdata
