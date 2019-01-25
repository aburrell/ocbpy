# -*- coding: utf-8 -*-
# Copyright (C) 2017
# Full license can be found in LICENSE.txt
#---------------------------------------------------------------------------
""" Perform OCB gridding for SuperDARN vorticity data

Functions
----------------------------------------------------------------------------
vort2ascii_ocb(vortfile, outfile, kwargs)
    Write and ASCII file with SuperDARN data and the OCB coordinates for each
    data point
load_vorticity_ascii_data(filename, save_all=False)
    Load vorticity block ASCII data files

Data
----------------------------------------------------------------------------
Specialised SuperDARN data product, available from: gchi@bas.ac.uk
"""
import logbook as logging
import numpy as np

def vort2ascii_ocb(vortfile, outfile, ocb=None, ocbfile=None, max_sdiff=600,
                   save_all=False, min_sectors=7, rcent_dev=8.0, max_r=23.0,
                   min_r=10.0, min_j=0.15):
    """ Coverts the location of vorticity data in AACGM coordinates into a frame
    that is relative to the open-closed field-line boundary (OCB) as determined
    from a circle fit to the poleward boundary of the auroral oval

    Parameters
    ----------
    vortfile : (str)
        file containing the required vorticity file sorted by time
    outfile : (str)
        filename for the output data
    ocb : (ocbpy.ocboundary.OCBoundary or NoneType)
        Object containing open closed boundary data or None to load from file
    ocbfile : (str or NoneType)
        file containing the required OC boundary data sorted by time, or None
        to use ocb object or IMAGE WIC file (default=None)
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
    min_j : (float)
        Minimum unitless current magnitude scale difference (default=0.15)

    Returns
    ---------
    Void

    Notes
    --------
    Input header or col_names must include the names in the default string.
    """
    import ocbpy
    import ocbpy.ocb_scaling as ocbscal
    import datetime as dt

    assert ocbpy.instruments.test_file(vortfile), \
        logging.error("vorticity file cannot be opened[{:s}]".format(vortfile))
    assert isinstance(outfile, str), \
        logging.error("output filename is not a string [{:}]".format(outfile))

    # Read the vorticity data
    vdata = load_vorticity_ascii_data(vortfile, save_all=save_all)
    need_keys = ["VORTICITY", "CENTRE_MLAT", "DATETIME", "MLT"]
    
    if vdata is None or not all([kk in vdata.keys() for kk in need_keys]):
        estr = "unable to load necessary data from [{:s}]".format(vortfile)
        logging.error(estr)
        return

    # Load the OCB data
    if ocb is None or not isinstance(ocb, ocbpy.ocboundary.OCBoundary):
        vstart = vdata['DATETIME'][0] - dt.timedelta(seconds=max_sdiff+1)
        vend = vdata['DATETIME'][-1] + dt.timedelta(seconds=max_sdiff+1)
        ocb = ocbpy.ocboundary.OCBoundary(ocbfile, stime=vstart, etime=vend)

    if ocb.filename is None or ocb.records == 0:
        try:
            logging.error("no data in OCB file {:s} ".format(ocb.filename))
        except:
            logging.error("bad OCB file specified")
        return

    # Set the reference radius
    ref_r = 90.0 - abs(ocb.boundary_lat)

    # Open and test the file to ensure it can be written
    try:
        fout = open(outfile, 'w')
    except:
        logging.error("unable to create output file [{:}]".format(outfile))
        return

    # Write header line
    outline = "#DATE TIME "

    if save_all:
        vkeys = vdata.keys()
        vkeys.pop(vkeys.index("DATETIME"))
        outline = "{:s}{:s} ".format(outline, " ".join(vkeys))

    outline = "{:s}OCB_LAT OCB_MLT NORM_VORT\n".format(outline)
    
    try:
        fout.write(outline)
    except:
        estr = "unable to write [{:s}] because of error ".format(outline)
        estr = "{:s}[{:}]".format(estr, e)
        logging.error(estr)
        return

    # Initialise the ocb and vorticity indices
    ivort = 0
    num_vort = vdata['DATETIME'].shape[0]

    # Cycle through the data, matching vorticity and OCB records
    while ivort < num_vort and ocb.rec_ind < ocb.records:
        ivort = ocbpy.match_data_ocb(ocb, vdata['DATETIME'], idat=ivort,
                                     max_tol=max_sdiff, min_sectors=min_sectors,
                                     rcent_dev=rcent_dev, max_r=max_r,
                                     min_r=min_r, min_j=min_j)
        
        if ivort < num_vort and ocb.rec_ind < ocb.records:
            # Use the indexed OCB to convert the AACGM grid coordinate to one
            # related to the OCB
            nlat, nmlt = ocb.normal_coord(vdata['CENTRE_MLAT'][ivort],
                                          vdata['MLT'][ivort])
            nvort = ocbscal.normal_curl_evar(vdata['VORTICITY'][ivort],
                                             ocb.r[ocb.rec_ind], ref_r)

            # Format the output line
            #    DATE TIME (SAVE_ALL) OCB_LAT OCB_MLT NORM_VORT
            outline = "{:} ".format(vdata['DATETIME'][ivort])

            if save_all:
                for k in vkeys:
                    outline = "{:s}{:} ".format(outline, vdata[k][ivort])

            outline = "{:s}{:.2f} {:.6f} {:.6f}\n".format(outline, nlat, nmlt,
                                                          nvort)
            
            try:
                fout.write(outline)
            except e:
                estr = "unable to write [{:s}] ".format(outline)
                estr = "{:s}because of error [{:}]".format(estr, e)
                logging.error(estr)
                return

            # Move to next line
            ivort += 1

    # Close output file
    fout.close()
        
    return

def load_vorticity_ascii_data(vortfile, save_all=False):
    """Load SuperDARN vorticity data files.

    Parameters
    -----------
    vortfile : (str)
        SuperDARN vorticity file in block format
    save_all : (bool)
        Save all data from the file (True), or only data needed to calculate
        the OCB coordinates and normalised vorticity (False). (default=False)

    Returns
    ---------
    vdata : (dict)
        Dictionary of numpy arrays
    """
    from ocbpy.instruments import test_file
    import datetime as dt

    if not test_file(vortfile):
        return None

    # Open the data file
    try:
        fvort = open(vortfile, "r")
    except:
        logging.error("unable to open vorticity file [{:s}]".format(vortfile))
        return None

    # Initialise the output dictionary
    vkeys = ["YEAR", "MONTH", "DAY", "UTH", "VORTICITY", "MLT", "CENTRE_MLAT",
             "DATETIME"]
    if save_all:
        vkeys.extend(["R1BM1", "R1BM2", "R2BM1", "R2BM2", "AREA", "CENTRE_GLAT",
                      "CENTRE_GLON", "C1_GLAT", "C1_GLON", "C2_GLAT", "C2_GLON",
                      "C3_GLAT", "C3_GLON", "C4_GLAT", "C4_GLON", "CENTRE_MLON",
                      "C1_MLAT", "C1_MLON", "C2_MLAT", "C2_MLON", "C3_MLAT",
                      "C3_MLON", "C4_MLAT", "C4_MLON"])
    vdata = {k:list() for k in vkeys}
    vkeys = set(vkeys)
    
    # Set the data block keys
    bkeys = [["R1BM1", "R1BM2", "R2BM1", "R2BM2", "AREA", "VORTICITY", "MLT"],
             ["GFLG", "CENTRE_GLAT", "CENTRE_GLON", "C1_GLAT", "C1_GLON",
              "C2_GLAT", "C2_GLON", "C3_GLAT", "C3_GLON", "C4_GLAT", "C4_GLON"],
             ["MFLG", "CENTRE_MLAT", "CENTRE_MLON", "C1_MLAT", "C1_MLON",
              "C2_MLAT", "C2_MLON", "C3_MLAT", "C3_MLON", "C4_MLAT", "C4_MLON"]]
    
    # Read the lines and assign data.  Recall that blank lines in file are
    # returned as '\n'
    vline = fvort.readline()
    vsplit = vline.split()
    vinc = 0

    while len(vline) > 0:
        if vinc == 0:
            # This is a date line
            if len(vsplit) != 4:
                estr = "unexpected line encountered when date line "
                estr = "{:s}expected [{:s}]".format(estr, vline)
                logging.error(estr)
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
                estr = "unexpected line encountered when number of entries "
                estr = "{:s}line expected [{:s}]".format(estr, vline)
                logging.error(estr)
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
                    # Test to see that this line has the right number of col
                    if len(vsplit) != len(bklist):
                        estr = "unexpected line encountered for a data block "
                        estr = "{:s}[{:s}]".format(estr, vline)
                        logging.error(estr)
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
                    
                # All data lines for this entry have been processed, incriment
                ninc += 1
                    
            # All entries in block have been processed, reset incriment
            vinc = 0

        # Move to next line
        vline = fvort.readline()
        vsplit = vline.split()

    # Close file handle
    fvort.close()

    # Recast lists as numpy arrays
    for k in vdata.keys():
        vdata[k] = np.array(vdata[k])

    return vdata
