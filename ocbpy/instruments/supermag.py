# -*- coding: utf-8 -*-
# Copyright (C) 2017
# Full license can be found in LICENSE.txt
""" Perform OCB gridding for SuperMAG data

Functions
----------------------------------------------------------------------------
supermag2ascii_ocb(smagfile, outfile, kwargs)
     Write and ASCII file with SuperMAG data and the OCB coordinates for each
     data point
load_supermag_ascii_data(filename)
     Load SuperMAG ASCII data files

Data
----------------------------------------------------------------------------
SuperMAG data available at: http://supermag.jhuapl.edu/
"""
import logbook as logging
import numpy as np

def supermag2ascii_ocb(smagfile, outfile, ocb=None, ocbfile=None,
                       max_sdiff=600, min_sectors=7, rcent_dev=8.0, max_r=23.0,
                       min_r=10.0, min_j=0.15):
    """ Coverts the location of SuperMAG data into a frame that is relative to
    the open-closed field-line boundary (OCB) as determined  from a circle fit
    to the poleward boundary of the auroral oval

    Parameters
    ----------
    smagfile : (str)
        file containing the required SuperMAG file sorted by time
    outfile : (str)
        filename for the output data
    ocb : (OCBoundary or NoneType)
        OCBoundary object with data loaded from an OC boundary data file.
        If None, looks to ocbfile
    ocbfile : (str or NoneType)
        file containing the required OC boundary data sorted by time, or None
        to use IMAGE WIC or to pass in an OCBoundary object (default=None)
    max_sdiff : (int)
        maximum seconds between OCB and data record in sec (default=600)
    min_sectors : (int)
        Minimum number of MLT sectors required for good OCB (default=7).
    rcent_dev : (float)
        Maximum number of degrees between the new centre and the AACGM pole
        (default=8.0).
    max_r : (float)
        Maximum radius for open-closed field line boundary in degrees
        default=23.0).
    min_r : (float)
        Minimum radius for open-closed field line boundary in degrees
        (default=10.0).
    min_j : (float)
        Minimum unitless current magnitude scale difference (default=0.15)

    Returns
    ---------
    Void
    """
    import ocbpy
    import ocbpy.ocb_scaling as ocbscal
    import datetime as dt

    assert ocbpy.instruments.test_file(smagfile), \
    logging.error("supermag file cannot be opened [{:s}]".format(smagfile))
    assert isinstance(outfile, str), \
        logging.error("output filename is not a string [{:}]".format(outfile))

    # Read the superMAG data and calculate the magnetic field magnitude
    header, mdata = load_supermag_ascii_data(smagfile)

    # Remove the data with NaNs
    igood = [i for i, mlt in enumerate(mdata['MLT']) if not np.isnan(mlt)
             and not np.isnan(mdata['MLAT'][i]) and not np.isnan(mdata['BE'][i])
             and not np.isnan(mdata['BN'][i]) and not np.isnan(mdata['BZ'][i])]

    for k in mdata.keys():
        mdata[k] = mdata[k][igood]

    # Load the OCB data for the SuperMAG data period
    if ocb is None or not isinstance(ocb, ocbpy.ocboundary.OCBoundary):
        mstart = mdata['DATETIME'][0] - dt.timedelta(seconds=max_sdiff+1)
        mend = mdata['DATETIME'][-1] + dt.timedelta(seconds=max_sdiff+1)
        ocb = ocbpy.OCBoundary(ocbfile, stime=mstart, etime=mend)

    # Test the OCB data
    if ocb.filename is None or ocb.records == 0:
        try:
            logging.error("no data in OCB file {:s}".format(ocb.filename))
        except:
            logging.error("bad OCB file specified")
        return

    # Open and test the file to ensure it can be written
    try:
        fout = open(outfile, 'w')
    except:
        logging.error("unable to create output file [{:}]".format(outfile))
        return

    # Write the output line
    outline = "#DATE TIME NST STID "
    optional_keys = ["SML", "SMU", "SZA"]
    for okey in optional_keys:
        if okey in mdata.keys():
            outline = "{:s}{:s} ".format(outline, okey)

    outline = "{:s}MLAT MLT BMAG BN BE BZ OCB_MLAT OCB_MLT ".format(outline)
    outline = "{:s}OCB_BMAG OCB_BN OCB_BE OCB_BZ\n".format(outline)
    try:
        fout.write(outline)
    except:
        estr = "unable to write [{:s}] because of error ".format(outline)
        estr = "{:s}[{:}]".format(estr, e)
        logging.error(estr)
        return
    
    # Initialise the ocb and SuperMAG indices
    imag = 0
    nmag = mdata['DATETIME'].shape[0]
    
    # Cycle through the data, matching SuperMAG and OCB records
    while imag < nmag and ocb.rec_ind < ocb.records:
        imag = ocbpy.match_data_ocb(ocb, mdata['DATETIME'], idat=imag,
                                    max_tol=max_sdiff, min_sectors=min_sectors,
                                    rcent_dev=rcent_dev, max_r=max_r,
                                    min_r=min_r, min_j=min_j)
        
        if imag < nmag and ocb.rec_ind < ocb.records:
            # Set this value's AACGM vector values
            vdata = ocbscal.VectorData(imag, ocb.rec_ind, mdata['MLAT'][imag],
                                       mdata['MLT'][imag],
                                       aacgm_n=mdata['BN'][imag],
                                       aacgm_e=mdata['BE'][imag],
                                       aacgm_z=mdata['BZ'][imag],
                                       scale_func=ocbscal.normal_curl_evar)
            
            vdata.set_ocb(ocb)

            # Format the output line
            #    DATE TIME NST [SML SMU] STID [SZA] MLAT MLT BMAG BN BE BZ
            #    OCB_MLAT OCB_MLT OCB_BMAG OCB_BN OCB_BE OCB_BZ
            outline = "{:} {:d} {:s} ".format(mdata['DATETIME'][imag],
                                              mdata['NST'][imag],
                                              mdata['STID'][imag])

            for okey in optional_keys:
                if okey == "SZA":
                    outline = "{:s}{:.2f} ".format(outline, mdata[okey][imag])
                else:
                    outline = "{:s}{:d} ".format(outline, mdata[okey][imag])
            
            outline = "{:s}{:.2f} {:.2f} {:.2f} {:.2f} ".format(outline, \
            vdata.aacgm_lat, vdata.aacgm_mlt, vdata.aacgm_mag, vdata.aacgm_n)
            outline = "{:s}{:.2f} {:.2f} {:.2f} {:.2f} ".format(outline, \
                    vdata.aacgm_e, vdata.aacgm_z, vdata.ocb_lat, vdata.ocb_mlt)
            outline = "{:s}{:.2f} {:.2f} {:.2f} {:.2f}\n".format(outline, \
                    vdata.ocb_mag, vdata.ocb_n, vdata.ocb_e, vdata.ocb_z)
            try:
                fout.write(outline)
            except e:
                estr = "unable to write [{:s}] ".format(outline)
                estr = "{:s}because of error [{:}]".format(estr, e)
                logging.error(estr)
                return
            
            # Move to next line
            imag += 1

    # Close output file
    fout.close()
        
    return

#---------------------------------------------------------------------------
# load_supermag_ascii_data: A routine to open a supermag ascii file

def load_supermag_ascii_data(filename):
    """Open a SuperMAG ASCII data file and load it into a dictionary of nparrays

    Parameters
    ------------
    filename : (str)
        SuperMAG ASCI data file name

    Returns
    ----------
    out : (dict of numpy.arrays)
        The dict keys are specified by the header data line, the data
        for each key are stored in the numpy array
    """
    from ocbpy.instruments import test_file
    import datetime as dt
    
    fill_val = 999999
    header = list()
    ind = {"SMU":fill_val, "SML":fill_val}
    out = {"YEAR":list(), "MONTH":list(), "DAY":list(), "HOUR":list(),
           "MIN":list(), "SEC":list(), "DATETIME":list(), "NST":list(),
           "SML":list(), "SMU":list(), "STID":list(), "BN":list(), "BE":list(),
           "BZ":list(), "MLT":list(), "MLAT":list(), "DEC":list(), "SZA":list()}
    
    if not test_file(filename):
        return header, dict()
    
    #----------------------------------------------
    # Open the datafile and read the data
    try:
        f = open(filename, "r")
    except:
        logging.error("unable to open input file [{:s}]".format(filename))
        return header, dict()

    hflag = True
    n = -1
    for line in f.readlines():
        if hflag:
            # Fill the header list
            header.append(line)
            if line.find("=========================================") >= 0:
                hflag = False
                dflag = True
        else:
            # Fill the output dictionary
            if n < 0:
                # This is a date line
                n = 0
                lsplit = np.array(line.split(), dtype=int)
                dtime = dt.datetime(lsplit[0], lsplit[1], lsplit[2], lsplit[3],
                                    lsplit[4], lsplit[5])
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
                        ind = {"SMU":fill_val, "SML":fill_val}

    f.close()

    # Recast data as numpy arrays and replace fill value with np.nan
    for k in out:
        if k == "STID":
            out[k] = np.array(out[k], dtype=str)
        else:
            out[k] = np.array(out[k])

            if k in ['BE', 'BN', 'DEC', 'SZA', 'MLT', 'BZ']:
                out[k][out[k] == fill_val] = np.nan
    
    return header, out

# End load_supermag_ascii_data
