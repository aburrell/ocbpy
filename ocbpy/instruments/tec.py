# -*- coding: utf-8 -*-
# Copyright (C) 2017
# Full license can be found in LICENSE.txt
""" Perform OCB gridding for TEC data

Functions
----------------------------------------------------------------------------
madrigal_tec2ascii_ocb(tecfile, outfile, kwargs)
     Write and ASCII file with Madrigal TEC data and the OCB coordinates for
     each data point
load_madrigal_hdf5_tec(filename)
     Load Madrigal HDF5 TEC data files

Data
----------------------------------------------------------------------------
Madrigal data available at: http://madrigal.haystack.mit.edu/madrigal/

Note
----------------------------------------------------------------------------
Requires AACGM model, which is run through davitpy:
 https://github.com/vtsuperdarn/davitpy
"""
import logging
import numpy as np

def madrigal_tec2ascii_ocb(tecfile, outfile, ocb=None, ocbfile=None,
                           eq_boundary=45.0, max_sdiff=600, min_sectors=7,
                           rcent_dev=8.0, max_r=23.0, min_r=10.0, min_j=0.15):
    """ Coverts the location of SuperMAG data into a frame that is relative to
    the open-closed field-line boundary (OCB) as determined  from a circle fit
    to the poleward boundary of the auroral oval

    Parameters
    ----------
    tecfile : (str)
        file containing the required vorticity file sorted by time
    outfile : (str)
        filename for the output data
    ocb : (OCBoundary or NoneType)
        OCBoundary object with data loaded from an OC boundary data file.
        If None, looks to ocbfile
    ocbfile : (str or NoneType)
        file containing the required OC boundary data sorted by time, or None
        to use IMAGE WIC or to pass in an OCBoundary object (default=None)
    eq_boundary : (float)
        Minimum equatorward co-latitude for which OCB coordinates will be
        calculated (default=45.0)
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
    import datetime as dt
    
    try:
        from davitpy.models import aacgm
    except:
        estr = "Unable to compute OCB without AACGM coordinates\n"
        estr += "Currently using davitpy to impliment AACGM V2"
        logging.error(estr)
        return

    assert isinstance(outfile, str), \
        logging.error("output filename is not a string [{:}]".format(outfile))

    # Read the superMAG data and calculate the magnetic field magnitude
    tdata = load_madrigal_hdf5_tec(tecfile)

    # Remove the data from the opposite hemisphere
    igood = [i for i, lat in enumerate(tdata['gdlat'])
             if np.sign(lat) == ocb.hemisphere and abs(lat) > eq_boundary]

    for k in tdata.keys():
        tdata[k] = tdata[k][igood]

    # Load the OCB data for the SuperMAG data period
    if ocb is None or not isinstance(ocb, ocbpy.ocboundary.OCBoundary):
        tstart = tdata['datetime'][0] - dt.timedelta(seconds=max_sdiff+1)
        tend = tdata['datetime'][-1] + dt.timedelta(seconds=max_sdiff+1)
        ocb = ocbpy.OCBoundary(ocbfile, stime=tstart, etime=tend)

    # Test the OCB data
    if ocb.filename is None or ocb.records == 0:
        logging.error("no data in OCB file {:s}".format(ocb.filename))
        return

    # Open and test the file to ensure it can be written
    try:
        fout = open(outfile, 'w')
    except:
        logging.error("unable to create output file [{:}]".format(outfile))
        return

    # Write the output line
    outline = "#date time ut1_unix ut2_unix recno gdlat glon mlat mlon mlt "
    outline += "ocb_lat ocb_mlt tec dtec\n"
    try:
        fout.write(outline)
    except:
        estr = "unable to write [{:s}] because of error ".format(outline)
        estr = "{:s}[{:}]".format(estr, e)
        logging.error(estr)
        return
    
    # Initialise the ocb and vorticity indices
    itec = 0
    ntec = tdata['datetime'].shape[0]
    
    # Cycle through the data, matching vorticity and OCB records
    while itec < ntec and ocb.rec_ind < ocb.records:
        itec = ocbpy.match_data_ocb(ocb, tdata['datetime'], idat=itec,
                                    max_tol=max_sdiff, min_sectors=min_sectors,
                                    rcent_dev=rcent_dev, max_r=max_r,
                                    min_r=min_r, min_j=min_j)
        
        if itec < ntec and ocb.rec_ind < ocb.records:
            # Calculate the AACGM coordinates
            mlat, mlon, mlt = aacgm.get_aacgm_coord(tdata['gdlat'][itec],
                                                    tdata['glon'][itec], 350.0,
                                                    tdata['datetime'][itec])
            ocb_lat, ocb_mlt = ocb.normal_coord(mlat, mlt)

            # Format the output line
            #    date time ut1_unix ut2_unix recno gdlat glon mlat mlon mlt
            #    ocb_lat ocb_mlt tec dtec
            outline = "{:} {:d} {:d} {:d} ".format(tdata['datetime'][itec],
                                                   tdata['ut1_unix'][itec],
                                                   tdata['ut2_unix'][itec],
                                                   tdata['recno'][itec])
            outline += "{:.1f} {:.1f} {:.1f}".format(tdata['gdlat'][itec],
                                                     tdata['glon'][itec], mlat)
            outline += " {:.1f} {:.2f} {:.2f} ".format(mlon, mlt, ocb_lat)
            outline += "{:.2f} {:.1f} {:.1f}\n".format(ocb_mlt,
                                                       tdata['tec'][itec],
                                                       tdata['dtec'][itec])

            try:
                fout.write(outline)
            except e:
                estr = "unable to write [{:s}] ".format(outline)
                estr = "{:s}because of error [{:}]".format(estr, e)
                logging.error(estr)
                return
            
            # Move to next line
            itec += 1

    # Close output file
    fout.close()
        
    return

#---------------------------------------------------------------------------
# load_madrigal_hdf5_tec: A routine to open a Madrigal HDF5 TEC file

def load_madrigal_hdf5_tec(filename):
    """Open a Madrigal HDF5 TEC file and load it into a dictionary of nparrays

    Parameters
    ------------
    filename : (str)
        Madrigal HDF5 data file name

    Returns
    ----------
    out : (dict of numpy.arrays)
        The dict keys specify the data names, the data for each key are stored
        numpy arrays
    """
    from ocbpy.instruments import test_file
    import datetime as dt
    import h5py

    # Open the file in read-only mode
    try:
        f = h5py.File(filename, 'r')
    except:
        logging.error("unable to open file [{:s}]".format(filename))
        return

    # Load the variables into a standard numpy array and save the description
    # as attributes
    out = dict()

    # Cycle through the file groups and locate the datasets
    try:
        dhandle = f['Data']
    except:
        logging.error("unable to locate Dataset key")
        f.close()
        return

    try:
        dhandle = dhandle['Table Layout']
    except:
        logging.error("unable to locate [Table Layout] key")
        f.close()
        return

    if isinstance(dhandle, h5py.Dataset):
        okeys = dhandle.dtype.fields

        # Load the data
        for ok in okeys.keys():
            out[ok] = dhandle[ok]

        # Construct the datetime array
        out['datetime'] = np.array([dt.datetime.strptime("{:d} {:d} {:d} {:d} {:d} {:d}".format(out['year'][i], out['month'][i], out['day'][i], h, out['min'][i], out['sec'][i]), "%Y %m %d %H %M %S") for i,h in enumerate(out['hour'])])

    else:
        logging.error("dataset not where expected")

    # Close HDF5 file handle and return data
    f.close()
    return out
