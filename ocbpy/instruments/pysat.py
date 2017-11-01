# -*- coding: utf-8 -*-
# Copyright (C) 2017
# Full license can be found in LICENSE.txt
""" Perform OCB gridding for appropriate instrument data loaded in pysat

Functions
----------------------------------------------------------------------------
add_ocb_series()
    Add OCB coordinates to the pysat Instrument.DataFrame object
add_ocb_metadata()
    Add OCB metadata to the pysat Instrument object

Data
----------------------------------------------------------------------------
SuperMAG data available at: http://supermag.jhuapl.edu/
"""
import logging
import numpy as np

try:
    import pysat
    import pandas as pds
except:
    err = 'unable to load pysat and/or pandas modules; pysat is available at:\n'
    err += 'https://github.com/rstoneback/pysat'
    raise err

def add_ocb_series(pysat_data, mlat_attr, mlt_attr, evar_attrs=list(),
                   curl_evar_attrs=list(), vector_attrs=dict(), dat_ind=list(),
                   ocb=None, ocbfile=None, max_sdiff=600, min_sectors=7,
                   rcent_dev=8.0, max_r=23.0, min_r=10.0, min_j=0.15):
    """ Coverts the location of pysat data into a frame that is relative to
    the open-closed field-line boundary (OCB) as determined  from a circle fit
    to the poleward boundary of the auroral oval

    Parameters
    ----------
    pysat_data : (pandas.DataFrame)
        DataFrame class object containing magnetic coordinates
    mlat_attr : (str)
        DataFrame attribute pointing to Series of magnetic latitudes
    mlt_attr : (str)
        DataFrame attribute pointing to Series of magnetic longitudes
    evar_attrs : (list)
        List of DataFrame attribute pointing to Series of measurements that
        are proportional to the electric field (E); e.g. ion drift. (default=[])
    curl_evar_attrs : (list)
        List of DataFrame attribute pointing to Series of measurements that
        are proportional to the curl of E (e.g. ion vorticity). (default=[])
    vector_attrs : (dict)
        Dict of DataFrame attribute pointing to Series of measurements that
        are vectors that are proportional to either E or the curl of E. The
        key should correspond to one of the values in the evar_attrs or
        cur_evar_attrs list.  If this is not done, a scaling function must be
        provided.  The value corresponding to each key must be a dict that
        indicates the attributes holding data needed to initialise the
        ocbpy.ocb_scaling.VectorData object. (default={})
        Example: vector_attrs={"vel":{"aacgm_n":"vel_n", "aacgm_e":"vel_e",
                                      "dat_name":"velocity", "dat_units":"m/s"},
                               "dat":{"aacgm_n":"dat_n", "aacgm_e":"dat_e",
                                      "scale_func":local_scale_func}}
    dat_ind : list()
        List of indices to process.  If empty, all data should be from the same
        hemisphere (northern or southern) and be free of NaN. (default=[])
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
    ocb_attrs : (list)
        List of attributes added to pysat_data containing the OCB coordinates
        and any scaled measurements.
    """
    import ocbpy
    import ocbpy.ocb_scaling as ocbscal
    import datetime as dt

    assert isinstance(pysat_data, pds.core.frame.DataFrame), \
        logging.error("unexpected class for pysat data")
    assert hasattr(pysat_data, mlat_attr), \
        logging.error("unknown mag lat attribute [{:}]".format(mlat_attr))
    assert hasattr(pysat_data, mlt_attr), \
        logging.error("unknown MLT attribute [{:}]".format(mlt_attr))

    olat_attr = "{:s}_ocb".format(mlat_attr)
    omlt_attr = "{:s}_ocb".format(mlt_attr)
    ocb_attrs = [olat_attr, omlt_attr]

    for eattr in evar_attrs:
        assert hasattr(pysat_data, eattr), \
            logging.error("unknown E field attribute [{:}]").format(eattr)
        ocb_attrs.append("{:s}_ocb".format(eattr)

    for eattr in curl_evar_attrs:
        assert hasattr(pysat_data, eattr), \
            logging.error("unknown curl E field attribute [{:}]").format(eattr)
        ocb_attrs.append("{:s}_ocb").format(eattr)

    # Test the vector attributes to ensure that enough information
    # was provided and that it exists in the DataFrame
    nvect = len(vector_attrs.keys())
    if nvect > 0:
        vector_reqs = ["aacgm_n", "aacgm_e", "aacgm_z"]

        for eattr in vector_attrs.keys():
            vdim = 0
            vfunc = False
            for vinit in vector_attrs[eattr].keys():
                if vinit in vector_reqs:
                    assert hasattr(pysat_data, vinit), \
                logging.error("unknown vector attribute [{:}]").format(vinit))
                    vdim += 1

                if vinit in evar_attrs:
                    vector_attrs[eattr]["scale_func"] = ocbscal.normal_evar
                elif vinit in curl_evar_attrs:
                    vector_attrs[eattr]["scale_func"] = ocbscal.normal_curl_evar
                else:
                    assert "scale_func" in vector_attrs[eattr], \
            logging.error("missing scaling function for [{:}]").format(eattr))

    # Extract the AACGM locations
    aacgm_lat = getattr(pysat_data, mlat_attr)
    aacgm_mlt = getattr(pysat_data, mlt_attr)
    ndat = len(aacgm_lat)

    if len(dat_ind) == 0:
        dat_ind = np.arange(0, ndat+1, 1)

    # Load the OCB data for the data period, if desired
    if ocb is None or not isinstance(ocb, ocbpy.ocboundary.OCBoundary):
        dstart = pysat_data.index[dat_ind][0]-dt.timedelta(seconds=max_sdiff+1)
        dend = pysat_data.index[dat_ind][-1] + dt.timedelta(seconds=max_sdiff+1)
        ocb = ocbpy.OCBoundary(ocbfile, stime=dstart, etime=dend)

    # Test the OCB data
    if ocb.filename is None or ocb.records == 0:
        logging.error("no data in OCB file {:s}".format(ocb.filename))
        return

    # Initialise the OCB Series
    ocb_series = dict()
    for oattr in ocb_attrs:
        eattr = oattr[:-4]
        if eattr in vector_attr.keys():
            ocb_series[oattr] = pds.Series(np.empty(shape=aacgm_lat.shape,
                                                    dtype=ocbscal.VectorData),
                                           index=pysat_data.index)
        else:
            ocb_series[oattr] = pds.Series(np.empty(shape=aacgm_lat.shape,
                                                    dtype=float) * np.nan,
                                           index=pysat_data.index)

    # Cycle through the data, matching data and OCB records
    idat = 0
    ndat = len(dat_ind)
    while idat < ndat and ocb.rec_ind < ocb.records:
        idat = ocbpy.match_data_ocb(ocb, pysat_data.index[dat_ind], idat=idat,
                                    max_tol=max_sdiff, min_sectors=min_sectors,
                                    rcent_dev=rcent_dev, max_r=max_r,
                                    min_r=min_r, min_j=min_j)
        
        if idat < ndat and ocb.rec_ind < ocb.records:
            iser = dat_ind[idat]

            # Get the OCB coordinates
            (ocb_series[olat_attr][iser],
             ocb_series[omlt_attr][iser]) = ocb.normal_coord(aacgm_lat[iser],
                                                             aacgm_mlt[iser])

            if nvect > 0:
                # Set this value's AACGM vector values
                vector_default = {"ocb_lat":ocb_series[olat_attr][iser],
                                  "ocb_mlt":ocb_series[omlt_attr][iser],
                                  "aacgm_n":0.0, "aacgm_e":0.0, "aacgm_z":0.0,
                                  "aacgm_mag":np.nan, dat_name:None,
                                  dat_units:None, "scale_func":None}
                vector_init = dict(vector_default)

                for eattr in vector_attrs.keys():
                    for ikey
                    
                    ocb_data[eattr][iser] = ocbscal.VectorData(idat, \
                        ocb.rec_ind, aacgm_lat[iser], aacgm_mlt[iser], \
                                                    **vector_attrs[eattr])
            
            vdata.set_ocb(ocb)

            # Format the output line
            #    DATE TIME NST [SML SMU] STID [SZA] MLAT MLT BMAG BN BE BZ
            #    OCB_MLAT OCB_MLT OCB_BMAG OCB_BN OCB_BE OCB_BZ
            outline = "{:} {:d} {:s} ".format(mdata['DATETIME'][idat],
                                              mdata['NST'][idat],
                                              mdata['STID'][idat])

            for okey in optional_keys:
                if okey == "SZA":
                    outline = "{:s}{:.2f} ".format(outline, mdata[okey][idat])
                else:
                    outline = "{:s}{:d} ".format(outline, mdata[okey][idat])
            
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
            idat += 1

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
    f = open(filename, "r")

    if not f:
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
