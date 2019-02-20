# -*- coding: utf-8 -*-
# Copyright (C) 2017
# Full license can be found in LICENSE.txt
""" Perform OCB gridding for appropriate instrument data loaded in pysat

Functions
----------------------------------------------------------------------------
add_ocb_series()
    Add OCB coordinates to the pysat Instrument.DataFrame object
add_ocb_metadata()
    Add OCB metadata to the pysat Instrument object (NEEDED)

Module
----------------------------------------------------------------------------
pysat is available at: http://github.com/rstoneback/pysat
"""
import logging
import numpy as np

try:
    import pysat
    import pandas as pds
except:
    err = 'unable to load pysat and/or pandas modules; pysat is available at:\n'
    err += 'https://github.com/rstoneback/pysat'
    raise ImportError(err)

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
        curl_evar_attrs list.  If this is not done, a scaling function must be
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
        assert hasattr(pysat_data, eattr) or eattr in vector_attrs.keys(), \
            logging.error("unknown E field attribute [{:}]".format(eattr))
        ocb_attrs.append("{:s}_ocb".format(eattr))

    for eattr in curl_evar_attrs:
        assert hasattr(pysat_data, eattr) or eattr in vector_attrs.keys(), \
            logging.error("unknown curl E field attribute [{:}]").format(eattr)
        ocb_attrs.append("{:s}_ocb".format(eattr))

    # Test the vector attributes to ensure that enough information
    # was provided and that it exists in the DataFrame
    nvect = len(vector_attrs.keys())
    if nvect > 0:
        vector_reqs = ["aacgm_n", "aacgm_e", "aacgm_z"]

        for eattr in vector_attrs.keys():
            vdim = 0
            vfunc = False

            if eattr in evar_attrs:
                vector_attrs[eattr]["scale_func"] = ocbscal.normal_evar
                evar_attrs.pop(evar_attrs.index(eattr))
            elif eattr in curl_evar_attrs:
                vector_attrs[eattr]["scale_func"] = ocbscal.normal_curl_evar
                curl_evar_attrs.pop(curl_evar_attrs.index(eattr))
            else:
                assert "scale_func" in vector_attrs[eattr], \
            logging.error("missing scaling function for [{:}]".format(eattr))

            oattr = "{:s}_ocb".format(eattr)
            if not oattr in ocb_attrs:
                ocb_attrs.append(oattr)
            
            for vinit in vector_attrs[eattr].keys():
                if vinit in vector_reqs:
                    assert hasattr(pysat_data, vector_attrs[eattr][vinit]), \
                        logging.error("unknown vector attribute [{:}]".format(vector_attrs[eattr][vinit]))
                    vdim += 1

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
        if eattr in vector_attrs.keys():
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
    ref_r = 90.0 - abs(ocb.boundary_lat)
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
                vector_default = {"ocb_lat": ocb_series[olat_attr][iser],
                                  "ocb_mlt": ocb_series[omlt_attr][iser],
                                  "aacgm_n": 0.0, "aacgm_e": 0.0,
                                  "aacgm_z": 0.0, "aacgm_mag": np.nan,
                                  "dat_name": None, "dat_units": None,
                                  "scale_func": None}
                vector_init = dict(vector_default)

                for eattr in vector_attrs.keys():
                    oattr = "{:s}_ocb".format(eattr)
                    for ikey in vector_attrs[eattr].keys():
                        try:
                            vector_init[ikey] = getattr(pysat_data, \
                                            vector_attrs[eattr][ikey])[iser]
                        except:
                            # Not all vector attributes are DataFrame attributes
                            vector_init[ikey] = vector_attrs[eattr][ikey]
                    
                    ocb_series[oattr][iser] = ocbscal.VectorData(iser, \
                ocb.rec_ind, aacgm_lat[iser], aacgm_mlt[iser], **vector_init)
                    ocb_series[oattr][iser].set_ocb(ocb)

                unscaled_r = ocb.rfunc[ocb.rec_ind](ocb, aacgm_mlt[iser], \
                                                ocb.rfunc_kwargs[ocb.rec_ind])
                    
                for eattr in evar_attrs:
                    oattr = "{:s}_ocb".format(eattr)
                    evar = getattr(pysat_data, eattr)[iser]
                    ocb_series[oattr][iser] = ocbscal.normal_evar(evar, \
                                                            unscaled_r, ref_r)
                for eattr in curl_evar_attrs:
                    oattr = "{:s}_ocb".format(eattr)
                    evar = getattr(pysat_data, eattr)[iser]
                    ocb_series[oattr][iser] = ocbscal.normal_curl_evar(evar, \
                                                            unscaled_r, ref_r)

            
            # Move to next line
            idat += 1

    # Update DataFrame
    for oattr in ocb_series:
        setattr(pysat_data, oattr, ocb_series[oattr])

    return ocb_series.keys()
