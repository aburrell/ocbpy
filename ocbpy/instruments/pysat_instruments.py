# -*- coding: utf-8 -*-
# Copyright (C) 2017 AGB
# Full license can be found in LICENSE.txt
# ---------------------------------------------------------------------------
""" Perform OCB gridding for appropriate instrument data loaded in pysat

Functions
---------
add_ocb_to_data()
    Add OCB coordinates to the pysat Instrument.DataFrame object
add_ocb_to_metadata()
    Update pysat Metadata with OCB information

Module
------
pysat is available at: http://github.com/pysat/pysat or pypi

"""
from __future__ import absolute_import, unicode_literals

import datetime as dt
import numpy as np

try:
    import pysat
except ImportError as ierr:
    err = ''.join(['unable to load the pysat modules; pysat is available at:',
                   '\nhttps://github.com/pysat/pysat'])
    raise ImportError("{:s}\n{:}".format(err, ierr))

import ocbpy
import ocbpy.ocb_scaling as ocbscal


def add_ocb_to_data(pysat_inst, mlat_name='', mlt_name='', evar_names=list(),
                    curl_evar_names=list(), vector_names=dict(),
                    hemisphere=0, ocb=None, ocbfile='default', instrument='',
                    max_sdiff=600, min_sectors=7, rcent_dev=8.0, max_r=23.0,
                    min_r=10.0):
    """ Coverts the location of pysat data into OCB coordinates

    Parameters
    ----------
    pysat_inst : (pysat.Instrument)
        pysat.Instrument class object containing magnetic coordinates
    mlat_name : (str)
        Instrument data key or column for magnetic latitudes (default='')
    mlt_name : (str)
        Instrument data key or column formagnetic longitudes (default='')
    evar_names : (list)
        List of Instrument data keys or columns pointing to measurements that
        are proportional to the electric field (E); e.g. ion drift (default=[])
    curl_evar_names : (list)
        List of Instrument data keys or columns pointing to measurements that
        are proportional to the curl of E; e.g. ion vorticity (default=[])
    vector_names : (dict)
        Dict of Instrument data keys or columns pointing to measurements that
        are vectors that are proportional to either E or the curl of E. The
        key should correspond to one of the values in the evar_names or
        curl_evar_names list.  If this is not done, a scaling function must be
        provided.  The value corresponding to each key must be a dict that
        indicates the names holding data needed to initialise the
        ocbpy.ocb_scaling.VectorData object (default={})
    hemisphere : (int)
        Hemisphere to process (can only do one at a time).  1=Northern,
        -1=Southern, 0=Determine from data (default=0)
    ocb : (OCBoundary or NoneType)
        OCBoundary object with data loaded from an OC boundary data file.
        If None, looks to ocbfile
    ocbfile : (str)
        file containing the required OC boundary data sorted by time, ignorned
        if OCBoundary object supplied (default='default')
    instrument : (str)
        Instrument providing the OCBoundaries.  Requires 'image' or 'ampere'
        if a file is provided.  If using filename='default', also accepts
        'amp', 'si12', 'si13', 'wic', and '' (default='')
    max_sdiff : (int)
        maximum seconds between OCB and data record in sec (default=600)
    min_sectors : (int)
        Minimum number of MLT sectors required for good OCB (default=7)
    rcent_dev : (float)
        Maximum number of degrees between the new centre and the AACGM pole
        (default=8.0)
    max_r : (float)
        Maximum radius for open-closed field line boundary in degrees
        (default=23.0)
    min_r : (float)
        Minimum radius for open-closed field line boundary in degrees
        (default=10.0)

    Raises
    ------
    ValueError
        If the pysat Instrument doesn't have the necessary data values or
        if the input provided is not a pysat Instrument.

    Notes
    -----
    This may be run on a pysat instrument or as a custom function
    (using 'modify') when loading pysat data.

    Example
    -------
    Example vector name input looks like:
    vector_names={'vel': {'aacgm_n': 'vel_n', 'aacgm_e': 'vel_e',
                          'dat_name': 'velocity', 'dat_units': 'm/s'},
                   'dat': {'aacgm_n': 'dat_n', 'aacgm_e': 'dat_e',
                           'scale_func': local_scale_func}}

    """

    # Test the input
    if not isinstance(pysat_inst, pysat.Instrument):
        raise ValueError('unknown class, expected pysat.Instrument')

    if mlat_name not in pysat_inst.data.columns:
        raise ValueError(
            'unknown magnetic latitude name {:}'.format(mlat_name))

    if mlt_name not in pysat_inst.data.columns:
        raise ValueError(
            'unknown magnetic local time name {:}'.format(mlt_name))

    # Test to see that the rest of the data names are present
    if not np.all([eattr in pysat_inst.data.columns or
                   eattr in vector_names.keys() for eattr in evar_names]):
        raise ValueError('at least one unknown E field name')

    if not np.all([eattr in pysat_inst.data.columns or
                   eattr in vector_names.keys() for eattr in curl_evar_names]):
        raise ValueError('at least one unknown E field name')

    # Format the new data column names
    olat_name = "{:s}_ocb".format(mlat_name)
    omlt_name = "{:s}_ocb".format(mlt_name)
    ocor_name = "r_corr_ocb"
    ocb_names = [olat_name, omlt_name, ocor_name]

    # Get a list of all necessary pysat data names
    pysat_names = [mlat_name, mlt_name]

    for pkey in evar_names:
        if pkey in pysat_inst.data.columns and pkey not in pysat_names:
            pysat_names.append(pkey)

    for pkey in curl_evar_names:
        if pkey in pysat_inst.data.columns and pkey not in pysat_names:
            pysat_names.append(pkey)

    # Test the vector names to ensure that enough information
    # was provided and that it exists in the Instrument object
    #
    # Continue adding to pysat names
    nvect = len(vector_names.keys())
    vector_attrs = dict()
    if nvect > 0:
        vector_reqs = ["aacgm_n", "aacgm_e", "aacgm_z"]

        for eattr in vector_names.keys():
            vdim = 0

            if eattr in evar_names:
                vector_names[eattr]["scale_func"] = ocbscal.normal_evar
                evar_names.pop(evar_names.index(eattr))
            elif eattr in curl_evar_names:
                vector_names[eattr]["scale_func"] = ocbscal.normal_curl_evar
                curl_evar_names.pop(curl_evar_names.index(eattr))
            else:
                if 'scale_func' not in vector_names[eattr]:
                    raise ValueError('missing scaling function for {:}'.format(
                        eattr))

            oattr = "{:s}_ocb".format(eattr)
            if oattr not in ocb_names:
                ocb_names.append(oattr)

            for vinit in vector_names[eattr].keys():
                if vinit in vector_reqs:
                    if(vector_names[eattr][vinit] not in
                       pysat_inst.data.columns):
                        raise ValueError("unknown vector name {:}".format(
                            vector_names[eattr][vinit]))
                    else:
                        if vector_names[eattr][vinit] not in pysat_names:
                            pysat_names.append(vector_names[eattr][vinit])

                        if oattr not in vector_attrs.keys():
                            vector_attrs[oattr] = list()
                        vector_attrs[oattr].append(vector_names[eattr][vinit])
                    vdim += 1

    # Append the remaining OCB output names
    for eattr in evar_names:
        ocb_names.append("{:s}_ocb".format(eattr))

    for eattr in curl_evar_names:
        ocb_names.append("{:s}_ocb".format(eattr))

    # Extract the AACGM locations
    aacgm_lat = np.array(pysat_inst[mlat_name])
    aacgm_mlt = np.array(pysat_inst[mlt_name])
    ndat = len(aacgm_lat)

    # Load the OCB data for the data period, if desired
    if ocb is None or not isinstance(ocb, ocbpy.ocboundary.OCBoundary):
        dstart = pysat_inst.index[0] - dt.timedelta(seconds=max_sdiff+1)
        dend = pysat_inst.index[-1] + dt.timedelta(seconds=max_sdiff+1)

        # If hemisphere isn't specified, set it here
        if hemisphere == 0:
            hemisphere = np.sign(np.nanmax(aacgm_lat))

            # Ensure that all data is in the same hemisphere
            if hemisphere == 0:
                hemisphere = np.sign(np.nanmin(aacgm_lat))
            elif hemisphere != np.sign(np.nanmin(aacgm_lat)):
                raise ValueError("".join(["cannot process observations from "
                                          "both hemispheres at the same time;"
                                          "set hemisphere=+/-1 to choose."]))

        # Initialize the OCBoundary object
        ocb = ocbpy.OCBoundary(ocbfile, stime=dstart, etime=dend,
                               instrument=instrument, hemisphere=hemisphere)
    elif hemisphere == 0:
        # If the OCBoundary object is specified and hemisphere isn't use
        # the OCBoundary object to specify the hemisphere
        hemisphere = ocb.hemisphere

    # Ensure all data is from one hemisphere and is finite
    dat_ind = np.where((np.sign(aacgm_lat) == hemisphere)
                       & (np.isfinite(
                           np.max(pysat_inst.data.loc[:, pysat_names].values,
                                  axis=1))))[0]

    # Test the OCB data
    if ocb.filename is None or ocb.records == 0:
        ocbpy.logger.error("no data in OCB file {:}".format(ocb.filename))
        return

    # Initialise the OCB output
    ocb_output = dict()
    for oattr in ocb_names:
        eattr = oattr[:-4]
        if eattr in vector_names.keys():
            ocb_output[oattr] = np.empty(shape=aacgm_lat.shape,
                                         dtype=ocbscal.VectorData)
        else:
            ocb_output[oattr] = np.full(aacgm_lat.shape, np.nan, dtype=float)

    # Cycle through the data, matching data and OCB records
    idat = 0
    ndat = len(dat_ind)
    ref_r = 90.0 - abs(ocb.boundary_lat)
    while idat < ndat and ocb.rec_ind < ocb.records:
        idat = ocbpy.match_data_ocb(ocb, pysat_inst.index[dat_ind], idat=idat,
                                    max_tol=max_sdiff, min_sectors=min_sectors,
                                    rcent_dev=rcent_dev, max_r=max_r,
                                    min_r=min_r)

        if idat < ndat and ocb.rec_ind < ocb.records:
            iout = dat_ind[idat]

            # Get the OCB coordinates
            (ocb_output[olat_name][iout], ocb_output[omlt_name][iout],
             ocb_output[ocor_name][iout]) = ocb.normal_coord(aacgm_lat[iout],
                                                             aacgm_mlt[iout])

            # Scale and orient the vector values
            if nvect > 0:
                # Set this value's AACGM vector values
                vector_default = {"ocb_lat": ocb_output[olat_name][iout],
                                  "ocb_mlt": ocb_output[omlt_name][iout],
                                  "r_corr": ocb_output[ocor_name][iout],
                                  "aacgm_n": 0.0, "aacgm_e": 0.0,
                                  "aacgm_z": 0.0, "aacgm_mag": np.nan,
                                  "dat_name": None, "dat_units": None,
                                  "scale_func": None}
                vector_init = dict(vector_default)

                for eattr in vector_names.keys():
                    oattr = "{:s}_ocb".format(eattr)
                    for ikey in vector_names[eattr].keys():
                        # Not all vector names are DataFrame names
                        if(vector_names[eattr][ikey]
                           in pysat_inst.data.columns):
                            vector_init[ikey] = \
                                pysat_inst[vector_names[eattr][ikey]][iout]
                        else:
                            vector_init[ikey] = vector_names[eattr][ikey]

                    ocb_output[oattr][iout] = ocbscal.VectorData(
                        iout, ocb.rec_ind, aacgm_lat[iout], aacgm_mlt[iout],
                        **vector_init)
                    ocb_output[oattr][iout].set_ocb(ocb)

            unscaled_r = ocb.r[ocb.rec_ind] + ocb_output[ocor_name][iout]

            # Scale the E-field proportional variables
            for eattr in evar_names:
                oattr = "{:s}_ocb".format(eattr)
                evar = pysat_inst[eattr][iout]
                ocb_output[oattr][iout] = ocbscal.normal_evar(evar, unscaled_r,
                                                              ref_r)

            # Scale the variables proportial to the curl of the E-field
            for eattr in curl_evar_names:
                oattr = "{:s}_ocb".format(eattr)
                evar = pysat_inst[eattr][iout]
                ocb_output[oattr][iout] = ocbscal.normal_curl_evar(
                    evar, unscaled_r, ref_r)

            # Move to next line
            idat += 1

    # Update the pysat Instrument
    for oattr in ocb_output:
        # The update procedure is different for pandas and xarray
        if pysat_inst.pandas_format:
            set_data = ocb_output[oattr]
            rename_in = {"columns": {'ocb_key': oattr}, "inplace": True}
        else:
            set_data = (pysat_inst.data.coords.keys(), ocb_output[oattr])
            rename_in = {"name_dict": {'ocb_key': oattr}, "inplace": True}

        # Add the OCB data to the pysat Instrument data object
        pysat_inst.data = pysat_inst.data.assign(ocb_key=set_data)
        pysat_inst.data.rename(**rename_in)

        # Update the pysat Metadata
        eattr = oattr[:-4]
        notes = "OCB obtained from {:} data in file ".format(ocb.instrument)
        notes += "{:} using a boundary latitude of ".format(ocb.filename)
        notes += "{:.2f}".format(ocb.boundary_lat)

        if eattr in vector_names.keys():
            if vector_names[eattr]['scale_func'] is None:
                func_name = "None"
            else:
                func_name = vector_names[eattr]['scale_func'].__name__
            notes += " and was scaled using {:}".format(func_name)
            eattr = vector_attrs[oattr][0]
            isvector = True
        else:
            isvector = False

        add_ocb_to_metadata(pysat_inst, oattr, eattr, notes=notes,
                            isvector=isvector)

    return


def add_ocb_to_metadata(pysat_inst, ocb_name, pysat_name, overwrite=False,
                        notes='', isvector=False):
    """ Update pysat metadata for OCB data

    Parameters
    ----------
    pysat_inst : (pysat.Instrument)
        pysat.Instrument class object containing magnetic coordinates
    ocb_name : (str)
        Data column name for OCB data
    pysat_name : (str)
        Data column name for non-OCB version of this data
    overwrite : (boolean)
        Overwrite existing metadata, if present (default=False)
    notes : (str)
        Notes about this OCB data (default='')
    isvector : (boolean)
        Is this vector data or not (default=False)

    """

    # Test the input
    if not isinstance(pysat_inst, pysat.Instrument):
        raise ValueError('unknown class, expected pysat.Instrument')

    if not overwrite and ocb_name in pysat_inst.meta.data.index:
        ocbpy.logger.warning("OCB data already has metadata")

    else:
        if pysat_name not in pysat_inst.meta.data.index:
            name = ("OCB_" + ocb_name.split("_ocb")[0]).replace("_", " ")
            new_meta = {pysat_inst.fill_label: np.nan,
                        pysat_inst.name_label: name,
                        pysat_inst.desc_label: name.replace(
                            "OCB", "Open Closed field-line Boundary"),
                        pysat_inst.plot_label: name,
                        pysat_inst.axis_label: name}
        elif isvector:
            name = ("OCB_" + ocb_name.split("_ocb")[0]).replace("_", " ")
            new_meta = {pysat_inst.fill_label: None,
                        pysat_inst.name_label: name,
                        pysat_inst.desc_label: "".join([
                            "Open Closed field-line Boundary vector ",
                            pysat_inst.meta[pysat_name][
                                pysat_inst.desc_label]]),
                        pysat_inst.units_label:
                        pysat_inst.meta[pysat_name][pysat_inst.units_label],
                        pysat_inst.plot_label: name,
                        pysat_inst.axis_label: name}
        else:
            # Initialize with old values
            labels = list(pysat_inst.meta.data.keys())
            new_meta = {ll: pysat_inst.meta[pysat_name][ll] for ll in labels}

            # Update certain categories with OCB information
            new_meta[pysat_inst.fill_label] = np.nan
            new_meta[pysat_inst.name_label] = "".join([
                "OCB ", new_meta[pysat_inst.name_label]])
            new_meta[pysat_inst.desc_label] = "".join([
                "Open Closed field-line Boundary ",
                new_meta[pysat_inst.desc_label]])
            new_meta[pysat_inst.plot_label] = "".join([
                "OCB ", new_meta[pysat_inst.plot_label]])
            new_meta[pysat_inst.axis_label] = "".join([
                "OCB ", new_meta[pysat_inst.axis_label]])

        # Set the notes
        new_meta[pysat_inst.notes_label] = notes

        # Set new metadata
        pysat_inst.meta[ocb_name] = new_meta

    return
