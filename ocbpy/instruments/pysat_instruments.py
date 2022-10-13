# -*- coding: utf-8 -*-
# Copyright (C) 2017 AGB
# Full license can be found in LICENSE.txt
# ---------------------------------------------------------------------------
"""Perform OCB gridding for appropriate instrument data loaded in pysat.

Notes
-----
pysat is available at: http://github.com/pysat/pysat or pypi

"""

import datetime as dt
import numpy as np
import warnings

try:
    import pysat
except ImportError as ierr:
    err = ''.join(['unable to load the pysat modules; pysat is available at:',
                   '\nhttps://github.com/pysat/pysat'])
    raise ImportError("{:s}\n{:}".format(err, ierr))

import ocbpy
import ocbpy.ocb_scaling as ocbscal


def add_ocb_to_data(pysat_inst, mlat_name='', mlt_name='', evar_names=None,
                    curl_evar_names=None, vector_names=None, hemisphere=0,
                    ocb=None, ocbfile='default', instrument='', max_sdiff=60,
                    min_merit=None, max_merit=None, **kwargs):
    """Covert the location of pysat data into OCB, EAB, or Dual coordinates.

    Parameters
    ----------
    pysat_inst : pysat.Instrument
        pysat.Instrument class object containing magnetic coordinates
    mlat_name : str
        Instrument data key or column for magnetic latitudes (default='')
    mlt_name : str
        Instrument data key or column for magnetic local times (default='')
    evar_names : list or NoneType
        List of Instrument data keys or columns pointing to measurements that
        are proportional to the electric field (E); e.g. ion drift
        (default=None)
    curl_evar_names : list or NoneType
        List of Instrument data keys or columns pointing to measurements that
        are proportional to the curl of E; e.g. ion vorticity (default=None)
    vector_names : dict or NoneType
        Dict of Instrument data keys or columns pointing to measurements that
        are vectors that are proportional to either E or the curl of E. The
        key should correspond to one of the values in the evar_names or
        curl_evar_names list.  If this is not done, a scaling function must be
        provided.  The value corresponding to each key must be a dict that
        indicates the names holding data needed to initialise the
        ocbpy.ocb_scaling.VectorData object (default=None)
    hemisphere : int
        Hemisphere to process (can only do one at a time).  1=Northern,
        -1=Southern, 0=Determine from data (default=0)
    ocb : ocbpy.OCBoundary, ocbpy.DualBoundary, or NoneType
        OCBoundary or DualBoundary object with data loaded already. If None,
        looks to `ocbfile` and creates an OCBoundary object. (default=None)
    ocbfile : str
        file containing the required OC boundary data sorted by time, ignorned
        if OCBoundary object supplied (default='default')
    instrument : str
        Instrument providing the OCBoundaries.  Requires 'image' or 'ampere'
        if a file is provided.  If using filename='default', also accepts
        'amp', 'si12', 'si13', 'wic', and '' (default='')
    max_sdiff : int
        maximum seconds between OCB and data record in sec (default=60)
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
        be removed in version 0.3.1+  (default=7)
    rcent_dev : float
        Maximum number of degrees between the new centre and the AACGM pole.
        Deprecated, will be removed in version 0.3.1+ (default=8.0)
    max_r : float
        Maximum radius for open-closed field line boundary in degrees.
        Deprecated, will be removed in version 0.3.1+ (default=23.0)
    min_r : float
        Minimum radius for open-closed field line boundary in degrees.
        Deprecated, will be removed in version 0.3.1+ (default=10.0)

    Raises
    ------
    ValueError
        If the pysat Instrument doesn't have the necessary data values or
        if the input provided is not a pysat Instrument.

    Notes
    -----
    This may be run on a pysat instrument or as a custom function when loading
    pysat data.

    Examples
    --------
    ::

       # Example vector name input looks like:
       vector_names={'vel': {'aacgm_n': 'vel_n', 'aacgm_e': 'vel_e',
                             'dat_name': 'velocity', 'dat_units': 'm/s'},
                      'dat': {'aacgm_n': 'dat_n', 'aacgm_e': 'dat_e',
                              'scale_func': local_scale_func}}

    """

    # Test the input
    if evar_names is None:
        evar_names = []

    if curl_evar_names is None:
        curl_evar_names = []

    if vector_names is None:
        vector_names = {}

    if not isinstance(pysat_inst, pysat.Instrument):
        raise ValueError('unknown class, expected pysat.Instrument')

    if mlat_name not in pysat_inst.variables:
        raise ValueError(
            'unknown magnetic latitude name {:}'.format(mlat_name))

    if mlt_name not in pysat_inst.variables:
        raise ValueError(
            'unknown magnetic local time name {:}'.format(mlt_name))

    # Test to see that the rest of the data names are present
    if not np.all([eattr in pysat_inst.variables
                   or eattr in vector_names.keys() for eattr in evar_names]):
        raise ValueError('at least one unknown E field name')

    if not np.all([eattr in pysat_inst.variables
                   or eattr in vector_names.keys()
                   for eattr in curl_evar_names]):
        raise ValueError('at least one unknown E field name')

    # Format the new data column names
    olat_name = "{:s}_ocb".format(mlat_name)
    omlt_name = "{:s}_ocb".format(mlt_name)
    ocor_name = "r_corr_ocb"
    ocb_names = [olat_name, omlt_name, ocor_name]
    ocb_vect_attrs = ['ocb_n', 'ocb_e', 'ocb_z', 'ocb_mag', 'unscaled_r',
                      'scaled_r']

    # Get a list of all necessary pysat data names
    pysat_names = [mlat_name, mlt_name]

    for pkey in evar_names:
        if pkey in pysat_inst.variables and pkey not in pysat_names:
            pysat_names.append(pkey)

    for pkey in curl_evar_names:
        if pkey in pysat_inst.variables and pkey not in pysat_names:
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
                    if vector_names[eattr][vinit] not in pysat_inst.variables:
                        raise ValueError("unknown vector name {:}".format(
                            vector_names[eattr][vinit]))
                    else:
                        if vector_names[eattr][vinit] not in pysat_names:
                            pysat_names.append(vector_names[eattr][vinit])

                        if oattr not in vector_attrs.keys():
                            vector_attrs[oattr] = list()
                        vector_attrs[oattr].append(vector_names[eattr][vinit])
                    vdim += 1

    # Determine how many of the pysat names are variables
    pysat_var_names = len(pysat_names)
    if not pysat_inst.pandas_format:
        for pyname in pysat_names:
            if pyname in pysat_inst.data.coords:
                pysat_var_names -= 1

    # Append the remaining OCB output names
    for eattr in evar_names:
        ocb_names.append("{:s}_ocb".format(eattr))

    for eattr in curl_evar_names:
        ocb_names.append("{:s}_ocb".format(eattr))

    # Extract the magnetic locations as numpy arrays
    aacgm_lat = np.array(pysat_inst[mlat_name])
    aacgm_mlt = np.array(pysat_inst[mlt_name])
    ndat = len(aacgm_lat)

    # Load the OCB data for the data period, if desired
    if ocb is None or (not isinstance(ocb, ocbpy.OCBoundary)
                       and not isinstance(ocb, ocbpy.DualBoundary)):
        dstart = pysat_inst.index[0] - dt.timedelta(seconds=max_sdiff + 1)
        dend = pysat_inst.index[-1] + dt.timedelta(seconds=max_sdiff + 1)

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
    if pysat_inst.pandas_format:
        finite_mask = ((np.sign(aacgm_lat) == hemisphere)
                       & np.isfinite(pysat_inst[:, pysat_names].max(axis=1)))
        dat_coords = [pysat_inst.index.name]
        combo_shape = [pysat_inst.index.shape[0]]
    else:
        nan_inst = pysat_inst.data.where(np.sign(pysat_inst.data[mlat_name])
                                         >= hemisphere)
        finite_mask = np.isfinite(nan_inst[pysat_names].to_array().max(
            'variable'))
        dat_coords = [coord for coord in pysat_inst[pysat_names].coords]
    dat_ind = np.where(finite_mask)

    # Test the OCB data
    if ocb.records == 0:
        ocbpy.logger.error("no data in Boundary file(s)")
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

    # Ensure the MLT and MLat data are the same shape
    if(aacgm_lat.shape != aacgm_mlt.shape
       or aacgm_lat.shape[0] != pysat_inst.index.shape[0]):
        ocb_coords = [mlt_coord for mlt_coord
                      in pysat_inst[mlt_name].coords.keys()]
        if pysat_inst.index.name in ocb_coords:
            combo_shape = list(aacgm_mlt.shape)
        else:
            # Ensure MLT has time dependence
            ocb_coords.insert(0, pysat_inst.index.name)
            combo_shape = [pysat_inst.index.shape[0]]
            combo_shape.extend(list(aacgm_mlt.shape))
            out_mlt, _ = np.meshgrid(aacgm_mlt, pysat_inst.index)

            if out_mlt.shape != combo_shape:
                aacgm_mlt = out_mlt.reshape(combo_shape)

        # Expand the coordinates if the MLat coordinates are not the
        # same as the MLT coordinates
        for lat_coord in pysat_inst[mlat_name].coords:
            if lat_coord not in pysat_inst[mlt_name].coords:
                combo_shape.append(pysat_inst[lat_coord].shape[0])
                ocb_coords.append(lat_coord)

        # Reshape the data
        out_lat, out_mlt = np.meshgrid(aacgm_lat, aacgm_mlt)
        aacgm_lat = out_lat.reshape(combo_shape)
        aacgm_mlt = out_mlt.reshape(combo_shape)
    else:
        ocb_coords = [pysat_inst.index.name]

    # See if the data index has more dimensions than the coordinates
    if len(dat_ind) > len(ocb_coords):
        combo_shape = list(aacgm_lat.shape)
        for dcoord in dat_coords:
            if dcoord not in ocb_coords:
                ocb_coords.append(dcoord)
                combo_shape.append(pysat_inst[dcoord].shape[0])

        # Reverse and transpose the arrays
        combo_shape.reverse()
        out_lat = np.full(shape=combo_shape, fill_value=aacgm_lat.transpose())
        out_mlt = np.full(shape=combo_shape, fill_value=aacgm_mlt.transpose())
        aacgm_lat = out_lat.transpose()
        aacgm_mlt = out_mlt.transpose()

    # Initialise the OCB output
    ocb_output = dict()
    for oattr in ocb_names:
        eattr = oattr[:-4]
        if eattr in vector_names.keys():
            for vattr in ocb_vect_attrs:
                ovattr = '_'.join([oattr, vattr])
                ovattr = ovattr.replace('ocb_ocb_', 'ocb_')
                ocb_output[ovattr] = np.full(aacgm_lat.shape, np.nan,
                                             dtype=float)
        else:
            ocb_output[oattr] = np.full(aacgm_lat.shape, np.nan, dtype=float)

    # Cycle through the data, matching data and OCB records
    idat = 0
    ndat = len(dat_ind[0])
    if hasattr(ocb, "boundary_lat"):
        ref_r = 90.0 - abs(ocb.boundary_lat)
    else:
        ref_r = 90.0 - abs(ocb.ocb.boundary_lat)

    while idat < ndat and ocb.rec_ind < ocb.records:
        idat = ocbpy.match_data_ocb(ocb, pysat_inst.index[dat_ind[0]],
                                    idat=idat, max_tol=max_sdiff,
                                    min_merit=min_merit, max_merit=max_merit,
                                    **kwargs)

        if idat < ndat and ocb.rec_ind < ocb.records:
            # Find all the indices with the same time
            time_ind = np.where(pysat_inst.index[dat_ind[0]]
                                == pysat_inst.index[dat_ind[0]][idat])
            idat = time_ind[0][-1]

            if len(dat_ind) > 1:
                iout = tuple(dind[time_ind] for dind in dat_ind)
                if pysat_var_names > 1:
                    # If there is more than one variable, need to downselect
                    time_sel = pysat_inst[pysat_names].to_array().max(
                        'variable')
                else:
                    time_sel = pysat_inst[pysat_names]

                time_mask = np.isfinite(time_sel.where(
                    finite_mask & (pysat_inst[pysat_inst.index.name]
                                   == pysat_inst.index[dat_ind[0]][idat])))
                vind = iout[0]
            else:
                iout = dat_ind[0][time_ind]
                vind = iout
                time_mask = None

            # Get the OCB coordinates
            nout = ocb.normal_coord(aacgm_lat[iout], aacgm_mlt[iout])

            if len(nout) == 3:
                ocb_output[olat_name][iout] = nout[0]
                ocb_output[omlt_name][iout] = nout[1]
                ocb_output[ocor_name][iout] = nout[2]
            else:
                ocb_output[olat_name][iout] = nout[0]
                ocb_output[omlt_name][iout] = nout[1]
                ocb_output[ocor_name][iout] = nout[3]

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
                        vname = vector_names[eattr][ikey]
                        if vname in pysat_inst.variables:
                            # Test to see if the input is appropriately shaped
                            if(not pysat_inst.pandas_format
                               and len(ocb_coords) > len(pysat_inst[
                                   vname].coords)):
                                raise ValueError(''.join([
                                    'vector variables must all have the same',
                                    ' dimensions']))

                            if time_mask is None:
                                vector_init[ikey] = pysat_inst[vname][iout]
                            else:
                                vector_init[ikey] = pysat_inst[vname].where(
                                    time_mask, drop=True).values.flatten()
                        else:
                            vector_init[ikey] = vname

                    # Perform the vector scaling
                    vout = ocbscal.VectorData(vind, ocb.rec_ind,
                                              aacgm_lat[iout],
                                              aacgm_mlt[iout],
                                              **vector_init)
                    vout.set_ocb(ocb)

                    # Assign the vector attributes to the output
                    for vattr in ocb_vect_attrs:
                        ovattr = '_'.join([oattr, vattr])
                        ovattr = ovattr.replace('ocb_ocb_', 'ocb_')
                        ocb_output[ovattr][iout] = getattr(vout, vattr)

            if hasattr(ocb, "ocb"):
                unscaled_r = ocb.ocb.r[ocb.ocb.rec_ind] + ocb_output[
                    ocor_name][iout]
            else:
                unscaled_r = ocb.r[ocb.rec_ind] + ocb_output[
                    ocor_name][iout]

            # Scale the E-field proportional variables
            for eattr in evar_names:
                oattr = "{:s}_ocb".format(eattr)
                if time_mask is None:
                    evar = pysat_inst[eattr][iout]
                else:
                    evar = pysat_inst[eattr].where(time_mask,
                                                   drop=True).values.flatten()
                ocb_output[oattr][iout] = ocbscal.normal_evar(
                    evar, unscaled_r, ref_r)

            # Scale the variables proportial to the curl of the E-field
            for eattr in curl_evar_names:
                oattr = "{:s}_ocb".format(eattr)
                if time_mask is None:
                    evar = pysat_inst[eattr][iout]
                else:
                    evar = pysat_inst[eattr].where(time_mask,
                                                   drop=True).values.flatten()
                ocb_output[oattr][iout] = ocbscal.normal_curl_evar(
                    evar, unscaled_r, ref_r)

            # Move to next line
            idat += 1

    # Update the pysat Instrument
    for oattr in ocb_output.keys():
        # The update procedure is different for pandas and xarray
        if pysat_inst.pandas_format:
            set_data = {oattr: ocb_output[oattr]}
            pysat_inst.data = pysat_inst.data.assign(**set_data)
        else:
            set_data = {oattr: (ocb_coords, ocb_output[oattr])}
            pysat_inst.data = pysat_inst.data.assign(set_data)

        # Update the pysat Metadata
        eattr = oattr.split('_ocb')[0]
        if hasattr(ocb, "instrument"):
            notes = "".join(["OCB obtained from ", ocb.instrument,
                             " data in file ", ocb.filename,
                             "using a boundary latitude of ",
                             "{:.2f}".format(ocb.boundary_lat)])
        else:
            notes = "".join(["OCB obtained from ", ocb.ocb.instrument, " data",
                             " in file ", ocb.ocb.filename, " using a ",
                             "boundary latitude of ",
                             "{:.2f}".format(ocb.ocb.boundary_lat), " and EAB",
                             "EAB obtained from ", ocb.eab.instrument,
                             " data in file ", ocb.eab.filename, "using a ",
                             "boundary latitude of ",
                             "{:.2f}".format(ocb.eab.boundary_lat)])

        if eattr in vector_names.keys():
            if vector_names[eattr]['scale_func'] is None:
                func_name = "None"
            else:
                func_name = vector_names[eattr]['scale_func'].__name__
            notes += " and was scaled using {:}".format(func_name)
            eattr = vector_attrs['_'.join([eattr, 'ocb'])][0]
            isvector = True
        else:
            isvector = False

        add_ocb_to_metadata(pysat_inst, oattr, eattr, notes=notes,
                            isvector=isvector)

    return


def add_ocb_to_metadata(pysat_inst, ocb_name, pysat_name, overwrite=False,
                        notes='', isvector=False):
    """Update pysat metadata for OCB data.

    Parameters
    ----------
    pysat_inst : pysat.Instrument
        pysat.Instrument class object containing magnetic coordinates
    ocb_name : str
        Data column name for OCB data
    pysat_name : str
        Data column name for non-OCB version of this data
    overwrite : bool
        Overwrite existing metadata, if present (default=False)
    notes : str)
        Notes about this OCB data (default='')
    isvector : bool
        Is this vector data or not (default=False)

    Raises
    ------
    ValueError
        If input pysat Instrument object is the wrong class

    """

    # Test the input
    if not isinstance(pysat_inst, pysat.Instrument):
        raise ValueError('unknown class, expected pysat.Instrument')

    if not overwrite and ocb_name in pysat_inst.meta.data.index:
        ocbpy.logger.warning("OCB data already has metadata")

    else:
        if pysat_name not in pysat_inst.meta.data.index:
            name = ("OCB_" + ocb_name.split("_ocb")[0]).replace("_", " ")
            new_meta = {pysat_inst.meta.labels.fill_val: np.nan,
                        pysat_inst.meta.labels.name: name,
                        pysat_inst.meta.labels.desc: name.replace(
                            "OCB", "Open Closed field-line Boundary"),
                        pysat_inst.meta.labels.min_val: -np.inf,
                        pysat_inst.meta.labels.max_val: np.inf}
        elif isvector:
            name = ("OCB_" + ocb_name.split("_ocb")[0]).replace("_", " ")
            new_meta = {pysat_inst.meta.labels.fill_val: np.nan,
                        pysat_inst.meta.labels.name: name,
                        pysat_inst.meta.labels.desc: "".join([
                            "Open Closed field-line Boundary vector ",
                            pysat_inst.meta[pysat_name][
                                pysat_inst.meta.labels.desc]]),
                        pysat_inst.meta.labels.units: pysat_inst.meta[
                            pysat_name][pysat_inst.meta.labels.units],
                        pysat_inst.meta.labels.min_val: pysat_inst.meta[
                            pysat_name][pysat_inst.meta.labels.min_val],
                        pysat_inst.meta.labels.max_val: pysat_inst.meta[
                            pysat_name][pysat_inst.meta.labels.max_val]}
        else:
            # Initialize with old values
            labels = [ll for ll in pysat_inst.meta.data.keys()]
            new_meta = {ll: pysat_inst.meta[pysat_name][ll] for ll in labels}

            # Update certain categories with OCB information
            new_meta[pysat_inst.meta.labels.fill_val] = np.nan
            new_meta[pysat_inst.meta.labels.name] = "".join([
                "OCB ", new_meta[pysat_inst.meta.labels.name]])
            new_meta[pysat_inst.meta.labels.desc] = "".join([
                "Open Closed field-line Boundary ",
                new_meta[pysat_inst.meta.labels.desc]])

        # Set the notes
        new_meta[pysat_inst.meta.labels.notes] = notes

        # Set new metadata
        pysat_inst.meta[ocb_name] = new_meta

    return
