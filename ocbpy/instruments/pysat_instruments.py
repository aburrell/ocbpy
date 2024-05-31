#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DOI: 10.5281/zenodo.1179230
# Full license can be found in License.md
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ---------------------------------------------------------------------------
"""Perform OCB gridding for appropriate instrument data loaded in pysat.

Notes
-----
pysat is available at: http://github.com/pysat/pysat or pypi

"""

import datetime as dt
import inspect
import numpy as np
import os

try:
    import pysat
except ImportError as ierr:
    err = ''.join(['unable to load the pysat modules; pysat is available at:',
                   '\nhttps://github.com/pysat/pysat'])
    raise ImportError("{:s}\n{:}".format(err, ierr))

import ocbpy
import ocbpy.ocb_scaling as ocbscal


def add_ocb_to_data(pysat_inst, mlat_name='', mlt_name='', height_name='',
                    evar_names=None, curl_evar_names=None, vector_names=None,
                    height=350.0, hemisphere=0, ocb=None, ocbfile='ocb',
                    instrument='', max_sdiff=60, min_merit=None, max_merit=None,
                    loc_coord='magnetic', vect_coord='magnetic', **kwargs):
    """Covert the location of pysat data into OCB, EAB, or Dual coordinates.

    Parameters
    ----------
    pysat_inst : pysat.Instrument
        pysat.Instrument class object containing magnetic coordinates
    mlat_name : str
        Instrument data key or column for latitudes (default='')
    mlt_name : str
        Instrument data key or column for local times (default='')
    height_name : str
        Instrument data key or column for altitude (default='')
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
    height : float or array-like
        Altitude value(s) to use if no height variable is provided by
        `height_name` (default=350.0)
    hemisphere : int
        Hemisphere to process (can only do one at a time).  1=Northern,
        -1=Southern, 0=Determine from data (default=0)
    ocb : ocbpy.OCBoundary, ocbpy.DualBoundary, or NoneType
        OCBoundary or DualBoundary object with data loaded already. If None,
        looks to `ocbfile` and creates an OCBoundary object. (default=None)
    ocbfile : str
        file containing the required OC boundary data sorted by time, ignorned
        if OCBoundary object supplied. To use the default for a boundary type,
        supply the desired boundary type; 'eab', 'ocb', or 'dual'.
        (default='ocb')
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
    loc_coord : str
        Name of the coordinate system for `mlat_name` and `mlt_name`; one of
        'magnetic', 'geocentric', or 'geodetic'. If not 'magnetic',
        `height_name` or `height` will be used to convert the data to magnetic
        coordinates. (default='magnetic')
    vect_coord : str
        Name of the coordinate system for `vect_n` and `vect_e`; one of
        'magnetic', 'geocentric', or 'geodetic'. If not 'magnetic',
        `height_name` or `height` will be used to convert the data to magnetic
        coordinates. (default='magnetic')
    kwargs : dict
        Dict with optional selection criteria or criteria for initializing a
        DualBoundary class object (in combination with `ocb=None` and
        `ocbfile='dual'`).  For the optional selection criteria, the key should
        correspond to a data attribute and the value must be a tuple with the
        first value specifying 'max', 'min', 'maxeq', 'mineq', or 'equal' and
        the second value specifying the value to use in the comparison.

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
       vector_names={'vel': {'vect_n': 'vel_n', 'vect_e': 'vel_e',
                             'dat_name': 'velocity', 'dat_units': 'm/s'},
                      'dat': {'vect_n': 'dat_n', 'vect_e': 'dat_e',
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

    # Ensure the correct data format
    max_sdiff = int(max_sdiff)

    # Extract the locations as numpy arrays
    lat = np.array(pysat_inst[mlat_name])
    lt = np.array(pysat_inst[mlt_name])
    ndat = len(lat)

    # Load the OCB data for the data period, if desired
    if ocb is None or (not isinstance(ocb, ocbpy.OCBoundary)
                       and not isinstance(ocb, ocbpy.EABoundary)
                       and not isinstance(ocb, ocbpy.DualBoundary)):
        dstart = pysat_inst.index[0] - dt.timedelta(seconds=max_sdiff + 1)
        dend = pysat_inst.index[-1] + dt.timedelta(seconds=max_sdiff + 1)

        # If hemisphere isn't specified, set it here
        if hemisphere == 0:
            hemisphere = np.sign(np.nanmax(lat))

            # Ensure that all data is in the same hemisphere
            if hemisphere == 0:
                hemisphere = np.sign(np.nanmin(lat))
            elif hemisphere != np.sign(np.nanmin(lat)):
                raise ValueError("".join(["cannot process observations from "
                                          "both hemispheres at the same time;"
                                          "set hemisphere=+/-1 to choose."]))

        # Determine the boundary type by filename, if possible
        if ocbfile is None:
            fileroot = ""
        else:
            fileroot = os.path.split(ocbfile)[-1].lower()

            if ocbfile.lower() in ['eab', 'ocb', 'dual']:
                ocbfile = "default"

        if fileroot.find("ocb") > 0 and fileroot.find("eab") < 0:
            # Initialize the OCBoundary object
            ocb = ocbpy.OCBoundary(ocbfile, stime=dstart, etime=dend,
                                   instrument=instrument, hemisphere=hemisphere)
        elif fileroot.find("ocb") < 0 and fileroot.find("eab") > 0:
            # Initialize the EABoundary object
            ocb = ocbpy.EABoundary(ocbfile, stime=dstart, etime=dend,
                                   instrument=instrument, hemisphere=hemisphere)
        elif fileroot == 'dual':
            # This works by assigning default to both, or allowing additional
            # inputs through `kwargs`
            init_keys = {'hemisphere': hemisphere, 'stime': dstart,
                         'etime': dend}
            if len(kwargs.keys()) > 0:
                sig = inspect.getfullargspec(ocbpy.DualBoundary.__init__)
                for key in kwargs.keys():
                    if key in sig.args:
                        init_keys[key] = kwargs[key]

                # Clean DualBoundary kwargs from kwarg input
                for key in init_keys:
                    if key in kwargs.keys():
                        del kwargs[key]

            if 'ocb_instrument' not in init_keys.keys():
                init_keys['ocb_instrument'] = instrument

            if 'eab_instrument' not in init_keys.keys():
                init_keys['eab_instrument'] = instrument

            # Initialize the dual-boundary object
            ocb = ocbpy.DualBoundary(**init_keys)
        else:
            # Can't determine desired boundary type
            raise ValueError("".join(["can't determine desired boundary type ",
                                      "from filename: ", repr(ocbfile)]))

    elif hemisphere == 0:
        # If the OCBoundary object is specified and hemisphere isn't use
        # the OCBoundary object to specify the hemisphere
        hemisphere = ocb.hemisphere

    # Format the new data column names
    bname = ocb.__class__.__name__.split('oundary')[0].lower()
    olat_name = "_".join([mlat_name, bname])
    omlt_name = "_".join([mlt_name, bname])
    ocor_name = "_".join(["r", "corr", bname])
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
        vector_reqs = ["vect_n", "vect_e", "vect_z"]

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

            oattr = "{:s}_{:s}".format(eattr, bname)
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
        ocb_names.append("{:s}_{:s}".format(eattr, bname))

    for eattr in curl_evar_names:
        ocb_names.append("{:s}_{:s}".format(eattr, bname))

    # Extract the height, if possible
    if height_name in pysat_inst.variables:
        height = np.array(pysat_inst[height_name])
    else:
        height = np.asarray(height)

    # Ensure all data is from one hemisphere and is finite
    if pysat_inst.pandas_format:
        finite_mask = ((np.sign(lat) == hemisphere)
                       & np.isfinite(pysat_inst[:, pysat_names].max(axis=1)))
        dat_coords = [pysat_inst.index.name]
        combo_shape = [pysat_inst.index.shape[0]]
    else:
        nan_inst = pysat_inst.data.where(np.sign(pysat_inst.data[mlat_name])
                                         >= hemisphere)
        finite_mask = np.isfinite(nan_inst[pysat_names].to_array().max(
            'variable'))
        dat_coords = [coord for coord in pysat_inst[pysat_names].coords.keys()]
    dat_ind = np.where(finite_mask)

    # Test the OCB data
    if ocb.records == 0:
        ocbpy.logger.error("no data in Boundary file(s)")
        return

    # Ensure the LT, Lat, and Height data are the same shape
    if lat.shape != lt.shape or lat.shape[0] != pysat_inst.index.shape[
            0] or (height.shape != lat.shape and len(height.shape) > 0):
        if pysat_inst.pandas_format:
            raise ValueError('unexpected height shape or bad lat/lt data')

        # Use local time to set the initial coordinates, since it will have
        # UT dependence and latitude may not
        ocb_coords = [lt_coord for lt_coord
                      in pysat_inst[mlt_name].coords.keys()]
        combo_shape = list(lt.shape)

        # Expand the coordinates if the lat coordinates are not the
        # same as the LT coordinates or height coordinatess
        for lat_coord in pysat_inst[mlat_name].coords:
            if lat_coord not in pysat_inst[mlt_name].coords:
                combo_shape.append(pysat_inst[lat_coord].shape[0])
                ocb_coords.append(lat_coord)

        # Reshape the latitude and local time data if necessary
        if lat.shape != tuple(combo_shape) or lt.shape != tuple(combo_shape):
            out_lat, out_lt = np.meshgrid(lat, lt)
            lat = out_lat.reshape(combo_shape)
            lt = out_lt.reshape(combo_shape)

        # Determine if reshaping for altitude is necessary
        if len(height.shape) == 0:
            height = np.full(shape=lat.shape, fill_value=height)
        elif height.shape != lat.shape:
            if height_name in pysat_inst.variables:
                new_coords = False
                for alt_coord in pysat_inst[height_name].coords:
                    if alt_coord not in ocb_coords:
                        combo_shape.append(pysat_inst[alt_coord].shape[0])
                        ocb_coords.append(alt_coord)
                        new_coords = True

                # Reshape the data
                if new_coords:
                    out_lat, out_height = np.meshgrid(lat, height)
                    out_lt, _ = np.meshgrid(lt, height)
                    lat = out_lat.reshape(combo_shape)
                    height = out_height.reshape(combo_shape)
                    lt = out_lt.reshape(combo_shape)
                else:
                    height = height.reshape(combo_shape)
            elif len(height.shape) == len(lat.shape):
                # Try and reshape the height
                height = height.reshape(combo_shape)
            else:
                # Can't reshape the height
                raise ValueError('unexpected height shape')
    else:
        ocb_coords = [pysat_inst.index.name]

    # See if the data has more dimensions than the OCB coordinates
    if len(dat_coords) > len(ocb_coords):
        combo_shape = list(lat.shape)
        for dcoord in dat_coords:
            if dcoord not in ocb_coords:
                ocb_coords.append(dcoord)
                combo_shape.append(pysat_inst[dcoord].shape[0])

        # Reverse and transpose the arrays
        combo_shape.reverse()
        out_lat = np.full(shape=combo_shape, fill_value=lat.transpose())
        out_lt = np.full(shape=combo_shape, fill_value=lt.transpose())
        out_height = np.full(shape=combo_shape, fill_value=height.transpose())
        lat = out_lat.transpose()
        lt = out_lt.transpose()
        height = out_height.transpose()

    # Initialise the OCB output
    ocb_output = dict()
    for oattr in ocb_names:
        eattr = oattr[:-1 * len(bname) - 1]
        if eattr in vector_names.keys():
            for vattr in ocb_vect_attrs:
                ovattr = '_'.join([oattr, vattr])
                ovattr = ovattr.replace('_ocb_', '_')
                if bname != "ocb":
                    ovattr = ovattr.replace("ocb", bname)
                ocb_output[ovattr] = np.full(lat.shape, np.nan, dtype=float)
        else:
            ocb_output[oattr] = np.full(lat.shape, np.nan, dtype=float)

    # Cycle through the data, matching data and OCB records
    idat = 0
    uind = np.unique(dat_ind[0])
    ndat = len(uind)
    if hasattr(ocb, "boundary_lat"):
        ref_r = 90.0 - abs(ocb.boundary_lat)
    else:
        ref_r = 90.0 - abs(ocb.ocb.boundary_lat)

    while idat < ndat and ocb.rec_ind < ocb.records:
        idat = ocbpy.match_data_ocb(ocb, pysat_inst.index[uind],
                                    idat=idat, max_tol=max_sdiff,
                                    min_merit=min_merit, max_merit=max_merit,
                                    **kwargs)

        if idat < ndat and ocb.rec_ind < ocb.records:
            time_ind = np.where(dat_ind[0] == uind[idat])[0]

            if len(dat_coords) > 1:
                iout = tuple(dind[time_ind] for dind in dat_ind)
                if pysat_var_names > 1:
                    # If there is more than one variable, need to downselect
                    time_sel = pysat_inst[pysat_names].to_array().max(
                        'variable')
                else:
                    time_sel = pysat_inst[pysat_names]

                time_mask = np.isfinite(time_sel.where(
                    finite_mask & (pysat_inst[pysat_inst.index.name]
                                   == pysat_inst.index[uind][idat])))
                vind = iout[0]
            else:
                iout = dat_ind[0][time_ind]
                vind = iout
                time_mask = None

            # Get the OCB coordinates
            nout = ocb.normal_coord(lat[iout], lt[iout], coords=loc_coord,
                                    height=height[iout])

            if len(nout) == 3:
                ocb_output[olat_name][iout] = nout[0]
                ocb_output[omlt_name][iout] = nout[1]
                ocb_output[ocor_name][iout] = nout[2]
            else:
                ocb_output[olat_name][iout] = nout[0]
                ocb_output[omlt_name][iout] = nout[1]
                ocb_output[ocor_name][iout] = nout[3]

            # Scale and orient the vector data
            if nvect > 0:
                # Set this value's vector data
                vector_default = {"ocb_lat": ocb_output[olat_name][iout],
                                  "ocb_mlt": ocb_output[omlt_name][iout],
                                  "r_corr": ocb_output[ocor_name][iout],
                                  "vect_n": 0.0, "vect_e": 0.0, "vect_z": 0.0,
                                  "vect_mag": np.nan, "dat_name": None,
                                  "dat_units": None, "scale_func": None,
                                  "loc_coord": loc_coord,
                                  "vect_coord": vect_coord,
                                  "height": height[iout]}
                vector_init = dict(vector_default)
                vshape = list()

                for eattr in vector_names.keys():
                    oattr = "{:s}_{:s}".format(eattr, bname)
                    for ikey in vector_names[eattr].keys():
                        # Not all vector names are DataFrame names
                        vname = vector_names[eattr][ikey]
                        if vname in pysat_inst.variables:
                            if time_mask is None:
                                vector_init[ikey] = pysat_inst[vname][iout]
                            else:
                                vector_init[ikey] = reshape_pad_mask_flatten(
                                    pysat_inst[vname], time_mask)

                            if vector_init[ikey].shape not in vshape:
                                vshape.append(vector_init[ikey].shape)
                        else:
                            vector_init[ikey] = vname

                    # Perform the vector scaling
                    vout = ocbscal.VectorData(vind, ocb.rec_ind, lat[iout],
                                              lt[iout], **vector_init)
                    vout.set_ocb(ocb)

                    # Assign the vector attributes to the output
                    for vattr in ocb_vect_attrs:
                        ovattr = '_'.join([oattr, vattr])
                        ovattr = ovattr.replace('_ocb_', '_')
                        ocb_output[ovattr][iout] = getattr(vout, vattr)

            if hasattr(ocb, "ocb"):
                unscaled_r = ocb.ocb.r[ocb.ocb.rec_ind] + ocb_output[
                    ocor_name][iout]
            else:
                unscaled_r = ocb.r[ocb.rec_ind] + ocb_output[
                    ocor_name][iout]

            # Scale the proportional variables
            for scale_names, scale_func in [
                    (evar_names, ocbscal.normal_evar),
                    (curl_evar_names, ocbscal.normal_curl_evar)]:
                for eattr in scale_names:
                    oattr = "{:s}_{:s}".format(eattr, bname)
                    if time_mask is None:
                        evar = pysat_inst[eattr][iout]
                    else:
                        evar = reshape_pad_mask_flatten(pysat_inst[eattr],
                                                        time_mask)

                    # Scale the variable
                    ocb_output[oattr][iout] = scale_func(evar, unscaled_r,
                                                         ref_r)

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
        eattr = oattr.split('_{:s}'.format(bname))[0]
        if hasattr(ocb, "instrument"):
            notes = "".join(["Boundary obtained from ", ocb.instrument,
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
            eattr = vector_attrs['_'.join([eattr, bname])][0]
            isvector = True
        else:
            isvector = False

        add_ocb_to_metadata(pysat_inst, oattr, eattr, notes=notes,
                            isvector=isvector)

    return


def reshape_pad_mask_flatten(data, mask):
    """Reshape, pad, mask, and flatten data.

    Parameters
    ----------
    data : xr.DataArray
        Data to be reshaped, padded, masked, and flattened for processing.
    mask : xr.DataArray
        Mask with the desired dimensions

    Returns
    -------
    flat : np.array
        Flattened array of good data, as specified by the mask

    """
    if np.all(mask.dims == data.dims):
        if mask.shape == data.shape:
            flat = data.where(mask, drop=True).values.flatten()
        else:
            raise ValueError('different shapes for the same dimesions')
    else:
        # Reshape this data variable for existing dims
        data_dims = [dim for dim in mask.dims if dim in data.dims]
        flat = data.transpose(*data_dims, ...).values

        # Pad by adding the additional dimensions if needed
        if len(data_dims) < len(mask.dims):
            try:
                flat = np.full(shape=tuple(reversed(list(mask.shape))),
                               fill_value=flat.transpose()).transpose()
            except Exception as xerr:
                # Using Exception instead of AssertionError because the
                # catch is not consistent
                raise ValueError(''.join(['vector variables must all have the',
                                          ' same shape, {:}'.format(xerr)]))

        flat = flat[mask.values].flatten()

    return flat


def add_ocb_to_metadata(pysat_inst, ocb_name, pysat_name, overwrite=False,
                        notes='', isvector=False):
    """Update pysat metadata for OCB data.

    Parameters
    ----------
    pysat_inst : pysat.Instrument
        pysat.Instrument class object containing magnetic coordinates
    ocb_name : str
        Data column name for boundary data
    pysat_name : str
        Data column name for non-adaptive boundary version of this data
    overwrite : bool
        Overwrite existing metadata, if present (default=False)
    notes : str)
        Notes about this boundary data (default='')
    isvector : bool
        Is this vector data or not (default=False)

    Raises
    ------
    ValueError
        If input pysat Instrument object is the wrong class

    """
    bound_desc = {"ocb": "Open Closed field-line Boundary",
                  "eab": "Equatorward Auroral Boundary",
                  "dualb": "Dual Boundary", "": ""}

    bname = ""
    for bkey in ocb_name.split("_"):
        if len(bkey) > 0 and bkey in bound_desc.keys():
            bname = bkey
            break

    bextra = "" if len(bname) == 0 else "{:s}_".format(bname.upper())

    # Test the input
    if not isinstance(pysat_inst, pysat.Instrument):
        raise ValueError('unknown class, expected pysat.Instrument')

    if not overwrite and ocb_name in pysat_inst.meta.data.index:
        ocbpy.logger.warning("Boundary data already has metadata")

    else:
        if pysat_name not in pysat_inst.meta.data.index:
            name = (bextra + ocb_name.split("_{:s}".format(bname))[0]).replace(
                "_", " ")
            new_meta = {pysat_inst.meta.labels.fill_val: np.nan,
                        pysat_inst.meta.labels.name: name,
                        pysat_inst.meta.labels.desc: name.replace(
                            bname.upper(), bound_desc[bname]),
                        pysat_inst.meta.labels.min_val: -np.inf,
                        pysat_inst.meta.labels.max_val: np.inf}
        elif isvector:
            name = (bextra + ocb_name.split("_{:s}".format(bname))[0]).replace(
                "_", " ")
            new_meta = {pysat_inst.meta.labels.fill_val: np.nan,
                        pysat_inst.meta.labels.name: name,
                        pysat_inst.meta.labels.desc: "".join([
                            bound_desc[bname], pysat_inst.meta[pysat_name][
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
                bname.upper(), " ", new_meta[pysat_inst.meta.labels.name]])
            new_meta[pysat_inst.meta.labels.desc] = "".join([
                bound_desc[bname], new_meta[pysat_inst.meta.labels.desc]])

        # Set the notes
        new_meta[pysat_inst.meta.labels.notes] = notes

        # Set new metadata
        pysat_inst.meta[ocb_name] = new_meta

    return
