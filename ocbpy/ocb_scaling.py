#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Scale data affected by magnetic field direction or electric field.

References
----------
.. [1] Chisham, G. (2017), A new methodology for the development of
   high-latitude ionospheric climatologies and empirical models, Journal of
   Geophysical Research: Space Physics, 122, doi:10.1002/2016JA023235.

"""

import aacgmv2
import numpy as np
import warnings

import ocbpy
from ocbpy import ocb_time
from ocbpy import vectors


class VectorData(object):
    """Object containing a vector data.

    Parameters
    ----------
    dat_ind : int or array-like
        Data index (zero offset) for the input
    ocb_ind : int or array-like
        OCBoundary or DualBoundary record index matched to this data index
        (zero offset)
    lat : float or array-like
        Vector latitude (degrees)
    lt : float or array-like
        Vector LT (hours)
    height : float or array-like
        Geocentric height above sea level (km) at which magnetic coordinates
        will be calculated if conversion is needed (default=350.0)
    loc_coord : str
        Name of the coordinate system for `lat` and `lt`; one of 'magnetic',
        'geocentric', or 'geodetic' (default='magnetic')
    ocb_lat : float or array-like
        Vector OCB latitude (degrees) (default=np.nan)
    ocb_mlt : float or array-like
        Vector OCB MLT (hours) (default=np.nan)
    vect_n : float or array-like
        Vector North-pointing component (positive towards North) (default=0.0)
    vect_e : float or array-like
        Vector East-pointing component (completes right-handed coordinate system
        (default = 0.0)
    vect_z : float or array-like
        Vector vertical-pointing component (positive down) (default=0.0)
    vect_mag : float or array-like
        Vector magnitude (default=np.nan)
    vect_coord : str
        Name of the coordinate system for `vect_n` and `vect_e`; one of
        'magnetic', 'geocentric', or 'geodetic' (default='magnetic')
    dat_name : str
        Data name (default=None)
    dat_units : str
        Data units (default=None)
    scale_func : function
        Function for scaling AACGM magnitude with arguements: [measurement
        value, mesurement AACGM latitude (degrees), mesurement OCB latitude
        (degrees)] (default=None)
    **kwargs : dict
        Accepts deprecated parameters: `aacgm_lat`, `aacgm_mlt`, `aacgm_n`,
        `aacgm_e`, `aacgm_z`, and `aacgm_mag`.

    Attributes
    ----------
    vshape : array-like
        Shape of output data
    unscaled_r : float or array-like
        Radius of polar cap in degrees
    scaled_r : float or array-like
        Radius of normalised OCB polar cap in degrees
    ocb_n : float or array-like
        OCB north component of data vector (default=np.nan)
    ocb_e : float or array-like
        OCB east component of data vector (default=np.nan)
    ocb_z : float or array-like
        OCB vertical component of data vector (default=np.nan)
    ocb_mag : float or array-like
        OCB magnitude of data vector (default=np.nan)
    ocb_quad : int or array-like
        AACGM quadrant of OCB pole (default=0)
    vec_quad : int or array-like
        AACGM quadrant of Vector (default=0)
    pole_angle : float or array-like
        Angle at vector location appended by AACGM and OCB poles in degrees
        (default=np.nan)
    aacgm_naz : float or array-like
        AACGM north azimuth of data vector in degrees; deprecated
        (default=np.nan)
    ocb_aacgm_lat : float or array-like
        AACGM latitude of OCB pole in degrees (default=np.nan)
    ocb_aacgm_mlt : float or array-like
        AACGM MLT of OCB pole in hours (default=np.nan)

    Notes
    -----
    May only handle one data type, so scale_func cannot be an array

    Warnings
    --------
    DeprecationWarning
        Several kwargs/attributes and method have been changed to reflect new
        allowed input types (data in geodetic or geographic coordinates).
        Support for the old parameters and methods will be removed in
        version 0.4.1+.

    """

    def __init__(self, dat_ind, ocb_ind, lat, lt, height=350.0,
                 loc_coord='magnetic', ocb_lat=np.nan, ocb_mlt=np.nan,
                 r_corr=np.nan, vect_n=0.0, vect_e=0.0, vect_z=0.0,
                 vect_mag=np.nan, vect_coord='magnetic', dat_name=None,
                 dat_units=None, scale_func=None, **kwargs):

        # Assign the vector data name and units
        self.dat_name = dat_name
        self.dat_units = dat_units

        # Assign the data and OCB indices
        self.dat_ind = dat_ind
        self.ocb_ind = ocb_ind

        # Assign the AACGM vector values, coordinates, and location
        self.vect_n = vect_n
        self.vect_e = vect_e
        self.vect_z = vect_z
        self.lat = lat
        self.lt = lt
        self.height = height
        self.loc_coord = loc_coord.lower()
        self.vect_coord = vect_coord.lower()

        # Check for deprecated values
        set_mag = True
        if len(kwargs.keys()) > 0:
            used_dep = list()
            dep_pairs = {'aacgm_n': 'vect_n', 'aacgm_e': 'vect_e',
                         'aacgm_lat': 'lat', 'aacgm_mlt': 'lt',
                         'aacgm_z': 'vect_z', 'aacgm_mag': 'vect_mag'}
            for dep_key in dep_pairs.keys():
                if dep_key in kwargs.keys():
                    # Save the deprecated kwarg to raise a single warning later
                    used_dep.append(dep_key)

                    # Update the new attribute
                    setattr(self, dep_pairs[dep_key], kwargs[dep_key])

                    if dep_key == 'aacgm_mag':
                        set_mag = False

            # Raise a warning
            if len(used_dep) < len(kwargs.keys()):
                ocbpy.logger.warning('unknown kwargs, ignored: {:}'.format(
                    [key for key in kwargs.keys() if key not in used_dep]))
            else:
                new_kwargs = [dep_pairs[dep_key] for dep_key in used_dep]
                warnings.warn("".join(['kwargs have been replaced with new ',
                                       'names that reflect their new scope. ',
                                       'Old kwargs will be removed in version ',
                                       '0.4.1+. Old kwargs used: ',
                                       repr(used_dep), '; replace with: ',
                                       repr(new_kwargs)]),
                              DeprecationWarning, stacklevel=2)

        if set_mag:
            # Set the magnitude if the deprecated kwarg was not supplied
            self.vect_mag = vect_mag

        # Test the coordinate systems for valid options
        self._test_coords()

        # Test the initalization shape and update the vector shapes if needed
        self._test_update_vector_shape()

        # Assign the OCB vector default values
        self.ocb_lat = ocb_lat
        self.ocb_mlt = ocb_mlt
        self.r_corr = r_corr
        self._test_update_bound_shape()

        # Assign the initial OCB vector default values and location, as well as
        # the default pole locations, relative angles, and quadrants
        self.clear_data()

        # Assign the vector scaling function
        self.scale_func = scale_func

        return

    def __repr__(self):
        """Provide an evaluatable representation of the VectorData object."""

        # Format the function representations
        if self.scale_func is None:
            repr_func = self.scale_func.__repr__()
        else:
            repr_func = ".".join([self.scale_func.__module__,
                                  self.scale_func.__name__])

        # Format the base output
        out = "".join(["ocbpy.ocb_scaling.VectorData(", repr(self.dat_ind),
                       ", ", repr(self.ocb_ind), ", ", repr(self.lat),
                       ", ", repr(self.lt), ", height=", repr(self.height),
                       ", loc_coord=", repr(self.loc_coord), ", ocb_lat=",
                       repr(self.ocb_lat), ", ocb_mlt=", repr(self.ocb_mlt),
                       ", r_corr=", repr(self.r_corr), ", vect_n=",
                       repr(self.vect_n), ", vect_e=", repr(self.vect_e),
                       ", vect_z=", repr(self.vect_z), ", vect_mag=",
                       repr(self.vect_mag), ", vect_coord=",
                       repr(self.vect_coord), ", dat_name=",
                       repr(self.dat_name), ", dat_units=",
                       repr(self.dat_units), ", scale_func=", repr_func, ")"])

        # Reformat the numpy representations
        out = out.replace('array', 'numpy.array')

        return out

    def __str__(self):
        """Provide user readable representation of the VectorData object."""

        out = "".join([
            "Vector data:",
            "" if self.dat_name is None else " {:s}".format(self.dat_name),
            "" if self.dat_units is None else " ({:s})".format(self.dat_units),
            "\nData Index {:}\tBoundary Index ".format(self.dat_ind),
            "{:}\n-------------------------------------------".format(
                self.ocb_ind)])

        # Print vector location(s)
        if self.dat_ind.shape == () and self.ocb_ind.shape == ():
            out = "\n".join([
                out, "Locations: [Lat. (degrees), LT (hours), Alt (km)]",
                "{:s}: [{:.3f}, {:.3f}, {:.1f}]".format(
                    self.loc_coord.rjust(9), self.lat, self.lt, self.height),
                "      OCB: [{:.3f}, {:.3f}, N/A]".format(self.ocb_lat,
                                                          self.ocb_mlt)])
        else:
            out = '\n'.join([
                out,
                "Locations: [Lat. (degrees), LT (hours), Alt (km), Index]"])

            if self.dat_ind.shape == self.ocb_ind.shape or len(
                    self.ocb_ind.shape) == 0:
                for i, dind in enumerate(self.dat_ind):
                    if len(self.ocb_lat.shape) == 0 and np.isnan(self.ocb_lat):
                        ocb_line = "      OCB: [nan, nan, N/A, {:d}]".format(
                            self.ocb_ind)
                    else:
                        ocb_line = "".join([
                            "      OCB: [{:.3f}, ".format(self.ocb_lat[i]),
                            "{:.3f}, ".format(self.ocb_mlt[i]),
                            "N/A, {:d}]".format(
                                self.ocb_ind if len(self.ocb_ind.shape) == 0
                                else self.ocb_ind[i])])
                    out = '\n'.join([
                        out, "{:s}: [{:.3f}, {:.3f}, {:.1f}, {:d}]".format(
                            self.loc_coord.rjust(9), self.lat[i], self.lt[i],
                            self.height[i], dind), ocb_line])
            else:
                out = '\n'.join([
                    out, "{:s}: [{:.3f}, {:.3f}, {:.1f}, {:d}]".format(
                        self.loc_coord.rjust(9), self.lat, self.lt,
                        self.height, self.dat_ind)])
                for i, oind in enumerate(self.ocb_ind):
                    out = '\n'.join([
                        out, "      OCB: [{:.3f}, {:.3f}, N/A, {:d}]\n".format(
                            self.ocb_lat[i], self.ocb_mlt[i], oind)])

        out = "\n".join([out, "-------------------------------------------"])
        if self.vect_mag.shape == () and self.ocb_mag.shape == ():
            out = '\n'.join([out, "    Value: Mag. [N, E, Z]",
                             "{:s}: {:.3g} [{:.3g}, {:.3g}, {:.3g}]".format(
                                 self.vect_coord, self.vect_mag, self.vect_n,
                                 self.vect_e, self.vect_z)])
            if not np.isnan(self.ocb_mag):
                out = '\n'.join([
                    out, "  OCB: {:.3g} [{:.3g}, {:.3g}, {:.3g}]".format(
                        self.ocb_mag, self.ocb_n, self.ocb_e, self.ocb_z)])
        else:
            out = '\n'.join([out, "   Value: Mag. [N, E, Z] Index"])
            for i, mag in enumerate(self.ocb_mag):
                if self.vect_mag.shape == () and i == 0:
                    vec_line = ''.join([
                        self.vect_coord, ": {:.3g}".format(self.vect_mag),
                        " [{:.3g}, {:.3g}, ".format(self.vect_n, self.vect_e),
                        "{:.3g}] {:d}".format(self.vect_z, self.dat_ind)])
                elif self.vect_mag.shape != ():
                    vec_line = ''.join([
                        self.vect_coord, ": {:.3g} [".format(self.vect_mag[i]),
                        "{:.3g}, {:.3g}".format(self.vect_n[i], self.vect_e[i]),
                        ", {:.3g}] {:d}".format(self.vect_z[i],
                                                self.dat_ind[i])])
                if not np.isnan(mag):
                    vec_line = "".join([
                        vec_line, "\n     OCB: {:.3g} [".format(mag),
                        "{:.3g}, {:.3g}, ".format(self.ocb_n[i], self.ocb_e[i]),
                        "{:.3g}] {:d}".format(
                            self.ocb_z[i], self.ocb_ind
                            if self.ocb_ind.shape == () else self.ocb_ind[i])])
                out = '\n'.join([out, vec_line])

        # Print the scaling information
        out = "\n".join([out, "-------------------------------------------",
                         "No magnitude scaling function provided"
                         if self.scale_func is None else
                         "Scaling function: {:s}\n".format(
                             self.scale_func.__name__)])

        return out

    def __setattr__(self, name, value):
        """Set attributes based on their type.

        Parameters
        ----------
        name : str
            Attribute name to be assigned to VectorData
        value
            Value (any type) to be assigned to attribute specified by name

        """
        # Determine the desired output type
        out_val = np.asarray(value)
        type_str = str(out_val.dtype)

        if type_str.find('int') < 0 and type_str.find('float') < 0:
            out_val = value

        # TODO(#133): remove after old attributes are deprecated
        dep_pairs = {'aacgm_n': 'vect_n', 'aacgm_e': 'vect_e',
                     'aacgm_lat': 'lat', 'aacgm_mlt': 'lt',
                     'aacgm_z': 'vect_z', 'aacgm_mag': 'vect_mag'}
        if name in dep_pairs.keys():
            warnings.warn("".join([name, ' has been replaced with ',
                                   dep_pairs[name], '. Old attribute will be ',
                                   'removed in version 0.4.1+.']),
                          DeprecationWarning, stacklevel=2)
            name = dep_pairs[name]

        # Use Object to avoid recursion
        super(VectorData, self).__setattr__(name, out_val)
        return

    # TODO(#133): remove after old attributes are deprecated
    def __getattribute__(self, name, **kwargs):
        """Get attributes, allowing access to deprecated names.

        Parameters
        ----------
        name : str
            Attribute name to be accessed from VectorData

        Returns
        -------
        value : any
            Value assigned to the attribute

        """
        # Define the deprecated attributes that are not properties
        dep_pairs = {'aacgm_n': 'vect_n', 'aacgm_e': 'vect_e',
                     'aacgm_z': 'vect_z'}
        if name in dep_pairs.keys():
            warnings.warn("".join([name, ' has been replaced with ',
                                   dep_pairs[name], '. Old attribute will be ',
                                   'removed in version 0.4.1+.']),
                          DeprecationWarning, stacklevel=2)
            name = dep_pairs[name]
        elif name == "aacgm_naz":
            warnings.warn('`aacgm_naz` will be removed in version 0.4.1+.',
                          DeprecationWarning, stacklevel=2)

        # Use Object to avoid recursion
        value = super(VectorData, self).__getattribute__(name)
        return value

    def _ocb_attr_setter(self, ocb_name, ocb_val):
        """Set OCB attributes.

        Parameters
        ----------
        ocb_name : str
            OCB attribute name
        ocb_val : any
            Value to be assigned to attribute specified by name

        """
        # Ensure the shape is correct
        if np.asarray(ocb_val).shape == () and self.ocb_ind.shape != ():
            ocb_val = np.full(shape=self.ocb_ind.shape, fill_value=ocb_val)

        self.__setattr__(ocb_name, ocb_val)
        return

    def _dat_attr_setter(self, dat_name, dat_val):
        """Set data attributes.

        Parameters
        ----------
        dat_name : str
            OCB attribute name
        dat_val : any
            Value to be assigned to attribute specified by name

        """
        # Ensure the shape is correct
        if np.asarray(dat_val).shape == () and self.dat_ind.shape != ():
            dat_val = np.full(shape=self.dat_ind.shape, fill_value=dat_val)

        self.__setattr__(dat_name, dat_val)
        return

    def _test_coords(self):
        """Test the location and vector coordinate specifications.

        Raises
        ------
        ValueError
            If an unknown coordinate system is supplied or a mix of gedetic
            and geocentric is supplied

        """
        good_coords = ['magnetic', 'geocentric', 'geodetic']

        if self.loc_coord not in good_coords:
            raise ValueError(''.join(['unknown location coordinate: ',
                                      repr(self.loc_coord), ', expects one of ',
                                      repr(good_coords)]))

        if self.vect_coord not in good_coords:
            raise ValueError(''.join(['unknown vector coordinate: ',
                                      repr(self.vect_coord),
                                      ', expects one of ', repr(good_coords)]))

        if self.vect_coord != self.loc_coord and 'magnetic' not in [
                self.vect_coord, self.loc_coord]:
            raise ValueError('incompatible vector and location coordinates')

        return

    def _test_update_vector_shape(self):
        """Test and update the shape of the VectorData attributes.

        Raises
        ------
        ValueError
            If mismatches in the attribute shapes are encountered

        Notes
        -----
        Sets the `vshape` attribute and updates the shape of `vect_n`,
        `vect_e`, and `vect_z` if needed

        """

        # Get the required input shapes
        vshapes = list()
        for vshape in [self.lat.shape, self.lt.shape, self.dat_ind.shape,
                       self.vect_n.shape, self.vect_e.shape, self.vect_z.shape]:
            if vshape not in vshapes:
                vshapes.append(vshape)

        # Determine the desired shape
        self.vshape = () if len(vshapes) == 0 else max(vshapes)

        # Evaluate for potential mismatched attributes
        if len(vshapes) > 2 or (len(vshapes) == 2 and min(vshapes) != ()):
            raise ValueError('mismatched dimensions for VectorData inputs')

        if len(vshapes) > 1 and min(vshapes) == ():
            if self.dat_ind.shape == ():
                raise ValueError('data index shape must match vector shape')

            # Vector input needs to be the same length
            if self.vect_n.shape == ():
                self.vect_n = np.full(shape=self.vshape, fill_value=self.vect_n)
            if self.vect_e.shape == ():
                self.vect_e = np.full(shape=self.vshape, fill_value=self.vect_e)
            if self.vect_z.shape == ():
                self.vect_z = np.full(shape=self.vshape, fill_value=self.vect_z)
        return

    def _test_update_bound_shape(self):
        """Test and update the shape of the VectorData boundary attributes.

        Raises
        ------
        ValueError
            If mismatches in the attribute shapes are encountered

        """
        # Test the OCB input shape
        oshapes = list()
        for oshape in [self.ocb_lat.shape, self.ocb_mlt.shape,
                       self.r_corr.shape]:
            if oshape not in oshapes:
                oshapes.append(oshape)

        oshape = () if len(oshapes) == 0 else max(oshapes)

        if (self.ocb_ind.shape != () and (oshape == () or np.all(
                oshape != self.ocb_ind.shape))) or len(oshapes) > 2 or (
                    len(oshapes) == 2 and len(min(oshapes)) > 0):
            raise ValueError('OCB index and input shapes mismatched')

        # Compare and update the vector data shape if needed
        if self.ocb_ind.shape == ():
            oshape = self.vshape
        elif self.dat_ind.shape == ():
            self.vshape = oshape
        else:
            oshape = np.asarray(oshape)
            if self.vshape.size != oshape.size or oshape != self.vshape:
                raise ValueError('Mismatched OCB and Vector input shapes')
        return

    def _assign_normal_coord_output(self, out_coord, ind=None):
        """Get and assign OCB coordinates.

        Parameters
        ----------
        out_coord : tuple
            Tuple of outputs from `normal_coord` method
        ind : int or NoneType
            Index for assigning data to the local OCB attributes

        """
        # OCBoundary and EABoundary have thre outputs, DualBoundary has four
        if len(out_coord) == 3:
            if ind is None:
                (self.ocb_lat, self.ocb_mlt, self.r_corr) = out_coord
            else:
                (self.ocb_lat[ind], self.ocb_mlt[ind],
                 self.r_corr[ind]) = out_coord
        else:
            if ind is None:
                (self.ocb_lat, self.ocb_mlt, _, self.r_corr) = out_coord
            else:
                (self.ocb_lat[ind], self.ocb_mlt[ind], _,
                 self.r_corr[ind]) = out_coord
        return

    @property
    def vect_mag(self):
        """Magntiude of the vector(s)."""
        return self._vect_mag

    @vect_mag.setter
    def vect_mag(self, vect_mag):
        # Assign the vector magnitude(s)
        vect_sqrt = np.sqrt(self.vect_n**2 + self.vect_e**2 + self.vect_z**2)

        if np.all(np.isnan(vect_mag)):
            self._vect_mag = vect_sqrt
        else:
            if np.any(np.greater(abs(vect_mag - vect_sqrt), 1.0e-3,
                                 where=~np.isnan(vect_mag))):
                ocbpy.logger.warning("".join([
                    "inconsistent vector components with a maximum difference ",
                    "of {:} > 1.0e-3".format(abs(vect_mag - vect_sqrt).max())]))
            self._vect_mag = vect_mag
        return

    @property
    def aacgm_mag(self):
        """Deprecated magnitude of the vector(s)."""
        # TODO(#133): remove after old attributes are deprecated
        warnings.warn("".join(['`aacgm_mag` has been replaced with `vect_mag`,',
                               ' and will be removed in version 0.4.1+.']),
                      DeprecationWarning, stacklevel=2)
        return self.vect_mag

    @property
    def dat_ind(self):
        """Data index(es)."""
        return self._dat_ind

    @dat_ind.setter
    def dat_ind(self, dat_ind):
        # Set the data indices, and clear old data if needed
        if not hasattr(self, "dat_ind"):
            self._dat_ind = dat_ind
        else:
            self._dat_ind = dat_ind

            # Test the data and reset if necessary
            self._test_update_vector_shape()

            # Test the boundary shape
            self._test_update_bound_shape()

            # Reset the calculated boundary data
            self.clear_data()

            # Re-calculate the vector magnitude
            self.vect_mag = np.nan
        return

    @property
    def ocb_ind(self):
        """Boundary index(es)."""
        return self._ocb_ind

    @ocb_ind.setter
    def ocb_ind(self, ocb_ind):
        # Set the OCB indices, and clear old data if needed
        if not hasattr(self, 'ocb_ind'):
            self._ocb_ind = ocb_ind
        else:
            self._ocb_ind = ocb_ind

            # Test the boundaries and reset if necessary
            try:
                self._test_update_bound_shape()
            except ValueError as verr:
                if str(verr).find('OCB index and input shapes mismatch') == 0:
                    ocbpy.logger.warning(
                        '{:s}, unsetting boundary inputs'.format(str(verr)))
                    self.ocb_lat = np.nan
                    self.ocb_mlt = np.nan
                    self.r_corr = np.nan

                    if self.dat_ind.shape == ():
                        self.vshape = ocb_ind.shape

                    # Re-test boundary shape
                    self._test_update_bound_shape()
                else:
                    # Can't figure out how to get here, but keeping for now
                    raise ValueError(verr)

            # Clear the rest of the data
            self.clear_data()
        return

    @property
    def ocb_lat(self):
        """Boundary latitude in degrees."""
        return self._ocb_lat

    @ocb_lat.setter
    def ocb_lat(self, ocb_lat):
        # Set the boundary latitude value and ensure the shape is correct
        self._ocb_attr_setter('_ocb_lat', ocb_lat)
        return

    @property
    def ocb_mlt(self):
        """Boundary magnetic local time in hours."""
        return self._ocb_mlt

    @ocb_mlt.setter
    def ocb_mlt(self, ocb_mlt):
        # Set the boundary MLT value and ensure the shape is correct
        self._ocb_attr_setter('_ocb_mlt', ocb_mlt)
        return

    @property
    def r_corr(self):
        """Boundary radius correction in degrees."""
        return self._r_corr

    @r_corr.setter
    def r_corr(self, r_corr):
        # Set the boundary radius correction and ensure the shape is correct
        self._ocb_attr_setter('_r_corr', r_corr)
        return

    @property
    def lat(self):
        """Vector latitude in degrees."""
        return self._lat

    @lat.setter
    def lat(self, lat):
        # Set the boundary latitude value and ensure the shape is correct
        self._dat_attr_setter('_lat', lat)
        return

    @property
    def aacgm_lat(self):
        """Deprecated magnitude of the vector(s)."""
        # TODO(#133): remove after old attributes are deprecated
        warnings.warn("".join(['`aacgm_lat` has been replaced with `lat`, and ',
                               'will be removed in version 0.4.1+.']),
                      DeprecationWarning, stacklevel=2)
        return self.lat

    @property
    def lt(self):
        """Vector local time in hours."""
        return self._lt

    @lt.setter
    def lt(self, lt):
        # Set the vector LT value and ensure the shape is correct
        self._dat_attr_setter('_lt', lt)
        return

    @property
    def aacgm_mlt(self):
        """Deprecated magnitude of the vector(s)."""
        # TODO(#133): remove after old attributes are deprecated
        warnings.warn("".join(['`aacgm_mlt` has been replaced with `lt`, and ',
                               'will be removed in version 0.4.1+.']),
                      DeprecationWarning, stacklevel=2)
        return self.lt

    @property
    def height(self):
        """Vector in km."""
        return self._height

    @height.setter
    def height(self, height):
        # Set the vector height value and ensure the shape is correct
        self._dat_attr_setter('_height', height)
        return

    def clear_data(self):
        """Clear or initialize the output data attributes."""
        warnings.simplefilter("ignore")

        # Assign the OCB vector default values and location
        self.ocb_n = np.full(shape=self.vshape, fill_value=np.nan)
        self.ocb_e = np.full(shape=self.vshape, fill_value=np.nan)
        self.ocb_z = np.full(shape=self.vshape, fill_value=np.nan)
        self.ocb_mag = np.full(shape=self.vshape, fill_value=np.nan)

        # Assign the default pole locations, relative angles, and quadrants
        self.ocb_quad = np.zeros(shape=self.vshape)
        self.vec_quad = np.zeros(shape=self.vshape)
        self.pole_angle = np.full(shape=self.vshape, fill_value=np.nan)
        # TODO(#133): remove `aacgm_naz`
        self.aacgm_naz = np.full(shape=self.vshape, fill_value=np.nan)
        self.ocb_aacgm_lat = np.full(shape=self.vshape, fill_value=np.nan)
        self.ocb_aacgm_mlt = np.full(shape=self.vshape, fill_value=np.nan)

        warnings.resetwarnings()
        return

    def set_ocb(self, ocb, scale_func=None, trace_method='ALLOWTRACE'):
        """Set the OCBoundary values for provided data (updates all attributes).

        Parameters
        ----------
        ocb : ocbpy.OCBoundary or ocbpy.DualBoundary
            OCB, EAB, or Dual boundary object
        scale_func : function
            Function for scaling the vector magnitude with arguments:
            measurement value, measurement latitude (degrees), and measurement
            boundary-adjusted latitude (degrees). Not necessary if defined
            earlier or no scaling is needed. (default=None)
        trace_method : str
            Desired AAGCM tracing method (default='ALLOWTRACE')

        """
        # Update the data values to be in magnetic coordinates
        dtime = ocb.dtime[ocb.rec_ind] if self.ocb_ind.shape == () else [
            ocb.dtime[ind] for ind in self.ocb_ind]
        self.update_vect_coords_to_mag(dtime, ocb.hemisphere)

        # If the OCB vector coordinates weren't included in the initial info,
        # update them here
        if(np.all(np.isnan(self.ocb_lat)) or np.all(np.isnan(self.ocb_mlt))
           or np.all(np.isnan(self.r_corr))):
            # Because the boundary locations and magnetic field are both time
            # dependent, we can't call this function with multiple OCB/EABs
            if self.ocb_ind.shape == ():
                # Initialise the boundary index
                ocb.rec_ind = self.ocb_ind

                # Calculate the coordinates and save the output
                out_coord = ocb.normal_coord(self.lat, self.lt,
                                             coords=self.loc_coord,
                                             height=self.height,
                                             method=trace_method)
                self._assign_normal_coord_output(out_coord)
            else:
                # Cycle through the boundary indices
                for i, ocb.rec_ind in enumerate(self.ocb_ind):
                    # Handle time different, depending on the OCB and
                    # data shapes. Calcualte the coordinates and save the output
                    if self.ocb_ind.shape == self.dat_ind.shape:
                        out_coord = ocb.normal_coord(self.lat[i], self.lt[i],
                                                     coords=self.loc_coord,
                                                     height=self.height[i])
                    else:
                        out_coord = ocb.normal_coord(self.lat, self.lt,
                                                     coords=self.loc_coord,
                                                     height=self.height,
                                                     method=trace_method)
                    self._assign_normal_coord_output(out_coord, i)

        # Exit if the OCB coordinates can't be calculated at this location
        if(np.all(np.isnan(self.ocb_lat)) or np.all(np.isnan(self.ocb_mlt))
           or np.all(np.isnan(self.r_corr))):
            return

        # Set the AACGM coordinates of the OCB pole
        if hasattr(ocb, "ocb"):
            iocb = ocb.ocb_ind[self.ocb_ind]
            self.unscaled_r = ocb.ocb.r[iocb] + self.r_corr
            self.scaled_r = np.full(
                shape=self.unscaled_r.shape,
                fill_value=(90.0 - abs(ocb.ocb.boundary_lat)))
            self.ocb_aacgm_mlt, self.ocb_aacgm_lat = vectors.get_pole_loc(
                ocb.ocb.phi_cent[iocb], ocb.ocb.r_cent[iocb])
        else:
            self.unscaled_r = ocb.r[self.ocb_ind] + self.r_corr
            self.scaled_r = np.full(shape=self.unscaled_r.shape,
                                    fill_value=(90.0 - abs(ocb.boundary_lat)))
            self.ocb_aacgm_mlt, self.ocb_aacgm_lat = vectors.get_pole_loc(
                ocb.phi_cent[self.ocb_ind], ocb.r_cent[self.ocb_ind])

        # Get the angle at the data vector appended by the AACGM and OCB poles
        self.calc_vec_pole_angle()

        # Set the OCB and Vector quadrants
        if np.any(~np.isnan(self.pole_angle)):
            self.define_quadrants()

            # Set the scaling function
            if self.scale_func is None:
                if scale_func is None:
                    # This is not necessarily a bad thing, if the value does
                    # not need to be scaled.
                    ocbpy.logger.info("no scaling function provided")
                else:
                    self.scale_func = scale_func

            # Assign the OCB vector default values and location.  Will also
            # update the AACGM north azimuth of the vector.
            self.scale_vector()

        return

    def define_quadrants(self):
        """Define AACGM MLT quadrants for the OCB pole and data vector.

        Notes
        -----
        North (N) and East (E) are defined by the AACGM directions centred on
        the data vector location, assuming vertical is positive downwards
        Quadrants: 1 [N, E]; 2 [N, W]; 3 [S, W]; 4 [S, E]

        Requires `ocb_aacgm_mlt`, `lt`, `pole_angle`, `vect_n`, and `vect_e`.
        Both `loc_coord` and `vect_coord` must be 'magnetic'. Updates `ocb_quad`
        and `vec_quad`

        Raises
        ------
        ValueError
            If the required input is undefined or incorrect

        """
        # When defining quadrants, we will need the vector information in
        # magnetic coordinates
        if self.loc_coord != "magnetic" or self.vect_coord != "magnetic":
            raise ValueError('need magnetic coordinates to define quadrants')

        # Test input, where it is allowable to have empty vector input
        if np.all(np.isnan(self.ocb_aacgm_mlt)):
            raise ValueError("OCB pole location required")

        if np.all(np.isnan(self.lt)):
            raise ValueError("Vector location required")

        if np.all(np.isnan(self.pole_angle)):
            raise ValueError("vector angle in poles-vector triangle required")

        # Determine where the OCB pole is relative to the data vector
        self.ocb_quad = vectors.define_pole_quadrants(
            self.lt, self.ocb_aacgm_mlt, self.pole_angle)

        # Now determine which quadrant the vector is pointed into
        self.vec_quad = vectors.define_vect_quadrants(self.vect_n, self.vect_e)

        return

    def scale_vector(self):
        """Normalise a variable proportional to the curl of the electric field.

        Raises
        ------
        ValueError
            If the required input is not defined

        Notes
        -----
        Requires `lat`, `lt`, `ocb_aacgm_mlt`, `ocb_aacgm_lat`, and
        `pole_angle`. Updates `ocb_n`, `ocb_e`, `ocb_z`, and `ocb_mag`.
        Temporarily updates `aacgm_naz`, which has been deprecated and will
        be removed in version 0.4.1+.

        """
        warnings.simplefilter("ignore")

        # Test input
        if np.all(np.isnan(self.lat)) or np.all(np.isnan(self.lt)):
            raise ValueError("Vector locations required")

        if np.all(np.isnan(self.ocb_aacgm_mlt)):
            raise ValueError("OCB pole location required")

        if np.all(np.isnan(self.pole_angle)):
            raise ValueError("vector angle in poles-vector triangle required")

        # Adjust the vector to OCB coordinates without scaling
        self.ocb_n, self.ocb_e, self.ocb_z = vectors.adjust_vector(
            self.lt, self.lat, self.vect_n, self.vect_e, self.vect_z,
            self.vec_quad, self.ocb_aacgm_mlt, self.ocb_aacgm_lat,
            self.pole_angle, self.ocb_quad)

        # TODO(#133): remove `aacgm_naz`
        vmag = np.sqrt(self.vect_n**2 + self.vect_e**2)
        if len(vmag.shape) == 0:
            self.aacgm_naz = np.degrees(np.arccos(self.vect_n / vmag))
        else:
            zero_mask = ((self.vect_n == 0.0) & (self.vect_e == 0.0))
            ns_mask = ((self.pole_angle == 0.0) | (self.pole_angle == 180.0))
            norm_mask = ~(zero_mask + ns_mask)
            self.aacgm_naz[norm_mask] = np.degrees(np.arccos(
                self.vect_n[norm_mask] / vmag[norm_mask]))

        # Scale the outputs, if desired
        if self.scale_func is not None:
            if len(self.ocb_n.shape) == 0:
                self.ocb_n = np.full(
                    shape=self.ocb_n.shape, fill_value=self.scale_func(
                        self.vect_n, self.unscaled_r, self.scaled_r))
                self.ocb_e = np.full(
                    shape=self.ocb_e.shape, fill_value=self.scale_func(
                        self.vect_e, self.unscaled_r, self.scaled_r))
                self.ocb_z = np.full(
                    shape=self.ocb_z.shape, fill_value=self.scale_func(
                        self.vect_z, self.unscaled_r, self.scaled_r))
            else:
                self.ocb_n = self.scale_func(self.ocb_n, self.unscaled_r,
                                             self.scaled_r)
                self.ocb_e = self.scale_func(self.ocb_e, self.unscaled_r,
                                             self.scaled_r)
                self.ocb_z = self.scale_func(self.ocb_z, self.unscaled_r,
                                             self.scaled_r)

        # Calculate the scaled OCB vector magnitude
        self.ocb_mag = np.sqrt(self.ocb_n**2 + self.ocb_e**2 + self.ocb_z**2)

        warnings.resetwarnings()
        return

    def calc_ocb_polar_angle(self):
        """Calculate the OCB north azimuth angle.

        Returns
        -------
        ocb_naz : float or array-like
            Angle between measurement vector and OCB pole in degrees

        Raises
        ------
        ValueError
            If the required input is undefined

        Notes
        -----
        Requires `ocb_quad`, `vec_quad`, `aacgm_naz`, and `pole_angle`

        """
        # TODO(#133): deprecation warning, method is no longer needed here
        warnings.warn("".join(["`calc_ocb_polar_angle` method deprecated, and",
                               " will be removed in version 0.4.1+. Instead, ",
                               "use `ocbpy.vectors.calc_dest_polar_angle`."]),
                      DeprecationWarning, stacklevel=2)

        # Test input
        warnings.simplefilter("ignore")
        if np.all(np.isnan(self.aacgm_naz)):
            raise ValueError("AACGM North polar angle undefined")

        if np.all(np.isnan(self.pole_angle)):
            raise ValueError("Vector angle undefined")

        # Calcuate the North azimuth angle for the OCB pole
        ocb_naz = vectors.calc_dest_polar_angle(
            self.ocb_quad, self.vec_quad, self.aacgm_naz, self.pole_angle)

        warnings.resetwarnings()
        return ocb_naz

    def calc_ocb_vec_sign(self, north=False, east=False, quads=None):
        """Calculate the sign of the North and East components.

        Parameters
        ----------
        north : bool
            Get the sign of the north component(s) (default=False)
        east : bool
            Get the sign of the east component(s) (default=False)
        quads : dict or NoneType
            Dictionary of boolean values or arrays of boolean values for OCB
            and Vector quadrants. (default=None)

        Returns
        -------
        vsigns : dict
            Dictionary with keys 'north' and 'east' containing the desired
            signs

        Raises
        ------
        ValueError
            If the required input is undefined

        Notes
        -----
        Requires `ocb_quad`, `vec_quad`, `aacgm_naz`, and `pole_angle`.
        Method is deprecated and will be removed in version 0.4.1+.

        """
        # TODO(#133): deprecation warning, method is no longer needed here
        warnings.warn("".join(["`calc_ocb_vec_sign` method deprecated, and",
                               " will be removed in version 0.4.1+. Instead, ",
                               "use `ocbpy.vectors.calc_dest_vec_sign`."]),
                      DeprecationWarning, stacklevel=2)

        # Test input
        warnings.simplefilter("ignore")
        if not np.any([north, east]):
            raise ValueError("must set at least one direction")

        if np.all(np.isnan(self.aacgm_naz)):
            raise ValueError("AACGM polar angle undefined")

        if np.all(np.isnan(self.pole_angle)):
            raise ValueError("Vector angle undefined")

        # Calcualte the sign of the North and East vector components
        vsigns = vectors.calc_dest_vec_sign(
            self.ocb_quad, self.vec_quad, self.aacgm_naz, self.pole_angle,
            north=north, east=east, quads=quads)

        warnings.resetwarnings()
        return vsigns

    def calc_vec_pole_angle(self):
        """Calc the angle between the AACGM pole, data, and the OCB pole.

        Raises
        ------
        ValueError
            If the input is undefined or inappropriate

        Notes
        -----
        Requires `lt` and `lat` in magnetic coordinates, as well as
        defined `ocb_aacgm_mlt` and `ocb_aacgm_lat` attributes. Updates
        `pole_angle` using spherical trigonometry.

        """
        # When defining vector-pole angles, we will need the vector location in
        # magnetic coordinates
        if self.loc_coord != "magnetic":
            raise ValueError(
                'need magnetic coordinates to define vector-pole angles')

        # Cast inputs as arrays
        self.lt = np.asarray(self.lt)
        self.lat = np.asarray(self.lat)
        self.ocb_aacgm_mlt = np.asarray(self.ocb_aacgm_mlt)
        self.ocb_aacgm_lat = np.asarray(self.ocb_aacgm_lat)

        # Test input
        if np.all(np.isnan(self.lt)):
            raise ValueError("Vector local time is undefined")

        if np.all(np.isnan(self.lat)):
            raise ValueError("Vector latitude is undefined")

        if np.all(np.isnan(self.ocb_aacgm_mlt)):
            raise ValueError("AACGM MLT of OCB pole(s) undefined")

        if np.all(np.isnan(self.ocb_aacgm_lat)):
            raise ValueError("AACGM latitude of OCB pole(s) undefined")

        # Find the angle between the AACGM pole, the vector location in AACGM
        # coordinates, and the high-latitude boundary pole
        self.pole_angle = vectors.calc_vec_pole_angle(
            self.lt, self.lat, self.ocb_aacgm_mlt, self.ocb_aacgm_lat)

        return

    def update_loc_coords(self, dtimes, coord='magnetic',
                          trace_method='ALLOWTRACE'):
        """Update location coordiantes to the desired system.

        Parameters
        ----------
        dtimes : dt.datetime or list-like
            Datetime or list of datetimes for conversion
        coord : str
            Desired coordinate system, accepts 'magnetic', 'geodetic', and
            'geocentric' (default='magnetic')
        trace_method : str
            Desired AAGCM tracing method (default='ALLOWTRACE')

        Raises
        ------
        ValueError
            If the time and location inputs are mismatched.

        Notes
        -----
        Updates `lat`, `lt`, and `loc_coord` attributes.

        """
        dtime = None

        if coord.lower() != self.loc_coord:
            # Ensure the data is shaped correctly
            if len(self.lt.shape) == 0 and len(self.lat.shape) == 0:
                if hasattr(dtimes, 'year'):
                    # There is only one time and one location
                    dtime = dtimes
                else:
                    # There are multiple times and one location
                    self.lt = np.full(shape=len(dtimes), fill_value=self.lt)
                    self.lat = np.full(shape=len(dtimes), fill_value=self.lat)
                    self.height = np.full(shape=len(dtimes),
                                          fill_value=self.height)
            else:
                if hasattr(dtimes, 'year'):
                    # There is one time and multiple locations
                    dtime = dtimes
                else:
                    # There are multiple times and locations, the length must
                    # be the same
                    if len(dtimes) != len(self.lt) or len(dtimes) != len(
                            self.lat):
                        raise ValueError('mismatched time and location inputs')

            # Initalize the AACGM method using the recommending tracing
            methods = [trace_method]

            # Handle the conversion to/from magnetic coordinates separately
            if coord.lower() == "magnetic":
                # Update the method
                if self.loc_coord == "geocentric":
                    methods.append(self.loc_coord.upper())
                methods.append("G2A")

                if dtime is None:
                    new_lat = list()
                    new_lt = list()
                    method = "|".join(methods)
                    for i, val in enumerate(dtimes):
                        # Get the longitude for this time
                        lon = ocb_time.slt2glon(self.lt[i], val)

                        # Convert to magnetic coordinates
                        out = aacgmv2.get_aacgm_coord(
                            self.lat[i], lon, self.height[i], val, method)

                        # Save the output
                        new_lat.append(out[0])
                        new_lt.append(out[2])
                else:
                    # Get the longitude
                    lon = ocb_time.slt2glon(self.lt, dtime)

                    # Convert to magnetic coordinates
                    new_lat, _, new_lt = aacgmv2.get_aacgm_coord_arr(
                        self.lat, lon, self.height, dtime, "|".join(methods))
            else:
                # Update the method
                if coord.lower() == "geocentric":
                    methods.append(coord.upper())
                methods.append("A2G")

                if dtime is None:
                    new_lat = list()
                    new_lt = list()
                    method = "|".join(methods)
                    for i, val in enumerate(dtimes):
                        # Get the longitude for this time
                        lon = aacgmv2.convert_mlt(self.lt[i], val, m2a=True)

                        # Convert latitude and longitude
                        out = aacgmv2.convert_latlon(
                            self.lat[i], lon, self.height[i], val, method)

                        # Convert to SLT and save the latitude
                        new_lt.append(ocb_time.glon2slt(out[1], val))
                        new_lat.append(out[0])
                else:
                    # Get the longitude
                    lon = aacgmv2.convert_mlt(self.lt, dtime, m2a=True)

                    # Convert latitude and longitude
                    new_lat, new_lon, _ = aacgmv2.convert_latlon_arr(
                        self.lat, lon, self.height, dtime, "|".join(methods))

                    # Convert to SLT
                    new_lt = ocb_time.glon2slt(new_lon, dtime)

            # Update the location attributes
            self.lat = np.asarray(new_lat)
            self.lt = np.asarray(new_lt)
            self.loc_coord = coord

        return

    def update_vect_coords_to_mag(self, dtimes, hemisphere,
                                  trace_method='ALLOWTRACE'):
        """Convert geographic vector components into AAGGMV2 coordinates.

        Parameters
        ----------
        dtimes : dt.datetime or list-like
            Datetime or list of datetimes for conversion
        hemisphere : int
            -1 for Southern, 1 for Northern
        trace_method : str
            Desired AAGCM tracing method (default='ALLOWTRACE')

        Notes
        -----
        This follows the procedure in `set_ocb`, and is complicated to reverse.

        """
        dtime = None

        if self.vect_coord != "magnetic":
            # Need the geographic and magnetic locations
            if self.loc_coord == 'magnetic':
                # Assign the magnetic location
                mag_lt = np.asarray(self.lt)
                mag_lat = np.asarray(self.lat)

                # Calculate the geographic location
                self.update_loc_coords(dtimes, coord=self.vect_coord,
                                       trace_method=trace_method)
                geo_lt = np.asarray(self.lt)
                geo_lat = np.asarray(self.lat)

                # Re-assign the location values
                self.lt = np.asarray(mag_lt)
                self.lat = np.asarray(mag_lat)
                self.loc_coord = 'magnetic'
            else:
                # Assign the geographic location
                geo_lt = np.asarray(self.lt)
                geo_lat = np.asarray(self.lat)

                # Update the location coordiantes to be magnetic
                self.update_loc_coords(dtimes, trace_method=trace_method)
                mag_lt = np.asarray(self.lt)
                mag_lat = np.asarray(self.lat)

            # Exit if the magnetic coordinates can't be calculated
            if np.all(np.isnan(mag_lat)) or np.all(np.isnan(mag_lt)):
                return

            # Ensure the geographic and magnetic coordinates are the same shape
            if geo_lt.shape != mag_lt.shape:
                if len(geo_lt.shape) == 0:
                    # The geographic values are singlular, expand them
                    geo_lt = np.full(shape=mag_lt.shape, fill_value=geo_lt)
                    geo_lat = np.full(shape=mag_lt.shape, fill_value=geo_lat)
                elif len(mag_lt.shape) == 0:
                    # The magnetic values are singular, expend them
                    mag_lt = np.full(shape=geo_lt.shape, fill_value=mag_lt)
                    mag_lat = np.full(shape=geo_lt.shape, fill_value=mag_lat)

            # Determine if the time input is list-like
            if hasattr(dtimes, 'year'):
                dtime = dtimes

            # Set the AACGM coordinates of the geographic pole
            methods = [trace_method]
            if self.vect_coord == "geocentric":
                methods.append(self.vect_coord.upper())
            methods.append("A2G")

            if dtime is None:
                mag_pole_glat = list()
                mag_pole_slt = list()
                method = '|'.join(methods)
                for i, val in enumerate(dtimes):
                    # Get the geographic pole lat and lon
                    out = aacgmv2.convert_latlon(
                        hemisphere * 90.0, 0.0, self.height[i], val, method)

                    # Save the SLT and latitude
                    mag_pole_slt.append(ocb_time.glon2slt(out[1], val))
                    mag_pole_glat.append(out[0])
            else:
                mag_pole_glat, mag_pole_lon, _ = aacgmv2.convert_latlon(
                    hemisphere * 90.0, 0.0, self.height, dtime,
                    method_code='|'.join(methods))
                mag_pole_slt = ocb_time.glon2slt(mag_pole_lon, dtime)

            mag_pole_glat = np.asarray(mag_pole_glat)
            mag_pole_slt = np.asarray(mag_pole_slt)

            # Get the angle at the data vector appended by the AACGM and
            # geographic poles
            pole_angle = vectors.calc_vec_pole_angle(
                geo_lt, geo_lat, mag_pole_slt, mag_pole_glat)

            # Set the pole and vector quadrants
            if np.any(~np.isnan(pole_angle)):
                pole_quad = vectors.define_pole_quadrants(geo_lt, mag_pole_slt,
                                                          pole_angle)
                vect_quad = vectors.define_vect_quadrants(self.vect_n,
                                                          self.vect_e)

            # Adjust the geographic vector to AACGM coordinates
            mag_n, mag_e, mag_z = vectors.adjust_vector(
                mag_lt, mag_lat, self.vect_n, self.vect_e, self.vect_z,
                vect_quad, mag_pole_slt, mag_pole_glat, pole_angle, pole_quad)

            # Assign the new vector data and coordinate specification
            self.vect_n = np.asarray(mag_n)
            self.vect_e = np.asarray(mag_e)
            self.vect_z = np.asarray(mag_z)
            self.vect_coord = "magnetic"

            # Re-calculate the vector magnitude
            self.vect_mag = np.nan

        return


def normal_evar(evar, unscaled_r, scaled_r):
    """Normalise a variable proportional to the electric field.

    Parameters
    ----------
    evar : float or array
        Variable related to electric field (e.g. velocity)
    unscaled_r : float or array
        Radius of polar cap in degrees
    scaled_r : float or array
        Radius of normalised OCB polar cap in degrees

    Returns
    -------
    nvar : float or array
        Normalised variable

    Notes
    -----
    Assumes that the cross polar cap potential is fixed across the polar cap
    regardless of the radius of the Open Closed field line Boundary.  This is
    commonly assumed when looking at statistical patterns that control the IMF
    (which accounts for dayside reconnection) and assume that the nightside
    reconnection influence is averaged out over the averaged period [1]_.

    """

    nvar = evar * unscaled_r / scaled_r

    return nvar


def normal_curl_evar(curl_evar, unscaled_r, scaled_r):
    """Normalise a variable proportional to the curl of the electric field.

    Parameters
    ----------
    curl_evar : float or array
        Variable related to electric field (e.g. vorticity)
    unscaled_r : float or array
        Radius of polar cap in degrees
    scaled_r : float or array
        Radius of normalised OCB polar cap in degrees

    Returns
    -------
    nvar : float or array
        Normalised variable

    Notes
    -----
    Assumes that the cross polar cap potential is fixed across the polar cap
    regardless of the radius of the Open Closed field line Boundary.  This is
    commonly assumed when looking at statistical patterns that control the IMF
    (which accounts for dayside reconnection) and assume that the nightside
    reconnection influence is averaged out over the averaged period [1]_.

    """

    nvar = curl_evar * (unscaled_r / scaled_r)**2

    return nvar


def hav(alpha):
    """Calculate the haversine.

    Parameters
    ----------
    alpha : float or array-like
        Angle in radians

    Returns
    -------
    hav_alpha : float or array-like
        Haversine of alpha, equal to the square of the sine of half-alpha

    """
    alpha = np.asarray(alpha)
    hav_alpha = np.sin(alpha * 0.5)**2

    return hav_alpha


def archav(hav):
    """Calculate the inverse haversine.

    Parameters
    ----------
    hav : float or array-like
        Haversine of an angle

    Returns
    -------
    alpha : float or array-like
        Angle in radians

    Notes
    -----
    The input must be positive.  However, any number with a magnitude below
    10-16 will be rounded to zero.  More negative numbers will return NaN.

    """

    # Cast the output as array-like
    hav = np.asarray(hav)

    # Initialize the output to NaN, so that values of NaN or negative
    # numbers will return NaN
    alpha = np.full(shape=hav.shape, fill_value=np.nan)

    # If the number is positive, calculate the angle
    norm_mask = (np.greater_equal(hav, 1.0e-16, where=~np.isnan(hav))
                 & ~np.isnan(hav))
    if np.any(norm_mask):
        if hav.shape == ():
            alpha = 2.0 * np.arcsin(np.sqrt(hav))
        else:
            alpha[norm_mask] = 2.0 * np.arcsin(np.sqrt(hav[norm_mask]))

    #  The number is small enough that machine precision may have changed
    # the sign, but it's a single-precission zero
    small_mask = (np.less(abs(hav), 1.0e-16, where=~np.isnan(hav))
                  & ~np.isnan(hav))
    if np.any(small_mask):
        if hav.shape == ():
            alpha = 0.0
        else:
            alpha[small_mask] = 0.0

    return alpha
