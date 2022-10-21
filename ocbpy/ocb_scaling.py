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

import numpy as np

import ocbpy


class VectorData(object):
    """Object containing a vector data.

    Parameters
    ----------
    dat_ind : int or array-like
        Data index (zero offset) for the input
    ocb_ind : int or array-like
        OCBoundary or DualBoundary record index matched to this data index
        (zero offset)
    aacgm_lat : float or array-like
        Vector AACGM latitude (degrees)
    aacgm_mlt : float or array-like
        Vector AACGM MLT (hours)
    ocb_lat : float or array-like
        Vector OCB latitude (degrees) (default=np.nan)
    ocb_mlt : float or array-like
        Vector OCB MLT (hours) (default=np.nan)
    aacgm_n : float or array-like
        AACGM North pointing vector (positive towards North) (default=0.0)
    aacgm_e : float or array-like
        AACGM East pointing vector (completes right-handed coordinate system
        (default = 0.0)
    aacgm_z : float or array-like
        AACGM Vertical pointing vector (positive down) (default=0.0)
    aacgm_mag : float or array-like
        Vector magnitude (default=np.nan)
    dat_name : str
        Data name (default=None)
    dat_units : str
        Data units (default=None)
    scale_func : function
        Function for scaling AACGM magnitude with arguements: [measurement
        value, mesurement AACGM latitude (degrees), mesurement OCB latitude
        (degrees)] (default=None)

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
        AACGM north azimuth of data vector in degrees (default=np.nan)
    ocb_aacgm_lat : float or array-like
        AACGM latitude of OCB pole in degrees (default=np.nan)
    ocb_aacgm_mlt : float or array-like
        AACGM MLT of OCB pole in hours (default=np.nan)

    Notes
    -----
    May only handle one data type, so scale_func cannot be an array

    """

    def __init__(self, dat_ind, ocb_ind, aacgm_lat, aacgm_mlt, ocb_lat=np.nan,
                 ocb_mlt=np.nan, r_corr=np.nan, aacgm_n=0.0, aacgm_e=0.0,
                 aacgm_z=0.0, aacgm_mag=np.nan, dat_name=None, dat_units=None,
                 scale_func=None):

        # Assign the vector data name and units
        self.dat_name = dat_name
        self.dat_units = dat_units

        # Assign the data and OCB indices
        self.dat_ind = dat_ind
        self.ocb_ind = ocb_ind

        # Assign the AACGM vector values and location
        self.aacgm_n = aacgm_n
        self.aacgm_e = aacgm_e
        self.aacgm_z = aacgm_z
        self.aacgm_lat = aacgm_lat
        self.aacgm_mlt = aacgm_mlt

        # Test the initalization shape and update the vector shapes if needed
        self._test_update_vector_shape()

        # Assign the vector magnitude(s)
        self.aacgm_mag = aacgm_mag

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
                       ", ", repr(self.ocb_ind), ", ", repr(self.aacgm_lat),
                       ", ", repr(self.aacgm_mlt), ", ocb_lat=",
                       repr(self.ocb_lat), ", ocb_mlt=", repr(self.ocb_mlt),
                       ", r_corr=", repr(self.r_corr), ", aacgm_n=",
                       repr(self.aacgm_n), ", aacgm_e=", repr(self.aacgm_e),
                       ", aacgm_z=", repr(self.aacgm_z), ", aacgm_mag=",
                       repr(self.aacgm_mag), ", dat_name=",
                       repr(self.dat_name), ", dat_units=",
                       repr(self.dat_units), ", scale_func=", repr_func, ")"])

        # Reformat the numpy representations
        out = out.replace('array', 'numpy.array')

        return out

    def __str__(self):
        """Provide user readable representation of the VectorData object."""

        out = "Vector data:"
        if self.dat_name is not None:
            out += " {:s}".format(self.dat_name)
        if self.dat_units is not None:
            out += " ({:s})".format(self.dat_units)

        out += "\nData Index {:}\tBoundary Index {:}\n".format(self.dat_ind,
                                                               self.ocb_ind)
        out += "-------------------------------------------\n"

        # Print AACGM vector location(s)
        if self.dat_ind.shape == () and self.ocb_ind.shape == ():
            out += "Locations: [Mag. Lat. (degrees), MLT (hours)]\n"
            out += "    AACGM: [{:.3f}, {:.3f}]\n".format(self.aacgm_lat,
                                                          self.aacgm_mlt)
            out += "      OCB: [{:.3f}, {:.3f}]\n".format(self.ocb_lat,
                                                          self.ocb_mlt)
        else:
            out += "Locations: [Mag. Lat. (degrees), MLT (hours), Index]\n"
            if self.dat_ind.shape == self.ocb_ind.shape:
                for i, dind in enumerate(self.dat_ind):
                    out += "    AACGM: [{:.3f}, {:.3f}, {:d}]\n".format(
                        self.aacgm_lat[i], self.aacgm_mlt[i], dind)
                    out += "      OCB: [{:.3f}, {:.3f}, {:d}]\n".format(
                        self.ocb_lat[i], self.ocb_mlt[i], self.ocb_ind[i])
            elif self.ocb_ind.shape == ():
                for i, dind in enumerate(self.dat_ind):
                    out += "    AACGM: [{:.3f}, {:.3f}, {:d}]\n".format(
                        self.aacgm_lat[i], self.aacgm_mlt[i], dind)
                    if self.ocb_lat.shape == () and np.isnan(self.ocb_lat):
                        out += "      OCB: [nan, nan, {:d}]\n".format(
                            self.ocb_ind)
                    else:
                        out += "      OCB: [{:.3f}, {:.3f}, {:d}]\n".format(
                            self.ocb_lat[i], self.ocb_mlt[i], self.ocb_ind)
            else:
                out += "    AACGM: [{:.3f}, {:.3f}, {:d}]\n".format(
                    self.aacgm_lat, self.aacgm_mlt, self.dat_ind)
                for i, oind in enumerate(self.ocb_ind):
                    out += "      OCB: [{:.3f}, {:.3f}, {:d}]\n".format(
                        self.ocb_lat[i], self.ocb_mlt[i], oind)

        out += "\n-------------------------------------------\n"
        if self.aacgm_mag.shape == () and self.ocb_mag.shape == ():
            out += "Value: Magnitude [N, E, Z]\n"
            out += "AACGM: {:.3g} [{:.3g}".format(self.aacgm_mag, self.aacgm_n)
            out += ", {:.3g}, {:.3g}]\n".format(self.aacgm_e, self.aacgm_z)
            if not np.isnan(self.ocb_mag):
                out += "  OCB: {:.3g} [{:.3g}".format(self.ocb_mag, self.ocb_n)
                out += ", {:.3g}, {:.3g}]\n".format(self.ocb_e, self.ocb_z)
        else:
            out += "Value: Magnitude [N, E, Z] Index\n"
            for i, mag in enumerate(self.ocb_mag):
                if self.aacgm_mag.shape == () and i == 0:
                    out += "AACGM: {:.3g} [".format(self.aacgm_mag)
                    out += "{:.3g}, {:.3g}, {:.3g}] {:d}\n".format(
                        self.aacgm_n, self.aacgm_e, self.aacgm_z, self.dat_ind)
                elif self.aacgm_mag.shape != ():
                    out += "AACGM: {:.3g} [".format(self.aacgm_mag[i])
                    out += "{:.3g}, {:.3g}, {:.3g}] ".format(
                        self.aacgm_n[i], self.aacgm_e[i], self.aacgm_z[i])
                    out += "{:d}\n".format(self.dat_ind[i])

                if not np.isnan(mag):
                    out += "  OCB: {:.3g} [{:.3g}, ".format(mag, self.ocb_n[i])
                    out += "{:.3g}, ".format(self.ocb_e[i])
                    out += "{:.3g}] {:d}\n".format(
                        self.ocb_z[i], self.ocb_ind if self.ocb_ind.shape == ()
                        else self.ocb_ind[i])

        out += "\n-------------------------------------------\n"
        if self.scale_func is None:
            out += "No magnitude scaling function provided\n"
        else:
            out += "Scaling function: {:s}\n".format(self.scale_func.__name__)

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

        # Use Object to avoid recursion
        super(VectorData, self).__setattr__(name, out_val)
        return

    def _ocb_attr_setter(self, ocb_name, ocb_val):
        """Set OCB attributes.

        Parameters
        ----------
        ocb_name : str
            OCB attribute name
        value
            Value (any type) to be assigned to attribute specified by name

        """
        # Ensure the shape is correct
        if np.asarray(ocb_val).shape == () and self.ocb_ind.shape != ():
            ocb_val = np.full(shape=self.ocb_ind.shape, fill_value=ocb_val)

        self.__setattr__(ocb_name, ocb_val)
        return

    def _test_update_vector_shape(self):
        """Test and update the shape of the VectorData attributes.

        Raises
        ------
        ValueError
            If mismatches in the attribute shapes are encountered

        Notes
        -----
        Sets the `vshape` attribute and updates the shape of `aacgm_n`,
        `aacgm_e`, and `aacgm_z` if needed

        """

        # Get the required input shapes
        vshapes = [self.aacgm_lat.shape, self.aacgm_mlt.shape,
                   self.dat_ind.shape, self.aacgm_n.shape, self.aacgm_e.shape,
                   self.aacgm_z.shape]
        vshapes = np.unique(np.asarray(vshapes, dtype=object))

        # Determine the desired shape
        self.vshape = () if len(vshapes) == 0 else vshapes.max()

        # Evaluate for potential mismatched attributes
        if len(vshapes) > 2 or (len(vshapes) == 2 and min(vshapes) != ()):
            raise ValueError('mismatched dimensions for VectorData inputs')

        if len(vshapes) > 1 and min(vshapes) == ():
            if self.dat_ind.shape == ():
                raise ValueError('data index shape must match vector shape')

            # Vector input needs to be the same length
            if self.aacgm_n.shape == ():
                self.aacgm_n = np.full(shape=self.vshape,
                                       fill_value=self.aacgm_n)
            if self.aacgm_e.shape == ():
                self.aacgm_e = np.full(shape=self.vshape,
                                       fill_value=self.aacgm_e)
            if self.aacgm_z.shape == ():
                self.aacgm_z = np.full(shape=self.vshape,
                                       fill_value=self.aacgm_z)
        return

    def _test_update_bound_shape(self):
        """Test and update the shape of the VectorData boundary attributes.

        Raises
        ------
        ValueError
            If mismatches in the attribute shapes are encountered

        Notes
        -----
        Updates the shape of `aacgm_n`, `aacgm_e`, and `aacgm_z` if needed

        """
        # Test the OCB input shape
        oshapes = np.unique([self.ocb_lat.shape, self.ocb_mlt.shape,
                             self.r_corr.shape])
        oshape = () if len(oshapes) == 0 else oshapes.max()

        if(oshape != self.ocb_ind.shape or len(oshapes) > 2
           or (len(oshapes) == 2 and min(oshapes) != ())):
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

    @property
    def aacgm_mag(self):
        """Magntiude of the AACGM vector(s)."""
        return self._aacgm_mag

    @aacgm_mag.setter
    def aacgm_mag(self, aacgm_mag):
        # Assign the vector magnitude(s)
        aacgm_sqrt = np.sqrt(self.aacgm_n**2 + self.aacgm_e**2
                             + self.aacgm_z**2)

        if np.all(np.isnan(aacgm_mag)):
            self._aacgm_mag = aacgm_sqrt
        else:
            if np.any(np.greater(abs(aacgm_mag - aacgm_sqrt), 1.0e-3,
                                 where=~np.isnan(aacgm_mag))):
                ocbpy.logger.warning("".join(["inconsistent AACGM components ",
                                              "with a maximum difference of ",
                                              "{:} > 1.0e-3".format(
                                                  abs(aacgm_mag
                                                      - aacgm_sqrt).max())]))
            self._aacgm_mag = aacgm_mag
        return

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

            # Re-calculate the AACGM magnitude
            self.aacgm_mag = np.nan
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

    def clear_data(self):
        """Clear or initialize the output data attributes."""

        # Assign the OCB vector default values and location
        self.ocb_n = np.full(shape=self.vshape, fill_value=np.nan)
        self.ocb_e = np.full(shape=self.vshape, fill_value=np.nan)
        self.ocb_z = np.full(shape=self.vshape, fill_value=np.nan)
        self.ocb_mag = np.full(shape=self.vshape, fill_value=np.nan)

        # Assign the default pole locations, relative angles, and quadrants
        self.ocb_quad = np.zeros(shape=self.vshape)
        self.vec_quad = np.zeros(shape=self.vshape)
        self.pole_angle = np.full(shape=self.vshape, fill_value=np.nan)
        self.aacgm_naz = np.full(shape=self.vshape, fill_value=np.nan)
        self.ocb_aacgm_lat = np.full(shape=self.vshape, fill_value=np.nan)
        self.ocb_aacgm_mlt = np.full(shape=self.vshape, fill_value=np.nan)
        return

    def set_ocb(self, ocb, scale_func=None):
        """Set the OCBoundary values for provided data (updates all attributes).

        Parameters
        ----------
        ocb : ocbpy.OCBoundary or ocbpy.DualBoundary
            OCB, EAB, or Dual boundary object
        scale_func : function
            Function for scaling AACGM magnitude with arguments:
            [measurement value, mesurement AACGM latitude (degrees),
            mesurement OCB latitude (degrees)]
            Not necessary if defined earlier or no scaling is needed.
            (default=None)

        """

        # If the OCB vector coordinates weren't included in the initial info,
        # update them here
        if(np.all(np.isnan(self.ocb_lat)) or np.all(np.isnan(self.ocb_mlt))
           or np.all(np.isnan(self.r_corr))):
            # Because the OCB and AACGM magnetic field are both time dependent,
            # can't call this function with multiple OCBs
            if self.ocb_ind.shape == ():
                # Initialise the OCB index
                ocb.rec_ind = self.ocb_ind

                # Calculate the coordinates and save the output
                out_coord = ocb.normal_coord(self.aacgm_lat, self.aacgm_mlt)

                if len(out_coord) == 3:
                    (self.ocb_lat, self.ocb_mlt, self.r_corr) = out_coord
                else:
                    (self.ocb_lat, self.ocb_mlt, _, self.r_corr) = out_coord
            else:
                # Cycle through the OCB indices
                for i, ocb.rec_ind in enumerate(self.ocb_ind):
                    # Calcualte the coordinates and save the output
                    if self.ocb_ind.shape == self.dat_ind.shape:
                        out_coord = ocb.normal_coord(self.aacgm_lat[i],
                                                     self.aacgm_mlt[i])
                    else:
                        out_coord = ocb.normal_coord(self.aacgm_lat,
                                                     self.aacgm_mlt)

                    if len(out_coord) == 3:
                        (self.ocb_lat[i], self.ocb_mlt[i],
                         self.r_corr[i]) = out_coord
                    else:
                        (self.ocb_lat[i], self.ocb_mlt[i], _,
                         self.r_corr[i]) = out_coord

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
            self.ocb_aacgm_mlt = ocbpy.ocb_time.deg2hr(
                ocb.ocb.phi_cent[iocb])
            self.ocb_aacgm_lat = 90.0 - ocb.ocb.r_cent[iocb]
        else:
            self.unscaled_r = ocb.r[self.ocb_ind] + self.r_corr
            self.scaled_r = np.full(shape=self.unscaled_r.shape,
                                    fill_value=(90.0 - abs(ocb.boundary_lat)))
            self.ocb_aacgm_mlt = ocbpy.ocb_time.deg2hr(
                ocb.phi_cent[self.ocb_ind])
            self.ocb_aacgm_lat = 90.0 - ocb.r_cent[self.ocb_ind]

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

        Requires `ocb_aacgm_mlt`, `aacgm_mlt`, and `pole_angle`.
        Updates `ocb_quad` and `vec_quad`

        Raises
        ------
        ValueError
            If the required input is undefined

        """

        # Cast the input as arrays
        self.ocb_aacgm_mlt = np.asarray(self.ocb_aacgm_mlt)
        self.aacgm_mlt = np.asarray(self.aacgm_mlt)
        self.pole_angle = np.asarray(self.pole_angle)

        # Test input
        if np.all(np.isnan(self.ocb_aacgm_mlt)):
            raise ValueError("OCB pole location required")

        if np.all(np.isnan(self.aacgm_mlt)):
            raise ValueError("Vector AACGM location required")

        if np.all(np.isnan(self.pole_angle)):
            raise ValueError("vector angle in poles-vector triangle required")

        # Determine where the OCB pole is relative to the data vector
        ocb_adj_mlt = self.ocb_aacgm_mlt - self.aacgm_mlt

        neg_mask = (np.less(ocb_adj_mlt, 0.0, where=~np.isnan(ocb_adj_mlt))
                    & ~np.isnan(ocb_adj_mlt))
        while np.any(neg_mask):
            if ocb_adj_mlt.shape == ():
                ocb_adj_mlt += 24.0
                neg_mask = [False]
            else:
                ocb_adj_mlt[neg_mask] += 24.0
                neg_mask = (np.less(ocb_adj_mlt, 0.0,
                                    where=~np.isnan(ocb_adj_mlt))
                            & ~np.isnan(ocb_adj_mlt))

        large_mask = (np.greater_equal(abs(ocb_adj_mlt), 24.0,
                                       where=~np.isnan(ocb_adj_mlt))
                      & ~np.isnan(ocb_adj_mlt))
        if np.any(large_mask):
            if ocb_adj_mlt.shape == ():
                ocb_adj_mlt -= 24.0 * np.sign(ocb_adj_mlt)
            else:
                ocb_adj_mlt[large_mask] -= 24.0 * np.sign(
                    ocb_adj_mlt[large_mask])

        # Find the quadrant in which the OCB pole lies
        nan_mask = (~np.isnan(self.pole_angle) & ~np.isnan(ocb_adj_mlt))
        quad1_mask = (np.less(self.pole_angle, 90.0, where=nan_mask)
                      & np.less(ocb_adj_mlt, 12.0, where=nan_mask) & nan_mask)
        quad2_mask = (np.less(self.pole_angle, 90.0, where=nan_mask)
                      & np.greater_equal(ocb_adj_mlt, 12.0, where=nan_mask)
                      & nan_mask)
        quad3_mask = (np.greater_equal(self.pole_angle, 90.0, where=nan_mask)
                      & np.greater_equal(ocb_adj_mlt, 12.0, where=nan_mask)
                      & nan_mask)
        quad4_mask = (np.greater_equal(self.pole_angle, 90.0, where=nan_mask)
                      & np.less(ocb_adj_mlt, 12.0, where=nan_mask) & nan_mask)

        if self.ocb_quad.shape == ():
            if np.all(quad1_mask):
                self.ocb_quad = np.asarray(1)
            elif np.all(quad2_mask):
                self.ocb_quad = np.asarray(2)
            elif np.all(quad3_mask):
                self.ocb_quad = np.asarray(3)
            elif np.all(quad4_mask):
                self.ocb_quad = np.asarray(4)
        else:
            self.ocb_quad[quad1_mask] = 1
            self.ocb_quad[quad2_mask] = 2
            self.ocb_quad[quad3_mask] = 3
            self.ocb_quad[quad4_mask] = 4

        # Now determine which quadrant the vector is pointed into
        nan_mask = (~np.isnan(self.aacgm_n) & ~np.isnan(self.aacgm_e))
        quad1_mask = (np.greater_equal(self.aacgm_n, 0.0, where=nan_mask)
                      & np.greater_equal(self.aacgm_e, 0.0, where=nan_mask)
                      & nan_mask)
        quad2_mask = (np.greater_equal(self.aacgm_n, 0.0, where=nan_mask)
                      & np.less(self.aacgm_e, 0.0, where=nan_mask) & nan_mask)
        quad3_mask = (np.less(self.aacgm_n, 0.0, where=nan_mask)
                      & np.less(self.aacgm_e, 0.0, where=nan_mask) & nan_mask)
        quad4_mask = (np.less(self.aacgm_n, 0.0, where=nan_mask)
                      & np.greater_equal(self.aacgm_e, 0.0, where=nan_mask)
                      & nan_mask)

        if self.vec_quad.shape == ():
            if np.all(quad1_mask):
                self.vec_quad = np.asarray(1)
            elif np.all(quad2_mask):
                self.vec_quad = np.asarray(2)
            elif np.all(quad3_mask):
                self.vec_quad = np.asarray(3)
            elif np.all(quad4_mask):
                self.vec_quad = np.asarray(4)
        else:
            self.vec_quad[quad1_mask] = 1
            self.vec_quad[quad2_mask] = 2
            self.vec_quad[quad3_mask] = 3
            self.vec_quad[quad4_mask] = 4

        return

    def scale_vector(self):
        """Normalise a variable proportional to the curl of the electric field.

        Raises
        ------
        ValueError
            If the required input is not defined

        Notes
        -----
        Requires `ocb_lat`, `ocb_mlt`, `ocb_aacgm_mlt`, and `pole_angle`.
        Updates `ocb_n`, `ocb_e`, `ocb_z`, and `ocb_mag`

        """

        # Ensure the input is array-like
        self.ocb_lat = np.asarray(self.ocb_lat)
        self.ocb_mlt = np.asarray(self.ocb_mlt)
        self.ocb_aacgm_mlt = np.asarray(self.ocb_aacgm_mlt)
        self.pole_angle = np.asarray(self.pole_angle)
        self.aacgm_n = np.asarray(self.aacgm_n)
        self.aacgm_e = np.asarray(self.aacgm_e)
        self.aacgm_z = np.asarray(self.aacgm_z)
        self.ocb_quad = np.asarray(self.ocb_quad)
        self.vec_quad = np.asarray(self.vec_quad)

        # Test input
        if np.all(np.isnan(self.ocb_lat)) or np.all(np.isnan(self.ocb_mlt)):
            raise ValueError("OCB coordinates required")

        if np.all(np.isnan(self.ocb_aacgm_mlt)):
            raise ValueError("OCB pole location required")

        if np.all(np.isnan(self.pole_angle)):
            raise ValueError("vector angle in poles-vector triangle required")

        # Determine the special case assignments
        zero_mask = ((self.aacgm_n == 0.0) & (self.aacgm_e == 0.0))
        ns_mask = ((self.pole_angle == 0.0) | (self.pole_angle == 180.0))
        norm_mask = ~(zero_mask + ns_mask)

        # There's no magnitude, so nothing to adjust
        if np.any(zero_mask):
            if self.aacgm_n.shape == ():
                self.ocb_n = np.zeros(shape=self.ocb_n.shape)
                self.ocb_e = np.zeros(shape=self.ocb_e.shape)
                self.ocb_z = np.zeros(shape=self.ocb_z.shape)
            else:
                self.ocb_n[zero_mask] = 0.0
                self.ocb_e[zero_mask] = 0.0
                self.ocb_z[zero_mask] = 0.0

        # The measurement is aligned with the AACGM and OCB poles
        if np.any(ns_mask):
            if self.scale_func is None:
                if self.aacgm_n.shape == ():
                    self.ocb_n = np.full(shape=self.ocb_n.shape,
                                         fill_value=self.aacgm_n)
                    self.ocb_e = np.full(shape=self.ocb_e.shape,
                                         fill_value=self.aacgm_e)
                    self.ocb_z = np.full(shape=self.ocb_z.shape,
                                         fill_value=self.aacgm_z)
                else:
                    self.ocb_n[ns_mask] = self.aacgm_n[ns_mask]
                    self.ocb_e[ns_mask] = self.aacgm_e[ns_mask]
                    self.ocb_z[ns_mask] = self.aacgm_z[ns_mask]
            else:
                if self.aacgm_n.shape == ():
                    self.ocb_n = np.full(shape=self.ocb_n.shape,
                                         fill_value=self.scale_func(
                                             self.aacgm_n, self.unscaled_r,
                                             self.scaled_r))
                    self.ocb_e = np.full(shape=self.ocb_e.shape,
                                         fill_value=self.scale_func(
                                             self.aacgm_e, self.unscaled_r,
                                             self.scaled_r))
                    self.ocb_z = np.full(shape=self.ocb_z.shape,
                                         fill_value=self.scale_func(
                                             self.aacgm_z, self.unscaled_r,
                                             self.scaled_r))
                else:
                    self.ocb_n[ns_mask] = self.scale_func(
                        self.aacgm_n[ns_mask], self.unscaled_r[ns_mask],
                        self.scaled_r[ns_mask])
                    self.ocb_e[ns_mask] = self.scale_func(
                        self.aacgm_e[ns_mask], self.unscaled_r[ns_mask],
                        self.scaled_r[ns_mask])
                    self.ocb_z[ns_mask] = self.scale_func(
                        self.aacgm_z[ns_mask], self.unscaled_r[ns_mask],
                        self.scaled_r[ns_mask])

            # Determine if the measurement is on or between the poles
            # This does not affect the vertical direction
            sign_mask = ((self.pole_angle == 0.0)
                         & np.greater_equal(self.aacgm_lat, self.ocb_aacgm_lat,
                                            where=~np.isnan(self.aacgm_lat))
                         & ~np.isnan(self.aacgm_lat))
            if np.any(sign_mask):
                if self.ocb_n.shape == ():
                    self.ocb_n *= -1.0
                    self.ocb_e *= -1.0
                else:
                    self.ocb_n[sign_mask] *= -1.0
                    self.ocb_e[sign_mask] *= -1.0

        # If there are still undefined vectors, assign them using the
        # typical case
        if np.any(norm_mask):
            # If not defined, get the OCB and vector quadrants
            if(np.any(self.ocb_quad[norm_mask] == 0)
               or np.any(self.vec_quad[norm_mask] == 0)):
                self.define_quadrants()

            # Get the unscaled 2D vector magnitude and
            # calculate the AACGM north azimuth in degrees
            if self.aacgm_n.shape == ():
                vmag = np.sqrt(self.aacgm_n**2 + self.aacgm_e**2)
                self.aacgm_naz = np.degrees(np.arccos(self.aacgm_n / vmag))
            else:
                vmag = np.sqrt(self.aacgm_n[norm_mask]**2
                               + self.aacgm_e[norm_mask]**2)
                self.aacgm_naz[norm_mask] = np.degrees(
                    np.arccos(self.aacgm_n[norm_mask] / vmag))

            # Get the OCB north azimuth in radians
            ocb_angle = np.radians(self.calc_ocb_polar_angle())

            # Get the sign of the North and East components
            vsigns = self.calc_ocb_vec_sign(north=True, east=True)

            # Scale the vector along the OCB north and account for
            # any changes associated with adjusting the size of the polar cap
            if self.scale_func is not None:
                if self.unscaled_r.shape == ():
                    un_r = self.unscaled_r
                    sc_r = self.scaled_r
                else:
                    un_r = self.unscaled_r[norm_mask]
                    sc_r = self.scaled_r[norm_mask]

                if self.aacgm_z.shape == ():
                    a_z = self.aacgm_z
                else:
                    a_z = self.aacgm_z[norm_mask]

                vmag = self.scale_func(vmag, un_r, sc_r)
                vz = self.scale_func(a_z, un_r, sc_r)
            else:
                if self.aacgm_z.shape == ():
                    vz = self.aacgm_z
                else:
                    vz = self.aacgm_z[norm_mask]
                    nan_mask = (np.isnan(vmag)
                                | (np.isnan(ocb_angle) if ocb_angle.shape == ()
                                   else np.isnan(ocb_angle[norm_mask])))
                    vz[nan_mask] = np.nan

            # Restrict the OCB angle to result in positive sines and cosines
            lmask = ocb_angle > np.pi / 2.0
            if np.any(lmask):
                if ocb_angle.shape == ():
                    ocb_angle = np.pi - ocb_angle
                else:
                    ocb_angle[lmask] = np.pi - ocb_angle[lmask]

            # Calculate the vector components
            if vmag.shape == ():
                self.ocb_n = np.full(shape=self.ocb_n.shape,
                                     fill_value=(vsigns['north'] * vmag
                                                 * np.cos(ocb_angle)))
                self.ocb_e = np.full(shape=self.ocb_e.shape,
                                     fill_value=(vsigns['east'] * vmag
                                                 * np.sin(ocb_angle)))
                self.ocb_z = np.full(shape=self.ocb_z.shape, fill_value=vz)
            else:
                self.ocb_n[norm_mask] = (vsigns['north'][norm_mask] * vmag
                                         * np.cos(ocb_angle[norm_mask]))
                self.ocb_e[norm_mask] = (vsigns['east'][norm_mask] * vmag
                                         * np.sin(ocb_angle[norm_mask]))
                self.ocb_z[norm_mask] = vz

        # Calculate the scaled OCB vector magnitude
        self.ocb_mag = np.sqrt(self.ocb_n**2 + self.ocb_e**2
                               + self.ocb_z**2)

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

        quad_range = np.arange(1, 5)

        # Test input
        if not np.any(np.isin(self.ocb_quad, quad_range)):
            raise ValueError("OCB quadrant undefined")

        if not np.any(np.isin(self.vec_quad, quad_range)):
            raise ValueError("Vector quadrant undefined")

        if np.all(np.isnan(self.aacgm_naz)):
            raise ValueError("AACGM polar angle undefined")

        if np.all(np.isnan(self.pole_angle)):
            raise ValueError("Vector angle undefined")

        # Initialise the output and set the quadrant dictionary
        nan_mask = (~np.isnan(self.aacgm_naz) & ~np.isnan(self.pole_angle))
        ocb_naz = np.full(shape=(self.aacgm_naz + self.pole_angle).shape,
                          fill_value=np.nan)
        quads = {oquad: {vquad: (self.ocb_quad == oquad)
                         & (self.vec_quad == vquad) & nan_mask
                         for vquad in quad_range} for oquad in quad_range}

        # Create masks for the different quadrant combinations
        abs_mask = (quads[1][1] | quads[2][2] | quads[3][3] | quads[4][4])
        add_mask = (quads[1][2] | quads[1][3] | quads[2][1] | quads[2][4]
                    | quads[3][1] | quads[4][2])
        mpa_mask = (quads[1][4] | quads[2][3])
        maa_mask = (quads[3][2] | quads[4][1])
        cir_mask = (quads[3][4] | quads[4][3])

        # Calculate OCB polar angle based on the quadrants and other angles
        if np.any(abs_mask):
            if ocb_naz.shape == ():
                ocb_naz = abs(self.aacgm_naz - self.pole_angle)
            else:
                ocb_naz[abs_mask] = abs(self.aacgm_naz
                                        - self.pole_angle)[abs_mask]

        if np.any(add_mask):
            if ocb_naz.shape == ():
                ocb_naz = self.pole_angle + self.aacgm_naz
                if ocb_naz > 180.0:
                    ocb_naz = 360.0 - ocb_naz
            else:
                ocb_naz[add_mask] = (self.pole_angle
                                     + self.aacgm_naz)[add_mask]
                lmask = (ocb_naz > 180.0) & add_mask
                if np.any(lmask):
                    ocb_naz[lmask] = 360.0 - ocb_naz[lmask]

        if np.any(mpa_mask):
            if ocb_naz.shape == ():
                ocb_naz = self.aacgm_naz - self.pole_angle
            else:
                ocb_naz[mpa_mask] = (self.aacgm_naz
                                     - self.pole_angle)[mpa_mask]

        if np.any(maa_mask):
            if ocb_naz.shape == ():
                ocb_naz = self.pole_angle - self.aacgm_naz
            else:
                ocb_naz[maa_mask] = (self.pole_angle
                                     - self.aacgm_naz)[maa_mask]

        if np.any(cir_mask):
            if ocb_naz.shape == ():
                ocb_naz = 360.0 - self.aacgm_naz - self.pole_angle
            else:
                ocb_naz[cir_mask] = (360.0 - self.aacgm_naz
                                     - self.pole_angle)[cir_mask]

        return ocb_naz

    def calc_ocb_vec_sign(self, north=False, east=False, quads=dict()):
        """Calculate the sign of the North and East components.

        Parameters
        ----------
        north : bool
            Get the sign of the north component(s) (default=False)
        east : bool
            Get the sign of the east component(s) (default=False)
        quads : dict
            Dictionary of boolean values or arrays of boolean values for OCB
            and Vector quadrants. (default=dict())

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
        Requires `ocb_quad`, `vec_quad`, `aacgm_naz`, and `pole_angle`

        """

        quad_range = np.arange(1, 5)

        # Ensure the required input is array-like
        self.ocb_quad = np.asarray(self.ocb_quad)
        self.vec_quad = np.asarray(self.vec_quad)
        self.aacgm_naz = np.asarray(self.aacgm_naz)
        self.pole_angle = np.asarray(self.pole_angle)

        # Test input
        if not np.any([north, east]):
            raise ValueError("must set at least one direction")

        if not np.any(np.isin(self.ocb_quad, quad_range)):
            raise ValueError("OCB quadrant undefined")

        if not np.any(np.isin(self.vec_quad, quad_range)):
            raise ValueError("Vector quadrant undefined")

        if np.all(np.isnan(self.aacgm_naz)):
            raise ValueError("AACGM polar angle undefined")

        if np.all(np.isnan(self.pole_angle)):
            raise ValueError("Vector angle undefined")

        # If necessary, initialise quadrant dictionary
        nan_mask = (~np.isnan(self.aacgm_naz) & ~np.isnan(self.pole_angle))
        if not np.all([kk in quads.keys() for kk in quad_range]):
            quads = {o: {v: (self.ocb_quad == o) & (self.vec_quad == v)
                         & nan_mask for v in quad_range} for o in quad_range}

        # Initialise output
        vsigns = {"north": np.zeros(shape=quads[1][1].shape),
                  "east": np.zeros(shape=quads[1][1].shape)}

        # Determine the desired vector signs
        if north:
            pole_minus = self.pole_angle - 90.0
            minus_pole = 90.0 - self.pole_angle
            pole_plus = self.pole_angle + 90.0

            pmask = (quads[1][1] | quads[2][2] | quads[3][3] | quads[4][4]
                     | ((quads[1][4] | quads[2][3])
                        & np.less_equal(self.aacgm_naz, pole_plus,
                                        where=nan_mask))
                     | ((quads[1][2] | quads[2][1])
                        & np.less_equal(self.aacgm_naz, minus_pole,
                                        where=nan_mask))
                     | ((quads[3][4] | quads[4][3])
                        & np.greater_equal(self.aacgm_naz, 180.0 - pole_minus,
                                           where=nan_mask))
                     | ((quads[3][2] | quads[4][1])
                        & np.greater_equal(self.aacgm_naz, pole_minus,
                                           where=nan_mask)))

            if np.any(pmask):
                if vsigns["north"].shape == ():
                    vsigns["north"] = 1
                else:
                    vsigns["north"][pmask] = 1

            if np.any(~pmask):
                if vsigns["north"].shape == ():
                    vsigns["north"] = -1
                else:
                    vsigns["north"][~pmask] = -1

        if east:
            minus_pole = 180.0 - self.pole_angle

            pmask = (quads[1][4] | quads[2][1] | quads[3][2] | quads[4][3]
                     | ((quads[1][1] | quads[4][4])
                        & np.greater_equal(self.aacgm_naz, self.pole_angle,
                                           where=nan_mask))
                     | ((quads[3][1] | quads[2][4])
                        & np.less_equal(self.aacgm_naz, minus_pole,
                                        where=nan_mask))
                     | ((quads[4][2] | quads[1][3])
                        & np.greater_equal(self.aacgm_naz, minus_pole,
                                           where=nan_mask))
                     | ((quads[2][2] | quads[3][3])
                        & np.less_equal(self.aacgm_naz, self.pole_angle,
                                        where=nan_mask)))

            if np.any(pmask):
                if vsigns["east"].shape == ():
                    vsigns["east"] = 1
                else:
                    vsigns["east"][pmask] = 1

            if np.any(~pmask):
                if vsigns["east"].shape == ():
                    vsigns["east"] = -1
                else:
                    vsigns["east"][~pmask] = -1

        return vsigns

    def calc_vec_pole_angle(self):
        """Calc the angle between the AACGM pole, data, and the OCB pole.

        Raises
        ------
        ValueError
            If the input is undefined or inappropriately sized arrays

        Notes
        -----
        Requires `aacgm_mlt`, `aacgm_lat`, `ocb_aacgm_mlt`, and `ocb_aacgm_lat`.
        Updates `pole_angle` using spherical trigonometry.

        """

        # Cast inputs as arrays
        self.aacgm_mlt = np.asarray(self.aacgm_mlt)
        self.aacgm_lat = np.asarray(self.aacgm_lat)
        self.ocb_aacgm_mlt = np.asarray(self.ocb_aacgm_mlt)
        self.ocb_aacgm_lat = np.asarray(self.ocb_aacgm_lat)

        # Test input
        if np.all(np.isnan(self.aacgm_mlt)):
            raise ValueError("AACGM MLT of Vector(s) undefinded")

        if np.all(np.isnan(self.aacgm_lat)):
            raise ValueError("AACGM latitude of Vector(s) undefined")

        if np.all(np.isnan(self.ocb_aacgm_mlt)):
            raise ValueError("AACGM MLT of OCB pole(s) undefined")

        if np.all(np.isnan(self.ocb_aacgm_lat)):
            raise ValueError("AACGM latitude of OCB pole(s) undefined")

        # Convert the AACGM MLT of the observation and OCB pole to radians,
        # then calculate the difference between them.
        del_long = ocbpy.ocb_time.hr2rad(self.ocb_aacgm_mlt - self.aacgm_mlt)

        if del_long.shape == ():
            if del_long < -np.pi:
                del_long += 2.0 * np.pi
        else:
            del_long[del_long < -np.pi] += 2.0 * np.pi

        # Initalize the output
        self.pole_angle = np.full(shape=del_long.shape, fill_value=np.nan)

        # Assign the extreme values
        if del_long.shape == ():
            if del_long in [-np.pi, 0.0, np.pi]:
                if abs(self.aacgm_lat) > abs(self.ocb_aacgm_lat):
                    self.pole_angle = 180.0
                else:
                    self.pole_angle = 0.0
                return
        else:
            zero_mask = (((del_long == 0) | (abs(del_long) == np.pi))
                         & np.greater(abs(self.aacgm_lat),
                                      abs(self.ocb_aacgm_lat),
                                      where=~np.isnan(del_long)))
            flat_mask = (((del_long == 0) | (abs(del_long) == np.pi))
                         & np.less_equal(abs(self.aacgm_lat),
                                         abs(self.ocb_aacgm_lat),
                                         where=~np.isnan(del_long)))

            self.pole_angle[flat_mask] = 180.0
            self.pole_angle[zero_mask] = 0.0
            update_mask = (~zero_mask & ~flat_mask)

            if not np.any(update_mask):
                return

        # Find the distance in radians between the two poles
        hemisphere = np.sign(self.ocb_aacgm_lat)
        rad_pole = hemisphere * np.pi * 0.5
        del_pole = hemisphere * (rad_pole - np.radians(self.ocb_aacgm_lat))

        # Get the distance in radians between the AACGM pole and the data point
        del_vect = hemisphere * (rad_pole - np.radians(self.aacgm_lat))

        # Use the Vincenty formula for a sphere
        del_ocb = np.arctan2(np.sqrt((np.cos(np.radians(self.ocb_aacgm_lat))
                                      * np.sin(del_long))**2
                                     + (np.cos(np.radians(self.aacgm_lat))
                                        * np.sin(
                                            np.radians(self.ocb_aacgm_lat))
                                        - np.sin(np.radians(self.aacgm_lat))
                                        * np.cos(
                                            np.radians(self.ocb_aacgm_lat))
                                        * np.cos(del_long))**2),
                             np.sin(np.radians(self.aacgm_lat))
                             * np.sin(np.radians(self.ocb_aacgm_lat))
                             + np.cos(np.radians(self.aacgm_lat))
                             * np.cos(np.radians(self.ocb_aacgm_lat))
                             * np.cos(del_long))

        # Use the half-angle formula to get the pole angle
        sum_sides = 0.5 * (del_vect + del_ocb + del_pole)
        half_angle = np.sqrt(np.sin(sum_sides) * np.sin(sum_sides - del_pole)
                             / (np.sin(del_vect) * np.sin(del_ocb)))

        if self.pole_angle.shape == ():
            self.pole_angle = np.degrees(2.0 * np.arccos(half_angle))
        else:
            self.pole_angle[update_mask] = np.degrees(
                2.0 * np.arccos(half_angle[update_mask]))

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
