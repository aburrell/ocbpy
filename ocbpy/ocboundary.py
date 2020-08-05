#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB & GC
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Hold, manipulate, and load the open-closed field line boundary data

Functions
---------
retrieve_all_good_indices(ocb)
    Retrieve all good boundary indices
match_data_ocb(ocb, dat_dtime, kwargs)
    Match data with open-closed field line boundaries

Classes
-------
OCBoundary  Loads, holds, and cycles the open-closed field line boundary data.
            Calculates magnetic coordinates relative to OCB (setting OCB at
            74 degrees latitude) given an AACGM location.

Moduleauthor
------------
Angeline G. Burrell (AGB), 15 April 2017, University of Texas, Dallas

References
----------
Chisham, G. (2017), A new methodology for the development of high-latitude
 ionospheric climatologies and empirical models, Journal of Geophysical
 Research: Space Physics, 122, doi:10.1002/2016JA023235.

"""
from __future__ import absolute_import, unicode_literals

import datetime as dt
import numpy as np
import types

import aacgmv2

import ocbpy
import ocbpy.ocb_correction as ocbcor
from ocbpy.ocb_time import slt2glon, convert_time, glon2slt, deg2hr, fix_range
from ocbpy.boundaries.files import get_default_file


class OCBoundary(object):
    """ Object containing open-closed field-line boundary (OCB) data

    Attributes
    ----------
    filename : (str or NoneType)
        OCBoundary filename or None, if problem loading default
    boundary_lat : (float)
        Typical OCBoundary latitude in AACGM coordinates.  Hemisphere will
        give this boundary the desired sign.  (default=74.0)
    hemisphere : (int)
        Integer (+/- 1) denoting northern/southern hemisphere (default=1)
    records : (int)
        Number of OCB records (default=0)
    rec_ind : (int)
        Current OCB record index (default=0; initialised=-1)
    dtime : (numpy.ndarray or NoneType)
        Numpy array of OCB datetimes (default=None)
    phi_cent : (numpy.ndarray or NoneType)
        Numpy array of floats that give the angle from AACGM midnight
        of the OCB pole in degrees (default=None)
    r_cent : (numpy.ndarray or NoneType)
        Numpy array of floats that give the AACGM co-latitude of the OCB
        pole in degrees (default=None)
    r : (numpy.ndarray or NoneType)
        Numpy array of floats that give the radius of the OCBoundary
        in degrees (default=None)
    rfunc : (numpy.ndarray, function, or NoneType)
        Non-circular boundaries must be specified by a boundary function that
        alters r at a specified AACGM MLT (in hours).  To allow the boundary
        shape to change with time, each temporal instance may have a different
        function. If a single function is provided, will recast as an array
        that specifies this function for all times (default=None)
    rfunc_kwargs : (numpy.ndarray, dict, or NoneType)
        Array of optional keyword arguements for rfunc. If None is specified,
        uses function defaults.  If dict is specified, recasts as an array
        of this dict for all times (default=None)
    (more) : (numpy.ndarray or NoneType)
        Numpy array of floats that hold the remaining values in input file

    Methods
    -------
    inst_defaults()
        Get the information needed to load an OCB file using instrument
        specific formatting, and update the boundary latitude for a given
        instrument type.
    load(hlines=0, ocb_cols='year soy num_sectors phi_cent r_cent r a r_err',
         datetime_fmt='', stime=None, etime=None)
        Load the data from the OCB file specified by self.filename
    get_next_good_ocb_ind(min_sectors=7, rcent_dev=8.0, max_r=23.0, min_r=10.0)
        Cycle to the next good OCB index
    normal_coord(aacgm_lat, aacgm_mlt)
        Calculate the OCB coordinates of an AACGM location
    revert_coord(ocb_lat, ocb_mlt)
        Calculate the AACGM location of OCB coordinates for this OCB

    """

    def __init__(self, filename="default", instrument='', hemisphere=1,
                 boundary_lat=74.0, stime=None, etime=None, rfunc=None,
                 rfunc_kwargs=None):
        """Object containing OCB data

        Parameters
        ----------
        filename : (str or NoneType)
            File containing the required open-closed circle boundary data
            sorted by time.  If NoneType, no file is loaded.  If 'default',
            ocbpy.boundaries.files.get_default_file is called.
            (default='default')
        instrument : (str)
            Instrument providing the OCBoundaries.  Requires 'image', 'ampere',
            or 'dmsp-ssj' if a file is provided.  If using filename='default',
            also accepts 'amp', 'si12', 'si13', 'wic', and ''.  (default='')
        hemisphere : (int)
            Integer (+/- 1) denoting northern/southern hemisphere (default=1)
        boundary_lat : (float)
            Latitude of the OCBoundary, which determines the resolution of
            data within the polar cap (default=74.0)
        stime : (datetime or NoneType)
            First time to load data or beginning of file.  If specifying time,
            be sure to start before the time of the data to allow the best
            match within the allowable time tolerance to be found.
            (default=None)
        etime : (datetime or NoneType)
            Last time to load data or ending of file.  If specifying time, be
            sure to end after the last data point you wish to match to, to
            ensure the best match within the allowable time tolerance is made.
            (default=None)
        min_fom : (float)
            Minimum acceptable figure of merit for data (default=0)
        rfunc : (np.ndarray, function, or NoneType)
            OCB radius correction function, if None will use instrument
            default. Function must have AACGM MLT as argument input.
            (default=None)
        rfunc_kwargs : (np.ndarray, dict, or NoneType)
            OCB radius correction function keyword arguments. (default={})

        """

        if not hasattr(instrument, "lower"):
            estr = "OCB instrument must be a string [{:}]".format(instrument)
            ocbpy.logger.error(estr)
            self.filename = None
            self.instrument = None
        else:
            self.instrument = instrument.lower()

            # If a filename wanted and not provided, get one
            if filename is None:
                self.filename = None
            elif not hasattr(filename, "lower"):
                estr = "filename is not a string [{:}]".format(filename)
                ocbpy.logger.warning(estr)
                self.filename = None
            elif filename.lower() == "default":
                self.filename, self.instrument = get_default_file(
                    stime, etime, hemisphere, self.instrument)
            else:
                self.filename = filename

            # If a filename is available, make sure it is good
            if self.filename is not None:
                if not ocbpy.instruments.test_file(self.filename):
                    # If the filename is bad, return an uninitialized object
                    estr = "cannot open OCB file [{:s}]".format(self.filename)
                    ocbpy.logger.warning(estr)
                    self.filename = None

        if hemisphere not in [1, -1]:
            raise ValueError("hemisphere must be 1 (north) or -1 (south)")

        self.hemisphere = hemisphere
        self.records = 0
        self.rec_ind = 0
        self.dtime = None
        self.phi_cent = None
        self.r_cent = None
        self.r = None
        self.rfunc = rfunc
        self.rfunc_kwargs = rfunc_kwargs
        self.min_fom = 0

        # Get the instrument defaults
        hlines, ocb_cols, datetime_fmt = self.inst_defaults()

        # Set the boundary latitude, if supplied
        self.boundary_lat = 74.0 if boundary_lat is None else boundary_lat

        # Ensure that the boundary is in the correct hemisphere
        if np.sign(boundary_lat) != np.sign(hemisphere):
            self.boundary_lat *= -1.0

        # If possible, load the data.  Any boundary correction is applied here.
        if self.filename is not None:
            if len(ocb_cols) > 0:
                self.load(hlines=hlines, ocb_cols=ocb_cols,
                          datetime_fmt=datetime_fmt, stime=stime, etime=etime)
            else:
                self.load(stime=stime, etime=etime)

        return

    def __repr__(self):
        """ Provide readable representation of the OCBoundary object """

        if self.filename is None:
            out = "No Open-Closed Boundary file specified\n"
        else:
            out = "Open-Closed Boundary file: {:s}\n".format(self.filename)
            out = "{:s}Source instrument: ".format(out)
            out = "{:s}{:s}\n".format(out, self.instrument.upper())
            out = "{:s}Open-Closed Boundary reference latitude: ".format(out)
            out = "{:s}{:.1f} degrees\n\n".format(out, self.boundary_lat)

            if self.records == 0:
                out = "{:s}No data loaded\n".format(out)
            else:
                out = "{:s}{:d} records from {:}".format(out, self.records,
                                                         self.dtime[0])
                out = "{:s} to {:}\n\n".format(out, self.dtime[-1])

                if self.records == 1:
                    irep = [0]
                else:
                    irep = np.unique(
                        np.arange(0, self.records, 1)[[0, 1, -2, -1]])

                head = "YYYY-MM-DD HH:MM:SS Phi_Centre R_Centre R"
                out = "{:s}{:s}\n{:-<77s}\n".format(out, head, "")
                for i in irep:
                    out = "{:s}{:} {:.2f}".format(out, self.dtime[i],
                                                  self.phi_cent[i])
                    out = "{:s} {:.2f} {:.2f}\n".format(out, self.r_cent[i],
                                                        self.r[i])

                # Determine which scaling functions are used
                if self.rfunc is not None:
                    out = "{:s}\nUses scaling function(s):\n".format(out)
                    fnames = list(set([ff.__name__ for ff in self.rfunc]))

                    for ff in fnames:
                        kw = list(set([self.rfunc_kwargs[i].__str__()
                                       for i, rf in enumerate(self.rfunc)
                                       if rf.__name__ == ff]))

                        for kk in kw:
                            out = "{:s}{:s}(**{:s})\n".format(out, ff, kk)

        return out

    def __str__(self):
        """ Provide readable representation of the OCBoundary object """

        out = self.__repr__()
        return out

    def inst_defaults(self):
        """ Get the information needed to load an OCB file using instrument
        specific formatting

        Returns
        -------
        hlines : (int)
            Number of header lines
        ocb_cols : (str)
            String containing the names for each data column
        datetime_fmt : (str)
            String containing the datetime format

        Notes
        -----
        Updates the min_fom attribute for AMPERE and DMSP-SSJ

        """

        if self.instrument == "image":
            hlines = 0
            ocb_cols = "year soy num_sectors phi_cent r_cent r a r_err"
            datetime_fmt = ""
        elif self.instrument == "ampere":
            hlines = 0
            ocb_cols = "date time r x y fom"
            datetime_fmt = "%Y%m%d %H:%M"
            self.min_fom = 0.15  # From Milan et al. (2015)
        elif self.instrument == "dmsp-ssj":
            hlines = 1
            ocb_cols = "sc date time r x y fom x_1 y_1 x_2 y_2"
            datetime_fmt = "%Y-%m-%d %H:%M:%S"
            self.min_fom = 3.0  # From Burrell et al. (2019)
        else:
            hlines = 0
            ocb_cols = ""
            datetime_fmt = ""

        return hlines, ocb_cols, datetime_fmt

    def load(self, hlines=0,
             ocb_cols="year soy num_sectors phi_cent r_cent r a r_err",
             datetime_fmt="", stime=None, etime=None):
        """Load the data from the specified Open-Closed Boundary file

        Parameters
        ----------
        ocb_cols : (str)
            String specifying format of OCB file.  All but the first two
            columns must be included in the string, additional data values will
            be ignored.  If 'year soy' aren't used, expects
            'date time' in 'YYYY-MM-DD HH:MM:SS' format.
            (default='year soy num_sectors phi_cent r_cent r a r_err')
        hlines : (int)
            Number of header lines preceeding data in the OCB file (default=0)
        datetime_fmt : (str)
            A string used to read in 'date time' data.  Not used if 'year soy'
            is specified. (default='')
        stime : (datetime or NoneType)
            Time to start loading data or None to start at beginning of file.
            (default=None)
        etime : (datetime or NoneType)
            Time to stop loading data or None to end at the end of the file.
            (default=None)

        """

        cols = ocb_cols.split()
        dflag = -1
        ldtype = [(k, float) if k != "num_sectors" else (k, int) for k in cols]

        if "soy" in cols and "year" in cols:
            dflag = 0
            ldtype[cols.index('year')] = ('year', int)
        elif "date" in cols and "time" in cols:
            dflag = 1
            ldtype[cols.index('date')] = ('date', '|U50')
            ldtype[cols.index('time')] = ('time', '|U50')

        if dflag < 0:
            estr = "missing time columns in [{:s}]".format(ocb_cols)
            ocbpy.logger.error(estr)
            return

        # Read the OCB data
        odata = np.rec.array(np.genfromtxt(self.filename, skip_header=hlines,
                                           dtype=ldtype))
        oname = list(odata.dtype.names)

        # Load the data into the OCBoundary object
        #
        # Start by getting the time and location in the desired format
        self.rec_ind = -1

        dt_list = list()
        if stime is None and etime is None:
            itime = np.arange(0, odata.shape[0], 1)
        else:
            itime = list()

        for i in range(odata.shape[0]):
            year = odata.year[i] if dflag == 0 else None
            soy = odata.soy[i] if dflag == 0 else None
            date = None if dflag == 0 else odata.date[i]
            tod = None if dflag == 0 else odata.time[i]

            dtime = convert_time(year=year, soy=soy, date=date, tod=tod,
                                 datetime_fmt=datetime_fmt)

            if stime is None and etime is None:
                dt_list.append(dtime)
            elif((stime is None or stime <= dtime) and
                 (etime is None or etime >= dtime)):
                dt_list.append(dtime)
                itime.append(i)

        if hasattr(odata, 'x') and hasattr(odata, 'y'):
            # Location is given by x-y coordinates where the origin lies
            # on the magnetic pole, the x-axis follows the dusk-dawn
            # meridian (positive towards dawn), and the y-axis follows the
            # midnight-noon meridian (positive towards noon)

            # Calculate the polar coordinates from the x-y coordinates
            odata.r_cent = np.sqrt(odata.x**2 + odata.y**2)
            oname.append("r_cent")

            # phi_cent is zero at magnetic midnight rather than dawn, so we
            # need to add 90.0 degrees from the arctangent.  Then convert all
            # degrees to their positive angles.
            odata.phi_cent = np.degrees(np.arctan2(odata.y, odata.x)) + 90.0
            odata.phi_cent[odata.phi_cent < 0.0] += 360.0
            oname.append("phi_cent")

        # Load the required information not contained in odata
        self.records = len(dt_list)
        self.dtime = np.array(dt_list)

        # Set the boundary function
        if self.rfunc is None:
            self._set_default_rfunc()
        elif isinstance(self.rfunc, types.FunctionType):
            self.rfunc = np.full(shape=self.records, fill_value=self.rfunc)
        elif hasattr(self.rfunc, "shape"):
            if self.rfunc.shape != self.dtime.shape:
                raise ValueError("Misshaped correction function array")
        else:
            raise ValueError("Unknown input type for correction function")

        # Set the boundary function keyword inputs
        if self.rfunc_kwargs is None:
            self.rfunc_kwargs = np.full(shape=self.records, fill_value={})
        elif isinstance(self.rfunc_kwargs, dict):
            self.rfunc_kwargs = np.full(shape=self.records,
                                        fill_value=self.rfunc_kwargs)
        elif hasattr(self.rfunc_kwargs, "shape"):
            if self.rfunc_kwargs.shape != self.dtime.shape:
                raise ValueError("Misshaped correction function keyword array")
        else:
            raise ValueError("Unknown input type for correction keywords")

        # Load the attributes saved in odata
        for nn in oname:
            setattr(self, nn, getattr(odata, nn)[itime])

        return

    def get_next_good_ocb_ind(self, min_sectors=7, rcent_dev=8.0, max_r=23.0,
                              min_r=10.0):
        """read in the next usuable OCB record from the data file.  Only uses
        the available parameters.

        Parameters
        ----------
        min_sectors : (int)
            Minimum number of MLT sectors required for good OCB. (default=7)
        rcent_dev : (float)
            Maximum number of degrees between the new centre and the AACGM pole
            (default=8.0)
        max_r : (float)
            Maximum radius for open-closed field line boundary in degrees.
            (default=23.0)
        min_r : (float)
            Minimum radius for open-closed field line boundary in degrees
            (default=10.0)

        Notes
        -----
        Updates self.rec_ind to the index of next good OCB record or a value
        greater than self.records if there aren't any more good records
        available after the starting point

        Comments
        --------
        IMAGE FUV checks:
        - more than 6 MLT boundary values have contributed to OCB circle
        - that the OCB 'pole' is with 8 degrees of the AACGM pole
        - that the OCB 'radius' is greater than 10 and less than 23 degrees
        AMPERE/DMSP-SSJ checks:
        - that the Figure of Merit is greater than or equal to the specified
          minimum

        """

        # Incriment forward from previous boundary
        self.rec_ind += 1

        while self.rec_ind < self.records:
            # Evaluate the current boundary for quality, using optional
            # parameters
            good = True
            if(hasattr(self, "num_sectors") and
               self.num_sectors[self.rec_ind] < min_sectors):
                good = False
            elif(hasattr(self, "fom")
                 and self.fom[self.rec_ind] < self.min_fom):
                good = False

            # Evaluate the current boundary for quality, using non-optional
            # parameters
            if(good and self.r_cent[self.rec_ind] <= rcent_dev
               and self.r[self.rec_ind] >= min_r
               and self.r[self.rec_ind] <= max_r):
                return

            # Cycle to next boundary
            self.rec_ind += 1

        return

    def normal_coord(self, lat, lt, coords='magnetic', height=350.0,
                     method='ALLOWTRACE'):
        """converts position(s) to normalised co-ordinates relative to the OCB

        Parameters
        ----------
        lat : (float or array-like)
            Input latitude (degrees), must be geographic, geodetic, or AACGMV2
        lt : (float or array-like)
            Input local time (hours), must be solar or AACGMV2 magnetic
        coords : (str)
            Input coordiate system.  Accepts 'magnetic', 'geocentric', or
            'geodetic' (default='magnetic')
        height : (float or array-like)
            Height (km) at which AACGMV2 coordinates will be calculated, if
            geographic coordinates are provided (default=350.0)
        method : (str)
            String denoting which type(s) of conversion to perform, if
            geographic coordinates are provided.  Expects either 'TRACE' or
            'ALLOWTRACE'.  See AACGMV2 for details.  (default='ALLOWTRACE')

        Returns
        -------
        ocb_lat : (float or array-like)
            Magnetic latitude relative to OCB (degrees)
        ocb_mlt : (float or array-like)
            Magnetic local time relative to OCB (hours)
        r_corr : (float or array-like)
            Radius correction to OCB (degrees)

        Comments
        --------
        Approximation - Conversion assumes a planar surface

        """

        # Cast input as arrays
        lat = np.asarray(lat)
        lt = np.asarray(lt)
        height = np.asarray(height)

        # Initialize output
        out_shape = max([lat.shape, lt.shape, height.shape])
        ocb_lat = np.full(shape=out_shape, fill_value=np.nan)
        ocb_mlt = np.full(shape=out_shape, fill_value=np.nan)
        r_corr = np.full(shape=out_shape, fill_value=np.nan)

        # Test the OCB record index
        if self.rec_ind < 0 or self.rec_ind >= self.records:
            return ocb_lat, ocb_mlt, r_corr

        # If needed, convert from geographic to magnetic coordinates
        if coords.lower().find('mag') < 0:
            # Convert from lt to longitude
            lon = slt2glon(lt, self.dtime[self.rec_ind])
            # If geocentric coordinates are specified, add this info to the
            # method flag
            if coords.lower() == 'geocentric':
                method = "|".join([method, coords.upper()])
            aacgm_lat, _, aacgm_mlt = aacgmv2.get_aacgm_coord_arr(
                lat, lon, height, self.dtime[self.rec_ind], method)
        else:
            aacgm_lat = lat
            aacgm_mlt = lt

        # Ensure the correct hemisphere is loaded for this data
        if np.any(np.sign(aacgm_lat) != self.hemisphere):
            if np.all(np.sign(aacgm_lat) != self.hemisphere):
                return ocb_lat, ocb_mlt, r_corr
            aacgm_lat[np.sign(aacgm_lat) != self.hemisphere] = np.nan

        # Calculate the center of the OCB
        phi_cent_rad = np.radians(self.phi_cent[self.rec_ind])
        xc = self.r_cent[self.rec_ind] * np.cos(phi_cent_rad)
        yc = self.r_cent[self.rec_ind] * np.sin(phi_cent_rad)

        # Calculate the desired point location relative to the AACGM pole
        scalep = 90.0 - self.hemisphere * aacgm_lat
        xp = scalep * np.cos(np.radians(aacgm_mlt * 15.0))
        yp = scalep * np.sin(np.radians(aacgm_mlt * 15.0))

        # Get the distance between the OCB pole and the point location.  This
        # distance is then scaled by r, the OCB radius.  For non-circular
        # boundaries, r is a function of MLT
        r_corr = self.rfunc[self.rec_ind](aacgm_mlt,
                                          **self.rfunc_kwargs[self.rec_ind])
        scalen = (90.0 - abs(self.boundary_lat)) / (self.r[self.rec_ind]
                                                    + r_corr)
        xn = (xp - xc) * scalen
        yn = (yp - yc) * scalen

        ocb_lat = self.hemisphere * (90.0 - np.sqrt(xn**2 + yn**2))
        ocb_mlt = deg2hr(np.degrees(np.arctan2(yn, xn)))
        ocb_mlt = fix_range(ocb_mlt, 0.0, 24.0)

        return ocb_lat, ocb_mlt, r_corr

    def revert_coord(self, ocb_lat, ocb_mlt, r_corr=0.0, coords='magnetic',
                     height=350.0, method='ALLOWTRACE'):
        """Converts the position of a measurement in normalised co-ordinates
        relative to the OCB into AACGM co-ordinates

        Parameters
        ----------
        ocb_lat : (float or array-like)
            Input OCB latitude in degrees
        ocb_mlt : (float or array-like)
            Input OCB local time in hours
        r_corr : (float or array-like)
            Input OCB radial correction in degrees, may be a function of
            AACGM MLT (default=0.0)
        coords : (str)
            Output coordiate system.  Accepts 'magnetic', 'geocentric', or
            'geodetic' (default='magnetic')
        height : (float or array-like)
            Geocentric height above sea level (km) at which AACGMV2 coordinates
            will be calculated, if geographic coordinates are desired
            (default=350.0)
        method : (str)
            String denoting which type(s) of conversion to perform, if
            geographic coordinates are provided.  Expects either 'TRACE' or
            'ALLOWTRACE'.  See AACGMV2 for details.  (default='ALLOWTRACE')

        Returns
        -------
        lat : (float or array-like)
            latitude (degrees)
        lt : (float or array-like)
            local time (hours)

        Comments
        --------
        Approximation - Conversion assumes a planar surface

        """

        # Cast input as arrays
        ocb_lat = np.asarray(ocb_lat)
        ocb_mlt = np.asarray(ocb_mlt)
        r_corr = np.asarray(r_corr)
        height = np.asarray(height)

        # Initialize output
        out_shape = max([ocb_lat.shape, ocb_mlt.shape, r_corr.shape,
                         height.shape])
        lat = np.full(shape=out_shape, fill_value=np.nan)
        lt = np.full(shape=out_shape, fill_value=np.nan)

        # Test the OCB index and hemisphere
        if self.rec_ind < 0 or self.rec_ind >= self.records:
            return lat, lt

        if np.all(np.sign(ocb_lat) != self.hemisphere):
            return lat, lt

        # Perform the coordinate transformation
        phi_cent_rad = np.radians(self.phi_cent[self.rec_ind])
        xc = self.r_cent[self.rec_ind] * np.cos(phi_cent_rad)
        yc = self.r_cent[self.rec_ind] * np.sin(phi_cent_rad)

        rn = 90.0 - self.hemisphere * ocb_lat

        thetan = ocb_mlt * np.pi / 12.0
        xn = rn * np.cos(thetan)
        yn = rn * np.sin(thetan)

        scale_ocb = (self.r[self.rec_ind]
                     + r_corr) / (90.0 - self.hemisphere * self.boundary_lat)
        xp = xn * scale_ocb + xc
        yp = yn * scale_ocb + yc

        aacgm_lat = self.hemisphere * (90.0 - np.sqrt(xp**2 + yp**2))
        aacgm_mlt = deg2hr(np.degrees(np.arctan2(yp, xp)))
        aacgm_mlt = fix_range(aacgm_mlt, 0.0, 24.0)

        # If needed, convert from magnetic to geographic coordinates
        if coords.lower().find('mag') < 0:
            # Convert from mlt to longitude
            lon = aacgmv2.convert_mlt(aacgm_mlt, self.dtime[self.rec_ind],
                                      m2a=True)
            # If geocentric coordinates are specified, add this info to the
            # method flag
            if coords.lower() == 'geocentric':
                method = "|".join([method, coords.upper()])
            method = "|".join([method, "A2G"])
            lat, lon, _ = aacgmv2.convert_latlon_arr(aacgm_lat, lon, height,
                                                     self.dtime[self.rec_ind],
                                                     method)

            # Convert from longitude to solar local time
            lt = glon2slt(lon, self.dtime[self.rec_ind])
        else:
            lat = aacgm_lat
            lt = aacgm_mlt

        return lat, lt

    def get_aacgm_boundary_lat(self, aacgm_lon, rec_ind=None,
                               overwrite=False):
        """Calculate the OCB latitude in AACGM coordinates at specified
        longitudes

        Parameters
        ----------
        aacgm_lon : (int, float, or array-like)
            AACGM longitude location(s) (in degrees) for which the OCB latitude
            will be calculated.
        rec_ind : (int, array-like, or NoneType)
            Record index for which the OCB AACGM latitude will be calculated,
            or None to calculate all boundary locations (default=None).
        overwrite : (boolean)
            Overwrite previous boundary locations if this time already has
            calculated boundary latitudes for a different set of input
            longitudes (default=False).

        Returns
        -------
        Updates OCBoundary object with list attributes.  If no boundary value
        is calculated at a certain time, the list is padded with None.  If
        a boundary latitude cannot be calculated at that time and longitude,
        that time and longitude is filled with NaN.

        'aacgm_boundary_lat' contains the AACGM latitude location(s) of the OCB
        (in degrees) for each requested time.

        'aacgm_boundary_lon' holds the aacgm_lon input for each requested
        time.  The requested longitude may differ from time to time, to allow
        easy comparison with satellite passes.

        """

        # Ensure the boundary longitudes span from 0-360 degrees
        aacgm_lon = np.asarray(aacgm_lon)
        aacgm_lon[aacgm_lon < 0.0] += 360.0
        aacgm_lon[aacgm_lon >= 360.0] -= 360.0

        if not hasattr(self, 'aacgm_boundary_lon'):
            self.aacgm_boundary_lon = [None for i in range(self.records)]

        if not hasattr(self, 'aacgm_boundary_lat'):
            self.aacgm_boundary_lat = [None for i in range(self.records)]

        # Get the indices to calculate the boundary latitudes
        if rec_ind is None:
            # Create array of all indices
            rinds = np.arange(0, self.records, 1)
        else:
            # Create array of indices as integers
            rinds = np.asarray(rec_ind).astype('int')

            # Ensure single values are stored as an interable object
            if len(rinds.shape) == 0:
                rinds = rinds.reshape(1,)

        # Calculate the boundary location for each requested time
        for i in rinds:
            # If data exists here and the overwrite option is off, skip
            if self.aacgm_boundary_lat[i] is None or overwrite:
                # Calculate the difference between the output longitude and the
                # longitude of the centre of the polar cap
                del_lon = np.radians(aacgm_lon - self.phi_cent[i])

                # Calculate the radius of the OCB in degrees
                r_corr = self.rfunc[i](deg2hr(aacgm_lon),
                                       **self.rfunc_kwargs[i])
                scale_r = self.r[i] + r_corr
                rad = self.r_cent[i] * np.cos(del_lon) \
                    + np.sqrt(scale_r**2 - (self.r_cent[i]
                                            * np.sin(del_lon))**2)

                # If the radius is negative, set to NaN
                if len(rad.shape) > 0:
                    rad[rad < 0.0] = np.nan
                else:
                    rad = np.nan if rad < 0.0 else float(rad)

                # Calculate the latitude of the OCB in AACGM coordinates
                self.aacgm_boundary_lat[i] = self.hemisphere * (90.0 - rad)

                # Save the longitude at this time
                self.aacgm_boundary_lon[i] = aacgm_lon
            else:
                estr = "".join(["unable to update AACGM boundary latitude at ",
                                "{:}, overwrite ".format(self.dtime[i]),
                                "blocked"])
                ocbpy.logger.warning(estr)

        return

    def _set_default_rfunc(self):
        """Set the default instrument OCB boundary function

        Notes
        -----
        Assign a function for each time in case we have a data set with a
        correction that changes with UT

        """

        if self.instrument in ["image", "dmsp-ssj"]:
            self.rfunc = np.full(shape=self.records,
                                 fill_value=ocbcor.circular)
        elif self.instrument == "ampere":
            self.rfunc = np.full(shape=self.records,
                                 fill_value=ocbcor.elliptical)
        else:
            raise ValueError("unknown instrument")

        return


def retrieve_all_good_indices(ocb):
    """Retrieve all good indices from the ocb structure

    Parameters
    ----------
    ocb : (OCBoundary)
        Class containing the open-close field line boundary data

    Returns
    -------
    good_ind : (list)
        List of indices containing good OCBs

    """

    # Save the current record index
    icurrent = ocb.rec_ind

    # Set the record index to allow us to cycle through the entire data set
    ocb.rec_ind = -1

    # Initialize the output data
    good_ind = list()

    # Cycle through all records
    while ocb.rec_ind < ocb.records:
        ocb.get_next_good_ocb_ind()
        if ocb.rec_ind < ocb.records:
            good_ind.append(int(ocb.rec_ind))

    # Reset the record index
    ocb.rec_ind = icurrent

    # Return the good indices
    return good_ind


def match_data_ocb(ocb, dat_dtime, idat=0, max_tol=600, min_sectors=7,
                   rcent_dev=8.0, max_r=23.0, min_r=10.0):
    """Matches data records with OCB records, locating the closest values
    within a specified tolerance

    Parameters
    ----------
    ocb : (OCBoundary)
        Class containing the open-close field line boundary data
    dat_dtime : (list or numpy array of datetime objects)
        Times where data exists
    idat : (int)
        Current data index (default=0)
    max_tol : (int)
        maximum seconds between OCB and data record in sec (default=600)
    min_sectors : (int)
        Minimum number of MLT sectors required for good OCB. (default=7)
    rcent_dev : (float)
        Maximum number of degrees between the new centre and the AACGM pole
        (default=8.0)
    max_r : (float)
        Maximum radius for open-closed field line boundary in degrees
        (default=23.0)
    min_r : (float)
        Minimum radius for open-closed field line boundary in degrees
        (default=10.0)

    Returns
    -------
    idat : (int or NoneType)
        Data index for match value, None if all of the data have been searched

    Notes
    -----
    Updates OCBoundary.rec_ind for matched value. None if all of the
    boundaries have been searched.

    """

    dat_records = len(dat_dtime)

    # Ensure that the indices are good
    if idat >= dat_records:
        return idat
    if ocb.rec_ind >= ocb.records:
        return idat

    # Get the first reliable circle boundary estimate if none was provided
    if ocb.rec_ind < 0:
        ocb.get_next_good_ocb_ind(min_sectors=min_sectors, rcent_dev=rcent_dev,
                                  max_r=max_r, min_r=min_r)
        if ocb.rec_ind >= ocb.records:
            estr = "".join(["unable to find a good OCB record in ",
                            ocb.filename])
            ocbpy.logger.error(estr)
            return idat
        else:
            estr = "".join(["found first good OCB record at ",
                            "{:}".format(ocb.dtime[ocb.rec_ind])])
            ocbpy.logger.info(estr)

        # Cycle past data occuring before the specified OC boundary point
        first_ocb = ocb.dtime[ocb.rec_ind] - dt.timedelta(seconds=max_tol)
        while dat_dtime[idat] < first_ocb:
            idat += 1

            if idat >= dat_records:
                ocbpy.logger.error("".join(["no input data close enough ",
                                            "to first record"]))
                return None

    # If the times match, return
    if ocb.dtime[ocb.rec_ind] == dat_dtime[idat]:
        return idat

    # If the times don't match, cycle through both datasets until they do
    while idat < dat_records and ocb.rec_ind < ocb.records:
        # Increase the OCB index until one lies within the desired boundary
        sdiff = (ocb.dtime[ocb.rec_ind] - dat_dtime[idat]).total_seconds()

        if sdiff < -max_tol:
            # Cycle to the next OCB value since the lowest vorticity value
            # is in the future
            ocb.get_next_good_ocb_ind(min_sectors=min_sectors,
                                      rcent_dev=rcent_dev, max_r=max_r,
                                      min_r=min_r)
        elif sdiff > max_tol:
            # Cycle to the next value if no OCB values were close enough
            estr = "".join(["no OCB data available within ",
                            "[{:d} s] of input measurement at".format(max_tol),
                            " [{:}]".format(dat_dtime[idat])])
            ocbpy.logger.info(estr)
            idat += 1
        else:
            # Make sure this is the OCB value closest to the input record
            last_sdiff = sdiff
            last_iocb = ocb.rec_ind
            ocb.get_next_good_ocb_ind(min_sectors=min_sectors,
                                      rcent_dev=rcent_dev, max_r=max_r,
                                      min_r=min_r)

            if ocb.rec_ind < ocb.records:
                sdiff = (ocb.dtime[ocb.rec_ind] -
                         dat_dtime[idat]).total_seconds()

                while abs(sdiff) < abs(last_sdiff):
                    last_sdiff = sdiff
                    last_iocb = ocb.rec_ind
                    ocb.get_next_good_ocb_ind(min_sectors=min_sectors,
                                              rcent_dev=rcent_dev, max_r=max_r,
                                              min_r=min_r)
                    if ocb.rec_ind < ocb.records:
                        sdiff = (ocb.dtime[ocb.rec_ind] -
                                 dat_dtime[idat]).total_seconds()

            sdiff = last_sdiff
            ocb.rec_ind = last_iocb

            # Return the requested indices
            return idat

    # Return from the last loop
    if idat == 0 and abs(sdiff) > max_tol:
        estr = "".join(["no OCB data available within ",
                        "[{:d} s] of first measurement ".format(max_tol),
                        "[{:}]".format(dat_dtime[idat])])
        ocbpy.logger.info(estr)

    return idat
