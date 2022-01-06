#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Hold, manipulate, and load the OCB and EAB data.

References
----------
.. [2] Angeline Burrell, Christer van der Meeren, & Karl M. Laundal. (2020).
   aburrell/aacgmv2 (All Versions). Zenodo. doi:10.5281/zenodo.1212694.

.. [3] Shepherd, S. G. (2014), Altitude‐adjusted corrected geomagnetic
   coordinates: Definition and functional approximations, Journal of
   Geophysical Research: Space Physics, 119, 7501–7521,
   doi:10.1002/2014JA020264.

"""

import datetime as dt
import numpy as np
import types

import aacgmv2

from ocbpy import logger
import ocbpy.ocb_correction as ocbcor
from ocbpy import ocb_time
from ocbpy.boundaries.files import get_default_file
from ocbpy.instruments import test_file


class OCBoundary(object):
    """Object containing open-closed field-line boundary (OCB) data.

    Parameters
    ----------
    filename : str or NoneType
        File containing the required open-closed boundary data sorted by time.
        If NoneType, no file is loaded.  If 'default',
        `ocbpy.boundaries.files.get_default_file` is called. (default='default')
    instrument : str
        Instrument providing the OCBoundaries.  Requires 'image', 'ampere', or
        'dmsp-ssj' if a file is provided.  If using filename='default', also
        accepts 'amp', 'si12', 'si13', 'wic', and ''.  (default='')
    hemisphere : int
        Integer (+/- 1) denoting northern/southern hemisphere (default=1)
    boundary_lat : float
        Typical OCBoundary latitude in AACGM coordinates.  Hemisphere will
        give this boundary the desired sign.  (default=74.0)
    stime : dt.datetime or NoneType
        First time to load data or beginning of file.  If specifying time, be
        sure to start before the time of the data to allow the best match within
        the allowable time tolerance to be found. (default=None)
    etime : dt.datetime or NoneType
        Last time to load data or ending of file.  If specifying time, be sure
        to end after the last data point you wish to match to, to ensure the
        best match within the allowable time tolerance is made. (default=None)
    rfunc : numpy.ndarray, function, or NoneType
        OCB radius correction function. If None, will use the instrument
        default. Function must have AACGM MLT (in hours) as argument input.
        To allow the boundary shape to change with univeral time, each temporal
        instance may have a different function (array input). If a single
        function is provided, will recast as an array that specifies this
        function for all times. (default=None)
    rfunc_kwargs : numpy.ndarray, dict, or NoneType
        Optional keyword arguements for `rfunc`. If None is specified,
        uses function defaults.  If dict is specified, recasts as an array
        of this dict for all times.  Array must be an array of dicts.
        (default=None)

    Attributes
    ----------
    records : int
        Number of OCB records (default=0)
    rec_ind : int
        Current OCB record index (default=0; initialised=-1)
    dtime : numpy.ndarray or NoneType
        Numpy array of OCB datetimes (default=None)
    phi_cent : numpy.ndarray or NoneType
        Numpy array of floats that give the angle from AACGM midnight
        of the OCB pole in degrees (default=None)
    r_cent : numpy.ndarray or NoneType
        Numpy array of floats that give the AACGM co-latitude of the OCB
        pole in degrees (default=None)
    r : numpy.ndarray or NoneType
        Numpy array of floats that give the radius of the OCBoundary
        in degrees (default=None)
    min_fom : float
        Minimum acceptable figure of merit for data (default=0)
    x, y, j_mag, etc. : numpy.ndarray or NoneType
        Numpy array of floats that hold the remaining values held in `filename`

    Raises
    ------
    ValueError
        Incorrect or incompatible input

    """

    def __init__(self, filename="default", instrument='', hemisphere=1,
                 boundary_lat=74.0, stime=None, etime=None, rfunc=None,
                 rfunc_kwargs=None):
        # Test the instrument input
        if not hasattr(instrument, "lower"):
            logger.error("OCB instrument must be a string [{:}]".format(
                instrument))
            self.filename = None
            self.instrument = None
        else:
            self.instrument = instrument.lower()

            # If a filename wanted and not provided, get one
            if filename is None:
                self.filename = None
            elif not hasattr(filename, "lower"):
                logger.warning("filename is not a string [{:}]".format(
                    filename))
                self.filename = None
            elif filename.lower() == "default":
                self.filename, self.instrument = get_default_file(
                    stime, etime, hemisphere, self.instrument)
            else:
                self.filename = filename

            # If a filename is available, make sure it is good
            if self.filename is not None:
                if not test_file(self.filename):
                    # If the filename is bad, return an uninitialized object
                    logger.warning("cannot open OCB file [{:s}]".format(
                        self.filename))
                    self.filename = None

        # Test the hemisphere input
        if hemisphere not in [1, -1]:
            raise ValueError("hemisphere must be 1 (north) or -1 (south)")

        # Set the default attribute values
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
        """Provide an evaluatable representation of the OCBoundary object."""
        class_name = repr(self.__class__).split("'")[1]

        # Get the start and end times
        stime = None if self.dtime is None else self.dtime[0]
        etime = None if self.dtime is None else self.dtime[-1]

        # Format the function representations
        if self.rfunc is None:
            repr_rfunc = repr(self.rfunc)
        else:
            rfuncs = [".".join([ff.__module__, ff.__name__])
                      for ff in self.rfunc]

            if len(set(rfuncs)) == 1:
                repr_rfunc = rfuncs[0]
            else:
                repr_rfunc = 'numpy.array([{:s}], dtype=object)'.format(
                    ', '.join(rfuncs))

        # Format the function kwarg representations
        if self.rfunc_kwargs is None:
            repr_rfunc_kwargs = repr(self.rfunc_kwargs)
        else:
            rfuncs_kwargs = [repr(rkwarg) for rkwarg in self.rfunc_kwargs]

            if len(set(rfuncs_kwargs)) == 1:
                repr_rfunc_kwargs = rfuncs_kwargs[0]
            else:
                repr_rfunc_kwargs = 'numpy.array([{:s}], dtype=object)'.format(
                    ', '.join(rfuncs_kwargs))

        # Format the output
        out = "".join([class_name, "(filename=", repr(self.filename),
                       ", instrument=", repr(self.instrument),
                       ", hemisphere={:d}, ".format(self.hemisphere),
                       "boundary_lat={:f}, stime=".format(self.boundary_lat),
                       repr(stime), ", etime=", repr(etime), ", rfunc=",
                       repr_rfunc, ", rfunc_kwargs=", repr_rfunc_kwargs, ")"])
        return out

    def __str__(self):
        """Provide readable representation of the OCBoundary object."""

        class_name = repr(self.__class__).split("'")[1]

        if self.filename is None:
            out = "No {:s} file specified\n".format(class_name)
        else:
            out = "Boundary file: {:s}\n".format(self.filename)
            out = "{:s}Source instrument: ".format(out)
            out = "{:s}{:s}\n".format(out, self.instrument.upper())
            out = "{:s}Boundary reference latitude: ".format(out)
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
                    fnames = list(set([".".join([ff.__module__, ff.__name__])
                                       for ff in self.rfunc]))

                    for ff in fnames:
                        kw = list(set([self.rfunc_kwargs[i].__str__()
                                       for i, rf in enumerate(self.rfunc)
                                       if rf.__name__ == ff.split(".")[-1]]))

                        for kk in kw:
                            out = "{:s}{:s}(**{:s})\n".format(out, ff, kk)

        return out

    def inst_defaults(self):
        """Get the instrument-specific OCB file loading information.

        Returns
        -------
        hlines : int
            Number of header lines
        ocb_cols : str
            String containing the names for each data column
        datetime_fmt : str
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
        """Load the data from the specified Open-Closed Boundary file.

        Parameters
        ----------
        ocb_cols : str
            String specifying format of OCB file.  All but the first two
            columns must be included in the string, additional data values will
            be ignored.  If 'year soy' aren't used, expects
            'date time' in 'YYYY-MM-DD HH:MM:SS' format.
            (default='year soy num_sectors phi_cent r_cent r a r_err')
        hlines : int
            Number of header lines preceeding data in the OCB file (default=0)
        datetime_fmt : str
            A string used to read in 'date time' data.  Not used if 'year soy'
            is specified. (default='')
        stime : dt.datetime or NoneType
            Time to start loading data or None to start at beginning of file.
            (default=None)
        etime : datetime or NoneType
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
            logger.error("missing time columns in [{:s}]".format(ocb_cols))
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

            dtime = ocb_time.convert_time(year=year, soy=soy, date=date,
                                          tod=tod, datetime_fmt=datetime_fmt)

            if stime is None and etime is None:
                dt_list.append(dtime)
            elif((stime is None or stime <= dtime)
                 and (etime is None or etime >= dtime)):
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
        """Read in the next usuable OCB record from the data file.

        Parameters
        ----------
        min_sectors : int
            Minimum number of MLT sectors required for good OCB. (default=7)
        rcent_dev : float
            Maximum number of degrees between the new centre and the AACGM pole
            (default=8.0)
        max_r : float
            Maximum radius for open-closed field line boundary in degrees.
            (default=23.0)
        min_r : float
            Minimum radius for open-closed field line boundary in degrees
            (default=10.0)

        Notes
        -----
        Updates self.rec_ind to the index of next good OCB record or a value
        greater than self.records if there aren't any more good records
        available after the starting point

        IMAGE FUV checks that:
        - more than 6 MLT boundary values have contributed to OCB circle
        - the OCB 'pole' is with 8 degrees of the AACGM pole
        - the OCB 'radius' is greater than 10 and less than 23 degrees
        AMPERE/DMSP-SSJ checks that:
        - the Figure of Merit is greater than or equal to the specified minimum

        """

        # Incriment forward from previous boundary
        self.rec_ind += 1

        while self.rec_ind < self.records:
            # Evaluate the current boundary for quality, using optional
            # parameters
            good = True
            if(hasattr(self, "num_sectors")
               and self.num_sectors[self.rec_ind] < min_sectors):
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
        """Converts position(s) to normalised co-ordinates relative to the OCB.

        Parameters
        ----------
        lat : float or array-like
            Input latitude (degrees), must be geographic, geodetic, or AACGMV2
        lt : float or array-like
            Input local time (hours), must be solar or AACGMV2 magnetic
        coords : str
            Input coordiate system.  Accepts 'magnetic', 'geocentric', or
            'geodetic' (default='magnetic')
        height : float or array-like
            Height (km) at which AACGMV2 coordinates will be calculated, if
            geographic coordinates are provided (default=350.0)
        method : str
            String denoting which type(s) of conversion to perform, if
            geographic coordinates are provided. Expects either 'TRACE' or
            'ALLOWTRACE'. See AACGMV2 for details [2]_. (default='ALLOWTRACE')

        Returns
        -------
        ocb_lat : float or array-like
            Magnetic latitude relative to OCB (degrees)
        ocb_mlt : float or array-like
            Magnetic local time relative to OCB (hours)
        r_corr : float or array-like
            Radius correction to OCB (degrees)

        Notes
        -----
        Approximation - Conversion assumes a planar surface

        See Also
        --------
        aacgmv2

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
            lon = ocb_time.slt2glon(lt, self.dtime[self.rec_ind])
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
        ocb_mlt = ocb_time.deg2hr(np.degrees(np.arctan2(yn, xn)))
        ocb_mlt = ocb_time.fix_range(ocb_mlt, 0.0, 24.0)

        return ocb_lat, ocb_mlt, r_corr

    def revert_coord(self, ocb_lat, ocb_mlt, r_corr=0.0, coords='magnetic',
                     height=350.0, method='ALLOWTRACE'):
        """Convert the position of a measurement in OCB into AACGM co-ordinates.

        Parameters
        ----------
        ocb_lat : float or array-like
            Input OCB latitude in degrees
        ocb_mlt : float or array-like
            Input OCB local time in hours
        r_corr : float or array-like
            Input OCB radial correction in degrees, may be a function of
            AACGM MLT (default=0.0)
        coords : str
            Output coordiate system.  Accepts 'magnetic', 'geocentric', or
            'geodetic' (default='magnetic')
        height : float or array-like
            Geocentric height above sea level (km) at which AACGMV2 coordinates
            will be calculated, if geographic coordinates are desired
            (default=350.0)
        method : str
            String denoting which type(s) of conversion to perform, if
            geographic coordinates are provided.  Expects either 'TRACE' or
            'ALLOWTRACE'.  See AACGMV2 for details [2]_.  (default='ALLOWTRACE')

        Returns
        -------
        lat : float or array-like
            latitude (degrees)
        lt : float or array-like
            local time (hours)

        Notes
        -----
        Approximation - Conversion assumes a planar surface

        See Also
        --------
        aacgmv2

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
        aacgm_mlt = ocb_time.deg2hr(np.degrees(np.arctan2(yp, xp)))
        aacgm_mlt = ocb_time.fix_range(aacgm_mlt, 0.0, 24.0)

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
            lt = ocb_time.glon2slt(lon, self.dtime[self.rec_ind])
        else:
            lat = aacgm_lat
            lt = aacgm_mlt

        return lat, lt

    def get_aacgm_boundary_lat(self, aacgm_mlt, rec_ind=None,
                               overwrite=False, set_lon=True):
        """Get the OCB latitude in AACGM coordinates at specified longitudes.

        Parameters
        ----------
        aacgm_mlt : int, float, or array-like
            AACGM longitude location(s) (in degrees) for which the OCB latitude
            will be calculated.
        rec_ind : int, array-like, or NoneType
            Record index for which the OCB AACGM latitude will be calculated,
            or None to calculate all boundary locations (default=None).
        overwrite : bool
            Overwrite previous boundary locations if this time already has
            calculated boundary latitudes for a different set of input
            longitudes (default=False).
        set_lon : bool
            Calculate the AACGM longitude of the OCB alongside the MLT
            (default=True).

        Notes
        -----
        Updates OCBoundary object with list attributes.  If no boundary value
        is calculated at a certain time, the list is padded with None.  If
        a boundary latitude cannot be calculated at that time and longitude,
        that time and longitude is filled with NaN.

        `aacgm_boundary_lat` contains the AACGM latitude location(s) of the OCB
        (in degrees) for each requested time [3]_.

        `aacgm_boundary_mlt` holds the aacgm_mlt input for each requested
        time.  The requested MLT may differ from time to time, to allow
        easy comparison with satellite passes [3]_.

        `aacgm_boundary_lon` holds the aacgm_lon input for each requested
        time.  This is calculated from `aacgm_boundary_mlt` by default [3]_.

        """

        # Ensure the boundary longitudes span from 0-360 degrees
        aacgm_mlt = np.asarray(aacgm_mlt)
        aacgm_mlt[aacgm_mlt < 0.0] += 24.0
        aacgm_mlt[aacgm_mlt >= 24.0] -= 24.0

        if not hasattr(self, 'aacgm_boundary_mlt'):
            self.aacgm_boundary_mlt = [None for i in range(self.records)]

        if not hasattr(self, 'aacgm_boundary_lat'):
            self.aacgm_boundary_lat = [None for i in range(self.records)]

        if set_lon and not hasattr(self, 'aacgm_boundary_lon'):
            self.aacgm_boundary_lon = [None for i in range(self.records)]

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
                # Calculate the difference between the output MLT and the
                # MLT of the centre of the polar cap, which is give in degrees
                del_mlt = ocb_time.hr2rad(aacgm_mlt
                                          - ocb_time.deg2hr(self.phi_cent[i]))

                # Calculate the radius of the OCB in degrees
                r_corr = self.rfunc[i](aacgm_mlt, **self.rfunc_kwargs[i])
                scale_r = self.r[i] + r_corr
                rad = self.r_cent[i] * np.cos(del_mlt) \
                    + np.sqrt(scale_r**2 - (self.r_cent[i]
                                            * np.sin(del_mlt))**2)

                # If the radius is negative, set to NaN
                if len(rad.shape) > 0:
                    rad[rad < 0.0] = np.nan
                else:
                    rad = np.nan if rad < 0.0 else float(rad)

                # Calculate the latitude of the OCB in AACGM coordinates
                self.aacgm_boundary_lat[i] = self.hemisphere * (90.0 - rad)

                # Save the MLT at this time
                self.aacgm_boundary_mlt[i] = aacgm_mlt

                # Set the longitude at this time
                if set_lon:
                    self.aacgm_boundary_lon[i] = np.asarray(
                        aacgmv2.convert_mlt(aacgm_mlt, self.dtime[i],
                                            m2a=True))
            else:
                logger.warning("".join(["unable to update AACGM boundary ",
                                        "latitude at {:}".format(self.dtime[i]),
                                        ", overwrite blocked"]))

        return

    def _set_default_rfunc(self):
        """Set the default instrument OCB boundary function.

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


class EABoundary(OCBoundary):
    """Object containing equatorward auroral boundary (EAB) data.

    Parameters
    ----------
    filename : str or NoneType
        File containing the required equatorward auroral boundary data sorted by
        time.  If NoneType, no file is loaded.  If 'default',
        `ocbpy.boundaries.files.get_default_file` is called. (default='default')
    instrument : str
        Instrument providing the EABoundaries.  Requires 'image' or 'dmsp-ssj'
        if a file is provided.  If using filename='default', also accepts
        'si12', 'si13', 'wic', and ''.  (default='')
    hemisphere : int
        Integer (+/- 1) denoting northern/southern hemisphere (default=1)
    boundary_lat : float
        Typical EABoundary latitude in AACGM coordinates.  Hemisphere will
        give this boundary the desired sign.  (default=64.0)
    stime : dt.datetime or NoneType
        First time to load data or beginning of file.  If specifying time, be
        sure to start before the time of the data to allow the best match within
        the allowable time tolerance to be found. (default=None)
    etime : dt.datetime or NoneType
        Last time to load data or ending of file.  If specifying time, be sure
        to end after the last data point you wish to match to, to ensure the
        best match within the allowable time tolerance is made. (default=None)
    rfunc : numpy.ndarray, function, or NoneType
        EAB radius correction function. If None, will use the instrument
        default. Function must have AACGM MLT (in hours) as argument input.
        To allow the boundary shape to change with univeral time, each temporal
        instance may have a different function (array input). If a single
        function is provided, will recast as an array that specifies this
        function for all times. (default=None)
    rfunc_kwargs : numpy.ndarray, dict, or NoneType
        Optional keyword arguements for `rfunc`. If None is specified,
        uses function defaults.  If dict is specified, recasts as an array
        of this dict for all times.  Array must be an array of dicts.
        (default=None)

    Attributes
    ----------
    records : int
        Number of EAB records (default=0)
    rec_ind : int
        Current EAB record index (default=0; initialised=-1)
    dtime : numpy.ndarray or NoneType
        Numpy array of EAB datetimes (default=None)
    phi_cent : numpy.ndarray or NoneType
        Numpy array of floats that give the angle from AACGM midnight
        of the EAB pole in degrees (default=None)
    r_cent : numpy.ndarray or NoneType
        Numpy array of floats that give the AACGM co-latitude of the EAB
        pole in degrees (default=None)
    r : numpy.ndarray or NoneType
        Numpy array of floats that give the radius of the EABoundary
        in degrees (default=None)
    min_fom : float
        Minimum acceptable figure of merit for data (default=0)
    x, y, j_mag, etc. : numpy.ndarray or NoneType
        Numpy array of floats that hold the remaining values held in `filename`

    Raises
    ------
    ValueError
        Incorrect or incompatible input

    """

    def __init__(self, filename="default", instrument='', hemisphere=1,
                 boundary_lat=64.0, stime=None, etime=None, rfunc=None,
                 rfunc_kwargs=None):

        # Process the defaults that differ for the EAB
        if rfunc is None:
            # Set to a function that will not alter the data
            rfunc = ocbcor.circular

        if hasattr(filename, "lower") and filename.lower() == "default":
            filename, instrument = get_default_file(stime, etime, hemisphere,
                                                    instrument, bound='eab')

        # Initialize the class
        OCBoundary.__init__(self, filename=filename, instrument=instrument,
                            hemisphere=hemisphere, boundary_lat=boundary_lat,
                            stime=stime, etime=etime, rfunc=rfunc,
                            rfunc_kwargs=rfunc_kwargs)

        return


class DualBoundary(object):
    """Object containing EAB and OCB data for dual-boundary coordinates.

    Parameters
    ----------
    eab_filename : str or NoneType
        File containing the required equatorward auroral boundary data sorted by
        time.  If NoneType, no file is loaded.  If 'default',
        `ocbpy.boundaries.files.get_default_file` is called. (default='default')
    ocb_filename : str or NoneType
        File containing the required open-closed field line boundary data sorted
        by time.  If NoneType, no file is loaded.  If 'default',
        `ocbpy.boundaries.files.get_default_file` is called. (default='default')
    eab_instrument : str
        Instrument providing the EABoundaries.  Requires 'image' or 'dmsp-ssj'
        if a file is provided.  If using filename='default', also accepts
        'si12', 'si13', 'wic', and ''.  (default='')
    ocb_instrument : str
        Instrument providing the OCBoundaries.  Requires 'image', 'ampere, or
        'dmsp-ssj' if a file is provided.  If using filename='default', also
        accepts 'si12', 'si13', 'wic', and ''.  (default='')
    hemisphere : int
        Integer (+/- 1) denoting northern/southern hemisphere (default=1)
    eab_lat : float
        Typical EABoundary latitude in AACGM coordinates.  Hemisphere will
        give this boundary the desired sign.  (default=64.0)
    ocb_lat : float
        Typical OCBoundary latitude in AACGM coordinates.  Hemisphere will
        give this boundary the desired sign.  (default=74.0)
    stime : dt.datetime or NoneType
        First time to load data or beginning of file.  If specifying time, be
        sure to start before the time of the data to allow the best match within
        the allowable time tolerance to be found. (default=None)
    etime : dt.datetime or NoneType
        Last time to load data or ending of file.  If specifying time, be sure
        to end after the last data point you wish to match to, to ensure the
        best match within the allowable time tolerance is made. (default=None)
    eab_rfunc : numpy.ndarray, function, or NoneType
        EAB radius correction function. If None, will use the instrument
        default. Function must have AACGM MLT (in hours) as argument input.
        To allow the boundary shape to change with univeral time, each temporal
        instance may have a different function (array input). If a single
        function is provided, will recast as an array that specifies this
        function for all times. (default=None)
    eab_rfunc_kwargs : numpy.ndarray, dict, or NoneType
        Optional keyword arguements for `eab_rfunc`. If None is specified,
        uses function defaults.  If dict is specified, recasts as an array
        of this dict for all times.  Array must be an array of dicts.
        (default=None)
    ocb_rfunc : numpy.ndarray, function, or NoneType
        OCB radius correction function. If None, will use the instrument
        default. Function must have AACGM MLT (in hours) as argument input.
        To allow the boundary shape to change with univeral time, each temporal
        instance may have a different function (array input). If a single
        function is provided, will recast as an array that specifies this
        function for all times. (default=None)
    ocb_rfunc_kwargs : numpy.ndarray, dict, or NoneType
        Optional keyword arguements for `ocb_rfunc`. If None is specified,
        uses function defaults.  If dict is specified, recasts as an array
        of this dict for all times.  Array must be an array of dicts.
        (default=None)

    Attributes
    ----------
    eab : ocbpy.EABoundary
        Equatorward auroral boundary data
    ocb : ocbpy.OCBoundary
        Open-closed field line boundary data

    Raises
    ------
    ValueError
        Incorrect or incompatible input

    """

    def __init__(self, eab_filename="default", ocb_filename="default",
                 eab_instrument='', ocb_instrument='', hemisphere=1,
                 eab_lat=64.0, ocb_lat=74.0, stime=None, etime=None,
                 eab_rfunc=None, eab_rfunc_kwargs=None, ocb_rfunc=None,
                 ocb_rfunc_kwargs=None, eab=None, ocb=None):

        # Initalize the subclass attributes
        if eab is None:
            self.eab = EABoundary(filename=eab_filename,
                                  instrument=eab_instrument,
                                  hemisphere=hemisphere, boundar_lat=eab_lat,
                                  stime=stime, etime=etime, rfunc=eab_rfunc,
                                  rfunc_kwargs=eab_rfunc_kwargs)
        else:
            self.eab = eab

        if ocb is None:
            self.ocb = OCBoundary(filename=ocb_filename,
                                  instrument=ocb_instrument,
                                  hemisphere=hemisphere, boundar_lat=ocb_lat,
                                  stime=stime, etime=etime, rfunc=ocb_rfunc,
                                  rfunc_kwargs=ocb_rfunc_kwargs)
        else:
            self.ocb = ocb

        return

    def __repr__(self):
        out_str = "".join([repr(self.__class__).split("'")[1], "(eab=",
                           repr(self.eab), ", ocb=", repr(self.ocb), ")"])

        return out_str

    def __str__(self):
        out_str = "Dual Boundary data\n{:s}\n{:s}".format(self.eab.__str__(),
                                                          self.ocb.__str__())

        return out_str
