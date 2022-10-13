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
import warnings

import aacgmv2

from ocbpy import logger
import ocbpy.ocb_correction as ocbcor
from ocbpy import cycle_boundary
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
        Number of boundary records (default=0)
    rec_ind : int
        Current boundary record index (default=0; initialised=-1)
    dtime : numpy.ndarray or NoneType
        Numpy array of boundary datetimes (default=None)
    phi_cent : numpy.ndarray or NoneType
        Numpy array of floats that give the angle from AACGM midnight
        of the boundary pole in degrees (default=None)
    r_cent : numpy.ndarray or NoneType
        Numpy array of floats that give the AACGM co-latitude of the boundary
        pole in degrees (default=None)
    r : numpy.ndarray or NoneType
        Numpy array of floats that give the radius of the boundary in degrees
        (default=None)
    fom : numpy.ndarray or NoneType
        Numpy array of floats that provides a figure of merit that can be used
        to evaluate the quality of the boundary (default=None)
    min_fom : float
        Minimum acceptable figure of merit for data (default=-np.inf)
    max_fom : float
        Maximum acceptable figure of merit for data (default=np.inf)
    x, y, etc. : numpy.ndarray or NoneType
        Numpy array of floats that hold the remaining values held in `filename`

    Methods
    -------
    inst_defaults
        Get the instrument-specific boundary file loading information.
    load
        Load the data from the specified boundary file.
    get_next_good_ocb_ind
        Cycle to the the next quality boundary record.
    normal_coord
        Convert data position(s) to normalised co-ordinates relative to the OCB.
    revert_coord
        Convert the position of a measurement in OCB into AACGM co-ordinates.
    get_aacgm_boundary_lat
        Calculate the OCB latitude in AACGM coordinates at specified MLTs.

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
        self.fom = None
        self.rfunc = rfunc
        self.rfunc_kwargs = rfunc_kwargs
        self.min_fom = -np.inf
        self.max_fom = np.inf

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
                       ", hemisphere={:d}, ".format(int(self.hemisphere)),
                       "boundary_lat={:f}, stime=".format(self.boundary_lat),
                       repr(stime), ", etime=", repr(etime), ", rfunc=",
                       repr_rfunc, ", rfunc_kwargs=", repr_rfunc_kwargs, ")"])
        return out

    def __str__(self):
        """Provide readable representation of the OCBoundary object."""

        class_name = repr(self.__class__).split("'")[1].split(".")[-1]

        if self.filename is None:
            out = "No {:s} file specified\n".format(class_name)
        else:
            out = "{:s} file: {:s}\n".format(class_name, self.filename)
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
            ocb_cols = "year soy num_sectors phi_cent r_cent r a r_err fom"
            datetime_fmt = ""
            self.max_fom = 5.0  # From Chisham et al. (in prep)
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
             ocb_cols="year soy num_sectors phi_cent r_cent r a r_err fom",
             datetime_fmt="", stime=None, etime=None):
        """Load the data from the specified boundary file.

        Parameters
        ----------
        ocb_cols : str
            String specifying format of OCB file.  All but the first two
            columns must be included in the string, additional data values will
            be ignored.  If 'year soy' aren't used, expects
            'date time' in 'YYYY-MM-DD HH:MM:SS' format.
            (default='year soy num_sectors phi_cent r_cent r a r_err r_merit')
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

    def get_next_good_ocb_ind(self, min_merit=None, max_merit=None, **kwargs):
        """Cycle to the the next quality OCB record.

        Parameters
        ----------
        min_merit : float or NoneType
            Minimum value for the default figure of merit or None to not
            apply a custom minimum (default=None)
        max_merit : float or NoneTye
            Maximum value for the default figure of merit or None to not apply
            a custom maximum (default=None)
        min_sectors : int
            Minimum number of MLT sectors required for good OCB. Deprecated,
            will be removed in version 0.3.1+ (default=7)
        rcent_dev : float
            Maximum number of degrees between the new centre and the AACGM pole
             Deprecated, will be removed in version 0.3.1+ (default=8.0)
        max_r : float
            Maximum radius for open-closed field line boundary in degrees.
            Deprecated, will be removed in version 0.3.1+ (default=23.0)
        min_r : float
            Minimum radius for open-closed field line boundary in degrees
            Deprecated, will be removed in version 0.3.1+ (default=10.0)
        kwargs : dict
            Dict with optional selection criteria.  The key should correspond
            to a data attribute and the value must be a tuple with the first
            value specifying 'max', 'min', 'maxeq', 'mineq', or 'equal' and the
            second value specifying the value to use in the comparison.

        Notes
        -----
        Updates self.rec_ind to the index of next good OCB record or a value
        greater than self.records if there aren't any more good records
        available after the starting point

        Deprecated IMAGE FUV checks that:
        - more than 6 MLT boundary values have contributed to OCB circle
        - the OCB 'pole' is with 8 degrees of the AACGM pole
        - the OCB 'radius' is greater than 10 and less than 23 degrees
        AMPERE/DMSP-SSJ and new IMAGE FUV checks that:
        - the Figure of Merit is greater than or equal to the specified minimum
          (`min_fom`) or less than or equal to the specified maximum (`max_fom`)

        """

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

                if hasattr(self, dep_comp[ckey][0]):
                    kwargs[dep_comp[ckey][0]] = dep_comp[ckey][1]
            else:
                if not hasattr(self, ckey):
                    logger.warning(
                        "Removing unknown selection attribute {:}".format(ckey))
                    del kwargs[ckey]

        # Adjust the FoM determination for custom inputs
        if min_merit is None:
            min_merit = self.min_fom

        if max_merit is None:
            max_merit = self.max_fom

        # Incriment forward from previous boundary
        self.rec_ind += 1

        while self.rec_ind < self.records:
            # Evaluate the current boundary for quality, using optional
            # parameters
            good = True

            for ckey in kwargs.keys():
                test_val = getattr(self, ckey)[self.rec_ind]
                if kwargs[ckey][0] == "min" and test_val <= kwargs[ckey][1]:
                    good = False
                    break
                elif kwargs[ckey][0] == "mineq" and test_val < kwargs[ckey][1]:
                    good = False
                    break
                elif kwargs[ckey][0] == "equal" and test_val != kwargs[ckey][1]:
                    good = False
                    break
                elif kwargs[ckey][0] == "maxeq" and test_val > kwargs[ckey][1]:
                    good = False
                    break
                elif kwargs[ckey][0] == "max" and test_val >= kwargs[ckey][1]:
                    good = False
                    break

            # Evaluate the current boundary for quality, using non-optional
            # parameters
            if good:
                test_val = self.fom[self.rec_ind]
                if test_val >= min_merit and test_val <= max_merit:
                    return

            # Cycle to next boundary
            self.rec_ind += 1

        return

    def normal_coord(self, lat, lt, coords='magnetic', height=350.0,
                     method='ALLOWTRACE'):
        """Convert position(s) to normalised co-ordinates relative to the OCB.

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
        xp = scalep * np.cos(ocb_time.hr2rad(aacgm_mlt))
        yp = scalep * np.sin(ocb_time.hr2rad(aacgm_mlt))

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
        """Calculate the OCB latitude in AACGM coordinates at specified MLTs.

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

        If the boundary radius is not defined at all MLT (possible for
        poorly constrained boundaries), then MLT with multiple boundary values
        will only return one possible solution.

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

    See Also
    --------
    ocbpy.OCBoundary

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
            self._set_default_rfunc(instrument)
            rfunc = self.rfunc
            self.rfunc = None

        if(hasattr(filename, "lower") and hasattr(instrument, "lower")
           and filename.lower() == "default"):
            filename, instrument = get_default_file(stime, etime, hemisphere,
                                                    instrument, bound='eab')

        # Initialize the class
        OCBoundary.__init__(self, filename=filename, instrument=instrument,
                            hemisphere=hemisphere, boundary_lat=boundary_lat,
                            stime=stime, etime=etime, rfunc=rfunc,
                            rfunc_kwargs=rfunc_kwargs)

        return

    def _set_default_rfunc(self, input_instrument):
        """Set the default instrument EAB boundary function.

        Parameters
        ----------
        input_instrument : str
           Input instrument name

        Notes
        -----
        Assign a function for each time in case we have a data set with a
        correction that changes with UT

        """

        if input_instrument in ["", "default", "image", "dmsp-ssj"]:
            self.rfunc = ocbcor.circular
        elif not hasattr(input_instrument, "lower"):
            # Allow an empty class object to be initialised
            self.rfunc = None
        else:
            # Only raise an error if the input instrument is a string
            raise ValueError("unknown instrument")

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
    eab : ocbpy.EABoundary or NoneType
        Equatorward auroral boundary data object or None to initialize here
        (default=None)
    ocb : ocbpy.OCBoundary
        Open-closed field line boundary data oject or Noneto initialize here
        (default=None)
    max_delta : int
        Maximum number of seconds allowed between paired EAB and OCB records
        (default=60)

    Attributes
    ----------
    eab
    ocb
    max_delta
    hemisphere
    dtime : numpy.ndarray
        Numpy array of paired boundary datetimes
    eab_ind : numpy.ndarray
        Numpy array of EAB indices for a good quality paired boundary
    ocb_ind : numpy.ndarray
        Numpy array of OCB indices for a good quality paired boundary
    rec_ind : int
        Current OCB record index (default=0; initialised=-1)
    records : int
        Maximum number of paired boundary records

    Methods
    -------
    set_good_ind
        Pair the good indices for the quality EABs and OCBs.
    get_next_good_ind
        Cycle to the the next quality paired boundary record.
    normal_coord
        Convert data position(s) to normalised co-ordinates relative to the OCB.
    revert_coord
        Convert the position of a measurement in OCB into AACGM co-ordinates.
    get_aacgm_boundary_lats
        Calculate the EAB and OCB latitude in AACGM coordinates.
    calc_r
        Calculate the scaled and unscaled radius at a normalised co-ordinates.

    Raises
    ------
    ValueError
        Incorrect or incompatible input, mismatched hemisphere assignment

    """

    def __init__(self, eab_filename="default", ocb_filename="default",
                 eab_instrument='', ocb_instrument='', hemisphere=1,
                 eab_lat=64.0, ocb_lat=74.0, stime=None, etime=None,
                 eab_rfunc=None, eab_rfunc_kwargs=None, ocb_rfunc=None,
                 ocb_rfunc_kwargs=None, eab=None, ocb=None, max_delta=60):

        # Ensure the boundary latitudes are realistic
        if abs(eab_lat) >= abs(ocb_lat):
            raise ValueError('OCB must be poleward of the EAB')

        if max_delta <= 0:
            raise ValueError(''.join(['Maximum time difference between ',
                                      'boundary measurements must be positive',
                                      ' or zero']))

        # Initalize the subclass attributes
        if eab is None:
            self.eab = EABoundary(filename=eab_filename,
                                  instrument=eab_instrument,
                                  hemisphere=hemisphere, boundary_lat=eab_lat,
                                  stime=stime, etime=etime, rfunc=eab_rfunc,
                                  rfunc_kwargs=eab_rfunc_kwargs)
        else:
            self.eab = eab

        if ocb is None:
            self.ocb = OCBoundary(filename=ocb_filename,
                                  instrument=ocb_instrument,
                                  hemisphere=hemisphere, boundary_lat=ocb_lat,
                                  stime=stime, etime=etime, rfunc=ocb_rfunc,
                                  rfunc_kwargs=ocb_rfunc_kwargs)
        else:
            self.ocb = ocb

        # Set the hemisphere
        self.hemisphere = self.ocb.hemisphere

        if self.hemisphere != self.eab.hemisphere:
            raise ValueError('mismatched hemisphere assignment')

        # Create a time index, saving indices where both boundaries are good
        self.max_delta = max_delta
        self.set_good_ind()

        return

    def __repr__(self):
        out_str = "".join([repr(self.__class__).split("'")[1], "(eab=",
                           repr(self.eab), ", ocb=", repr(self.ocb),
                           ", max_delta=", repr(self.max_delta), ")"])

        return out_str

    def __str__(self):
        out = "Dual Boundary data\n{:d} good boundary ".format(self.records)
        if self.records == 0:
            out = "{:s}pairs\nMaximum ".format(out)
        else:
            out = "{:s}pairs from {:} to {:}\nMaximum ".format(
                out, self.dtime[0], self.dtime[-1])
        out = "{:s} boundary difference of {:.1f} s".format(out, self.max_delta)
        out = "{:s}\n\n{:s}\n{:s}\n".format(out, self.eab.__str__(),
                                            self.ocb.__str__())

        return out

    def _get_current_aacgm_boundary(self):
        """Get the current AACGM boundaries for the EAB and OCB classes.

        Returns
        -------
        eab_blat : array-like or NoneType
            EAB boundary latitude in AACGM coordinates at the record index,
            None of no boundary is set.
        eab_bmlt : array-like or NoneType
            EAB boundary MLT in AACGM coordinates at the record index,
            None of no boundary is set.
        ocb_blat : array-like or NoneType
            OCB boundary latitude in AACGM coordinates at the record index,
            None of no boundary is set.
        ocb_bmlt : array-like or NoneType
            OCB boundary MLT in AACGM coordinates at the record index,
            None of no boundary is set.

        """

        # Set the OCB output
        if hasattr(self.ocb, "aacgm_boundary_lat"):
            ocb_blat = self.ocb.aacgm_boundary_lat[self.ocb.rec_ind]
            ocb_bmlt = self.ocb.aacgm_boundary_mlt[self.ocb.rec_ind]
        else:
            ocb_blat = None
            ocb_bmlt = None

        # Set the EAB output
        if hasattr(self.eab, "aacgm_boundary_lat"):
            eab_blat = self.eab.aacgm_boundary_lat[self.eab.rec_ind]
            eab_bmlt = self.eab.aacgm_boundary_mlt[self.eab.rec_ind]
        else:
            eab_blat = None
            eab_bmlt = None

        return eab_blat, eab_bmlt, ocb_blat, ocb_bmlt

    @property
    def rec_ind(self):
        """Record index that identifies the current good EAB/OCB pair."""
        return self._rec_ind

    @rec_ind.setter
    def rec_ind(self, rec_ind):
        # Set the record index, and cycle the sub-class boundary indices
        self._rec_ind = rec_ind

        if rec_ind < 0:
            self.ocb.rec_ind = rec_ind
            self.eab.rec_ind = rec_ind
        elif rec_ind >= self.records:
            del_ind = self.records - rec_ind
            self.ocb.rec_ind = self.ocb.records + del_ind
            self.eab.rec_ind = self.eab.records + del_ind
        else:
            self.ocb.rec_ind = self.ocb_ind[rec_ind]
            self.eab.rec_ind = self.eab_ind[rec_ind]

        return

    def set_good_ind(self, ocb_min_merit=None, ocb_max_merit=None,
                     ocb_kwargs=None, eab_min_merit=None, eab_max_merit=None,
                     eab_kwargs=None):
        """Pair the good indices for the quality EABs and OCBs.

        Parameters
        ----------
        ocb_min_merit : float or NoneType
            Minimum value for the default figure of merit or None to not apply
            a custom minimum (default=None)
        ocb_max_merit : float or NoneTye
            Maximum value for the default figure of merit or None to not apply
            a custom maximum (default=None)
        ocb_kwargs : dict or NoneType
            Dict with optional selection criteria.  The key should correspond
            to a data attribute and the value must be a tuple with the first
            value specifying 'max', 'min', 'maxeq', 'mineq', or 'equal' and the
            second value specifying the value to use in the comparison.
            None provides no optional selection criteria. (default=None)
        eab_min_merit : float or NoneType
            Minimum value for the default figure of merit or None to not apply
            a custom minimum (default=None)
        eab_max_merit : float or NoneTye
            Maximum value for the default figure of merit or None to not apply
            a custom maximum (default=None)
        eab_kwargs : dict or NoneType
            Dict with optional selection criteria.  The key should correspond
            to a data attribute and the value must be a tuple with the first
            value specifying 'max', 'min', 'maxeq', 'mineq', or 'equal' and the
            second value specifying the value to use in the comparison.
            None provides no optional selection criteria. (default=None)

        """

        # Initalize the class attributes
        if self.ocb.dtime is None or self.eab.dtime is None:
            self.records = 0
            self.dtime = None
            self.ocb_ind = None
            self.eab_ind = None
            return

        self.dtime = list()
        self.ocb_ind = list()
        self.eab_ind = list()

        # Save the current EAB record index
        icurrent = self.eab.rec_ind
        self.eab.rec_ind = -1

        # Get the good OCB indices
        if ocb_kwargs is None:
            ocb_kwargs = {}
        ocb_kwargs['min_merit'] = ocb_min_merit
        ocb_kwargs['max_merit'] = ocb_max_merit
        good_ocb = cycle_boundary.retrieve_all_good_indices(self.ocb,
                                                            **ocb_kwargs)

        # Initialize the EAB good index selection options
        if eab_kwargs is None:
            eab_kwargs = {}
        eab_kwargs['min_merit'] = eab_min_merit
        eab_kwargs['max_merit'] = eab_max_merit

        # Match the EABs with the good OCB times
        iocb = 0
        while self.eab.rec_ind < self.eab.records and iocb is not None:
            # Cycle the OCB record index to match the next good EAB index
            iocb = cycle_boundary.match_data_ocb(
                self.eab, self.ocb.dtime[good_ocb], idat=iocb,
                max_tol=self.max_delta, **eab_kwargs)

            # Save the paired data
            if iocb is not None and iocb < len(good_ocb):
                self.dtime.append(self.ocb.dtime[good_ocb[iocb]])
                self.ocb_ind.append(good_ocb[iocb])
                self.eab_ind.append(self.eab.rec_ind)
            else:
                iocb = None

            # Cycle to the next good EAB index
            self.eab.get_next_good_ocb_ind(**eab_kwargs)

        # Re-cast the class attributes as arrays
        self.dtime = np.asarray(self.dtime)
        self.ocb_ind = np.asarray(self.ocb_ind)
        self.eab_ind = np.asarray(self.eab_ind)

        # Reset the EAB index
        self.eab.rec_ind = icurrent

        # Set the number of good paired records
        self.records = len(self.dtime)

        # Set the data to the first good record
        self.rec_ind = 0

        return

    def get_next_good_ind(self):
        """Cycle the boundary attributes to the next good paired index."""

        if self.records > 0 and self.rec_ind < self.records:
            # Cycle to next boundary
            self.rec_ind += 1

        return

    def normal_coord(self, lat, lt, coords='magnetic', height=350.0,
                     method='ALLOWTRACE', overwrite=False):
        """Convert coordinates to be normalised relative to the EAB and OCB.

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
        overwrite : bool
            Allow the OCB and EAB AACGM boundary locations to be overwritten
            (default=False)

        Returns
        -------
        bound_lat : float or array-like
            Magnetic latitude relative to EAB and OCB (degrees)
        bound_mlt : float or array-like
            Magnetic local time relative to EAB and OCB (hours)
        ocb_lat : float or array-like
            Magnetic latitude relative to only the OCB (degrees)
        r_corr : float or array-like
            Radius correction to OCB (degrees)

        Notes
        -----
        Approximation - Conversion assumes a planar surface

        Defines `bound_mlt` relative to only the OCB.

        See Also
        --------
        aacgmv2
        ocbpy.OCBoundary.normal_coord

        """

        # Cast input as arrays
        lat = np.asarray(lat)
        lt = np.asarray(lt)
        height = np.asarray(height)
        out_shape = max([lat.shape, lt.shape, height.shape])

        if out_shape == ():
            lat = np.array([lat])
            lt = np.array([lt])
            height = np.array([height])
        else:
            # Ensure the input allows simple broadcasting
            if lat.shape != out_shape:
                if lat.shape == ():
                    lat = np.full(shape=out_shape, fill_value=float(lat))
                else:
                    raise ValueError('mismatched input shape for latitude')

            if lt.shape != out_shape:
                if lt.shape == ():
                    lt = np.full(shape=out_shape, fill_value=float(lt))
                else:
                    raise ValueError('mismatched input shape for local time')

            if height.shape != out_shape:
                if height.shape == ():
                    height = np.full(shape=out_shape, fill_value=float(height))
                else:
                    raise ValueError('mismatched input shape for height')

        # Test the dual-boundary record index
        if self.rec_ind < 0 or self.rec_ind >= self.records:
            if out_shape == ():
                return np.nan, np.nan, np.nan, np.nan
            else:
                bound_lat = np.full(shape=out_shape, fill_value=np.nan)
                bound_mlt = np.full(shape=out_shape, fill_value=np.nan)
                ocb_lat = np.full(shape=out_shape, fill_value=np.nan)
                r_corr = np.full(shape=out_shape, fill_value=np.nan)
                return bound_lat, bound_mlt, ocb_lat, r_corr

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
            aacgm_lat = np.array(lat)
            aacgm_mlt = np.array(lt)

        # Calculate the coordinates relative to the OCB
        ocb_lat, bound_mlt, r_corr = self.ocb.normal_coord(aacgm_lat, aacgm_mlt,
                                                           coords='magnetic',
                                                           height=height,
                                                           method=method)
        bound_lat = np.array(ocb_lat)

        if np.isnan(ocb_lat).all():
            if out_shape == ():
                return bound_lat[0], bound_mlt[0], ocb_lat[0], r_corr[0]
            else:
                return bound_lat, bound_mlt, ocb_lat, r_corr

        # Get the boundary locations in AACGM coordinates
        if not overwrite:
            orig_bound = self._get_current_aacgm_boundary()
            num_none = sum([obound is None for obound in orig_bound])

            if len(orig_bound) == num_none:
                overwrite = True

        self.get_aacgm_boundary_lats(aacgm_mlt, rec_ind=self.rec_ind,
                                     overwrite=True, set_lon=False)
        ocb_aacgm_boundary = self.ocb.aacgm_boundary_lat[self.ocb.rec_ind]
        eab_aacgm_boundary = self.eab.aacgm_boundary_lat[self.eab.rec_ind]

        # Normalize each of the points using the correct scaling factor
        imid = np.where((abs(aacgm_lat) < abs(ocb_aacgm_boundary))
                        & (abs(aacgm_lat) >= abs(eab_aacgm_boundary)))[0]
        iout = np.where(abs(aacgm_lat) < abs(eab_aacgm_boundary))[0]

        # Test to ensure that all points fall into a region
        iocb = np.where(abs(ocb_lat) >= abs(self.ocb.boundary_lat))[0]
        if len(iocb) + len(imid) + len(iout) < len(aacgm_lat):
            msg = "not all points fall into a boundary region"
            if np.isnan(ocb_aacgm_boundary).any() or np.isnan(
                    eab_aacgm_boundary).any():
                msg += ", boundaries are poorly defined"
            logger.warning(msg)

            ibad = [i for i in range(len(aacgm_lat)) if i not in iocb
                    and i not in imid and i not in iout]
            bound_lat[ibad] = np.nan

        if len(imid) > 0:
            bound_lat[imid] = self.ocb.boundary_lat - (
                ocb_aacgm_boundary[imid] - aacgm_lat[imid]) * (
                    self.ocb.boundary_lat - self.eab.boundary_lat) / (
                        ocb_aacgm_boundary[imid] - eab_aacgm_boundary[imid])

        if len(iout) > 0:
            bound_lat[iout] = self.eab.boundary_lat - (
                eab_aacgm_boundary[iout] - aacgm_lat[iout]) * (
                    self.eab.boundary_lat / eab_aacgm_boundary[iout])

        # If desired, replace the boundaries
        if not overwrite:
            if orig_bound[2] is not None:
                self.ocb.aacgm_boundary_lat[self.ocb.rec_ind] = orig_bound[2]
                self.ocb.aacgm_boundary_mlt[self.ocb.rec_ind] = orig_bound[3]
            if orig_bound[0] is not None:
                self.eab.aacgm_boundary_lat[self.eab.rec_ind] = orig_bound[0]
                self.eab.aacgm_boundary_mlt[self.eab.rec_ind] = orig_bound[1]

        if out_shape == ():
            return bound_lat[0], bound_mlt[0], ocb_lat[0], r_corr[0]
        else:
            return bound_lat, bound_mlt, ocb_lat, r_corr

    def revert_coord(self, ocb_lat, ocb_mlt, r_corr=0.0, is_ocb=True,
                     aacgm_mlt=None, coords='magnetic', height=350.0,
                     method='ALLOWTRACE', overwrite=False):
        """Convert from OCB or dual-boundary into AACGM co-ordinates.

        Parameters
        ----------
        ocb_lat : float or array-like
            Input OCB or dual-boundary latitude in degrees
        ocb_mlt : float or array-like
            Input OCB/dual-boundary local time in hours
        r_corr : float or array-like
            Input OCB radial correction in degrees, may be a function of
            AACGM MLT (default=0.0)
        is_ocb : bool
            Specifies that the input of `ocb_lat` is in OCB coordinates if True
            or in dual-boundary coordinates if False.  If False, `aacgm_mlt`
            must be provided (default=True)
        aacgm_mlt : float, array-like, or NoneType
            Output AACGM MLT of the dual-boundary data, only used if `is_ocb`
            is False (default=None)
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
        overwrite : bool
            Allow the OCB and EAB AACGM boundary locations to be overwritten
            (default=False)

        Returns
        -------
        lat : float or array-like
            latitude (degrees)
        lt : float or array-like
            local time (hours)

        Raises
        ------
        ValueError
            When necessary inputs are not fully supplied

        Notes
        -----
        Approximation - Conversion assumes a planar surface

        See Also
        --------
        aacgmv2
        ocbpy.OCBoundary.revert_coord

        """

        # Cast input as arrays and prepare the output
        ocb_lat = np.asarray(ocb_lat)
        ocb_mlt = np.asarray(ocb_mlt)
        height = np.asarray(height)
        out_shape = max([ocb_lat.shape, ocb_mlt.shape, height.shape])

        if out_shape == ():
            ocb_lat = np.array([ocb_lat])
            ocb_mlt = np.array([ocb_mlt])
            height = np.array([height])
        else:
            # Ensure the input allows simple broadcasting
            if ocb_lat.shape != out_shape:
                if ocb_lat.shape == ():
                    ocb_lat = np.full(shape=out_shape,
                                      fill_value=float(ocb_lat))
                else:
                    raise ValueError('mismatched input shape for OCB latitude')

            if ocb_mlt.shape != out_shape:
                if ocb_mlt.shape == ():
                    ocb_mlt = np.full(shape=out_shape,
                                      fill_value=float(ocb_mlt))
                else:
                    raise ValueError('mismatched input shape for OCB MLT')

            if height.shape != out_shape:
                if height.shape == ():
                    height = np.full(shape=out_shape, fill_value=float(height))
                else:
                    raise ValueError('mismatched input shape for height')

        # Get the OCB reversion
        lat, lt = self.ocb.revert_coord(ocb_lat, ocb_mlt, r_corr=r_corr,
                                        coords=coords, height=height,
                                        method=method)

        # If not already flagged, see if all OCB by location
        if not is_ocb:
            iocb = np.where(abs(ocb_lat) >= abs(self.ocb.boundary_lat))[0]

            if len(ocb_lat) == len(iocb):
                is_ocb = True

        # Continue calculating the output for the other regions, if needed
        if not is_ocb:
            # Ensure the AACGM MLT was provided on input
            if aacgm_mlt is None:
                raise ValueError(
                    ''.join(['cannot revert dual-boundary coordinates without',
                             ' the AACGM MLT']))

            aacgm_mlt = np.asarray(aacgm_mlt)

            if aacgm_mlt.shape == ():
                if out_shape == ():
                    aacgm_mlt = np.array([aacgm_mlt])
                else:
                    aacgm_mlt = np.full(shape=out_shape,
                                        fill_value=float(aacgm_mlt))

            # Get the boundary locations in AACGM coordinates
            if not overwrite:
                orig_bound = self._get_current_aacgm_boundary()
                num_none = sum([obound is None for obound in orig_bound])

                if len(orig_bound) == num_none:
                    overwrite = True

            self.get_aacgm_boundary_lats(aacgm_mlt, rec_ind=self.rec_ind,
                                         overwrite=True, set_lon=False)
            ocb_aacgm_boundary = self.ocb.aacgm_boundary_lat[self.ocb.rec_ind]
            eab_aacgm_boundary = self.eab.aacgm_boundary_lat[self.eab.rec_ind]

            # Raise warning for poorly defined boundaries, may cause NaN output
            if np.isnan(ocb_aacgm_boundary).any() or np.isnan(
                    eab_aacgm_boundary).any():
                msg = "".join(["not all points fall into a boundary region",
                               ", boundaries are poorly defined"])
                logger.warning(msg)

            # Revert each of the points using the correct scaling factor
            imid = np.where((abs(ocb_lat) < abs(self.ocb.boundary_lat))
                            & (abs(ocb_lat) >= abs(self.eab.boundary_lat)))[0]
            iout = np.where(abs(ocb_lat) < abs(self.eab.boundary_lat))[0]

            if len(imid) > 0:
                lat[imid] = ocb_aacgm_boundary[imid] - (
                    self.ocb.boundary_lat - ocb_lat[imid]) * (
                        ocb_aacgm_boundary[imid] - eab_aacgm_boundary[imid]) / (
                            self.ocb.boundary_lat - self.eab.boundary_lat)
                lt[imid] = aacgm_mlt[imid]

            if len(iout) > 0:
                lat[iout] = eab_aacgm_boundary[iout] - (
                    self.eab.boundary_lat - ocb_lat[iout]) * (
                    eab_aacgm_boundary[iout] / self.eab.boundary_lat)
                lt[iout] = aacgm_mlt[iout]

            # If desired, replace the boundaries
            if not overwrite:
                if orig_bound[2] is not None:
                    self.ocb.aacgm_boundary_lat[
                        self.ocb.rec_ind] = orig_bound[2]
                    self.ocb.aacgm_boundary_mlt[
                        self.ocb.rec_ind] = orig_bound[3]
                if orig_bound[0] is not None:
                    self.eab.aacgm_boundary_lat[
                        self.eab.rec_ind] = orig_bound[0]
                    self.eab.aacgm_boundary_mlt[
                        self.eab.rec_ind] = orig_bound[1]

            # If needed, convert from magnetic to geographic coordinates
            if coords.lower().find('mag') < 0:
                imag = list(imid) + list(iout)

                # Convert from mlt to longitude
                lon = aacgmv2.convert_mlt(aacgm_mlt[imag],
                                          self.dtime[self.rec_ind], m2a=True)

                # If geocentric coordinates are specified, add this info to the
                # method flag
                if coords.lower() == 'geocentric':
                    method = "|".join([method, coords.upper()])
                method = "|".join([method, "A2G"])

                # Convert from magnetic coordinates to geo coordinates
                lat[imag], lon, _ = aacgmv2.convert_latlon_arr(
                    lat[imag], lon, height, self.dtime[self.rec_ind], method)

                # Convert from longitude to solar local time
                lt[imag] = ocb_time.glon2slt(lon, self.dtime[self.rec_ind])

        if out_shape == ():
            return lat[0], lt[0]
        else:
            return lat, lt

    def get_aacgm_boundary_lats(self, aacgm_mlt, rec_ind=None,
                                overwrite=False, set_lon=True):
        """Calculate the OCB latitude in AACGM coordinates at specified MLTs.

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

        See Also
        --------
        ocbpy.OCBoundary.get_aacgm_boundary_lat

        """

        # Get the desired EAB and OCB record indices
        if rec_ind is None:
            eab_rec_ind = None
            ocb_rec_ind = None
        else:
            eab_rec_ind = self.eab_ind[rec_ind]
            ocb_rec_ind = self.ocb_ind[rec_ind]

        # Calculate the boundary locations
        self.eab.get_aacgm_boundary_lat(aacgm_mlt, rec_ind=eab_rec_ind,
                                        overwrite=overwrite, set_lon=set_lon)
        self.ocb.get_aacgm_boundary_lat(aacgm_mlt, rec_ind=ocb_rec_ind,
                                        overwrite=overwrite, set_lon=set_lon)
        return

    def calc_r(self, bound_lat, bound_mlt, aacgm_mlt, r_corr, overwrite=False):
        """Calculate the scaled and unscaled radius at a normalised co-ordinate.

        Parameters
        ----------
        bound_lat : array-like or float
            Normalised dual-boundary latitude in degrees
        bound_mlt : array-like or float
            Normalised dual-boundary MLT in hours
        aacgm_mlt : array-like or float
            MLT in AACGM coordinates in hours
        r_corr : array-like or float
            OCB radial correction in degrees
        overwrite : bool
            Overwrite previous boundary locations if this time already has
            calculated boundary latitudes for a different set of input
            longitudes (default=False).

        Returns
        -------
        scaled_r : array-like
            Scaled radius for the region (OCB, EAB, Sub-auroral) in degrees
        unscaled_r : array-like
            Unscaled radius for the region (OCB, EAB, Sub-auroral) in degrees

        """

        # Ensure all data is array-like
        bound_lat = np.asarray(bound_lat)
        bound_mlt = np.asarray(bound_mlt)
        aacgm_mlt = np.asarray(aacgm_mlt)
        r_corr = np.asarray(r_corr)

        isfloat = False
        if bound_lat.shape == ():
            bound_lat = np.array([bound_lat])
            bound_mlt = np.array([bound_mlt])
            aacgm_mlt = np.array([aacgm_mlt])
            r_corr = np.array([r_corr])
            isfloat = True

        # Initialize the output, start by assuming the data is inside the OCB
        out_shape = max([bound_lat.shape, bound_mlt.shape, r_corr.shape])
        scaled_r = np.full(shape=out_shape,
                           fill_value=90.0 - abs(self.ocb.boundary_lat))

        if self.rec_ind < 0 or self.rec_ind >= self.records:
            unscaled_r = np.full(shape=out_shape, fill_value=np.nan)
            return scaled_r, unscaled_r

        if out_shape == r_corr.shape:
            unscaled_r = self.ocb.r[self.ocb_ind[self.rec_ind]] + r_corr
        else:
            unscaled_r = np.full(shape=out_shape, fill_value=self.ocb.r[
                self.ocb_ind[self.rec_ind]] + r_corr)

        # Identify points in the other regions
        imid = np.where((abs(bound_lat) < abs(self.ocb.boundary_lat))
                        & (abs(bound_lat) >= abs(self.eab.boundary_lat)))[0]
        iout = np.where(abs(bound_lat) < abs(self.eab.boundary_lat))[0]

        # Get the boundary locations in AACGM coordinates
        if not overwrite:
            orig_bound = self._get_current_aacgm_boundary()
            num_none = sum([obound is None for obound in orig_bound])

            if len(orig_bound) == num_none:
                overwrite = True

        self.get_aacgm_boundary_lats(aacgm_mlt, rec_ind=self.rec_ind,
                                     overwrite=True, set_lon=False)
        ocb_aacgm_boundary = self.ocb.aacgm_boundary_lat[self.ocb.rec_ind]
        eab_aacgm_boundary = self.eab.aacgm_boundary_lat[self.eab.rec_ind]

        # Calculate the radii in the auroral region
        if len(imid) > 0:
            scaled_r[imid] = abs(self.ocb.boundary_lat - self.eab.boundary_lat)
            unscaled_r[imid] = abs(ocb_aacgm_boundary[imid]
                                   - eab_aacgm_boundary[imid])

        if len(iout) > 0:
            scaled_r[iout] = abs(self.eab.boundary_lat)
            unscaled_r[iout] = abs(eab_aacgm_boundary[iout])

        # If desired, replace the boundaries
        if not overwrite:
            if orig_bound[2] is not None:
                self.ocb.aacgm_boundary_lat[self.ocb.rec_ind] = orig_bound[2]
                self.ocb.aacgm_boundary_mlt[self.ocb.rec_ind] = orig_bound[3]
            if orig_bound[0] is not None:
                self.eab.aacgm_boundary_lat[self.eab.rec_ind] = orig_bound[0]
                self.eab.aacgm_boundary_mlt[self.eab.rec_ind] = orig_bound[1]

        if isfloat:
            return scaled_r[0], unscaled_r[0]
        else:
            return scaled_r, unscaled_r
