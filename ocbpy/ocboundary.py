#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
"""Hold, manipulate, and load the open-closed field line boundary data

Functions
-------------------------------------------------------------------------------
match_data_ocb(ocb, dat_dtime, kwargs)
    Match data with open-closed field line boundaries

Classes
-------------------------------------------------------------------------------
OCBoundary    Loads, holds, and cycles the open-closed field line boundary data.
              Calculates magnetic coordinates relative to OCB (setting OCB at
              74 degrees latitude) given an AACGM location.

Moduleauthor
-------------------------------------------------------------------------------
Angeline G. Burrell (AGB), 15 April 2017, University of Texas, Dallas (UTDallas)

References
-------------------------------------------------------------------------------
Chisham, G. (2017), A new methodology for the development of high-latitude
 ionospheric climatologies and empirical models, Journal of Geophysical
 Research: Space Physics, 122, doi:10.1002/2016JA023235.
"""
import logbook as logging
import numpy as np

class OCBoundary(object):
    """ Object containing open-closed field-line boundary (OCB) data

    Parameters
    ----------
    filename : (str or NoneType)
        File containing the required open-closed circle boundary data sorted by
        time.  If NoneType, no file is loaded.  If 'default', the
        default IMAGE FUV file is loaded (if available). (default='default')
    instrument : (str)
        Instrument providing the OCBoundaries (default='image')
    hemisphere : (int)
        Integer (+/- 1) denoting northern/southern hemisphere (default=1)
    boundary_lat : (float)
        Typical AACGM latitude of the OCBoundary or None to use
        instrument defaults (default=None)
    stime : (datetime or NoneType)
        First time to load data or beginning of file (default=None)
    etime : (datetime or NoneType)
        Last time to load data or ending of file (default=None)

    Returns
    ---------
    self : OCBoundary class object containing OCB file data

    Attributes
    -----------
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
    (more) : (numpy.ndarray or NoneType)
        Numpy array of floats that hold the remaining values in input file

    Methods
    ---------- 
    inst_defaults()
        Get the information needed to load an OCB file using instrument
        specific formatting, and update the boundary latitude for a given
        instrument type.
    load(hlines=0, ocb_cols='year soy num_sectors phi_cent r_cent r a r_err',
         datetime_fmt='', stime=None, etime=None)
        Load the data from the OCB file specified by self.filename
    get_next_good_ocb_ind(min_sectors=7, rcent_dev=8.0, max_r=23.0, min_r=10.0,
                          min_j=0.15)
        Cycle to the next good OCB index
    normal_coord(aacgm_lat, aacgm_mlt)
        Calculate the OCB coordinates of an AACGM location
    revert_coord(ocb_lat, ocb_mlt)
        Calculate the AACGM location of OCB coordinates for this OCB
    """

    def __init__(self, filename="default", instrument="image", hemisphere=1,
                 boundary_lat=None, stime=None, etime=None):
        """Object containing OCB data

        Parameters
        ----------
        filename : (str or NoneType)
            File containing OCB data.  If None class structure will be
            initialised, but no file will be loaded.  If 'default', the
            default file will be loaded.
        instrument : (str)
            Instrument providing the OCBoundaries (default='image')
        hemisphere : (int)
            Integer (+/- 1) denoting northern/southern hemisphere (default=1)
        boundary_lat : (float)
            Typical AACGM latitude of the OCBoundary or None to use
            instrument defaults (default=None)
        stime : (datetime or NoneType)
            First time to load data or beginning of file (default=None)
        etime : (datetime or NoneType)
            Last time to load data or ending of file (default=None)
        """
        import ocbpy

        if not isinstance(instrument, str):
            estr = "OCB instrument must be a string [{:s}]".format(instrument)
            logging.error(estr)
            self.filename = None
            self.instrument = None
        else:
            self.instrument = instrument.lower()

            if filename is None:
                self.filename = None
            elif not isinstance(filename, str):
                logging.warning("file is not a string [{:s}]".format(filename))
                self.filename = None
            elif filename.lower() == "default":
                if instrument.lower() == "image":
                    ocb_dir = ocbpy.__file__.split("/")
                    self.filename = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                                       ocbpy.__default_file__)
                    if not ocbpy.instruments.test_file(self.filename):
                        logging.warning("problem with default OC Boundary file")
                        self.filename = None
                else:
                    logging.warning("default OC Boundary file uses IMAGE data")
                    self.filename = None
            elif not ocbpy.instruments.test_file(filename):
                logging.warning("cannot open OCB file [{:s}]".format(filename))
                self.filename = None
            else:
                self.filename = filename

        self.hemisphere = hemisphere
        self.records = 0
        self.rec_ind = 0
        self.dtime = None
        self.phi_cent = None
        self.r_cent = None
        self.r = None

        # Get the instrument defaults
        hlines, ocb_cols, datetime_fmt = self.inst_defaults()

        if boundary_lat is not None:
            self.boundary_lat = hemisphere * boundary_lat

        # If possible, load the data
        if self.filename is not None:
            if len(ocb_cols) > 0:
                self.load(hlines=hlines, ocb_cols=ocb_cols,
                          datetime_fmt=datetime_fmt, stime=stime, etime=etime)
            else:
                self.load(stime=stime, etime=etime)

        return

    def __repr__(self):
        """ Provide readable representation of the OCBoundary object
        """
    
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

                irep = sorted(set([0, 1, self.records - 2, self.records - 1]))
                while irep[0] < 0:
                    irep.pop(0)

                head = "YYYY-MM-DD HH:MM:SS Phi_Centre R_Centre R"
                out = "{:s}{:s}\n{:-<77s}\n".format(out, head, "")
                for i in irep:
                    out = "{:s}{:} {:.2f}".format(out, self.dtime[i],
                                                  self.phi_cent[i])
                    out = "{:s} {:.2f} {:.2f}\n".format(out, self.r_cent[i],
                                                        self.r[i])

        return out

    def __str__(self):
        """ Provide readable representation of the OCBoundary object
        """

        out = self.__repr__()
        return out

    def inst_defaults(self):
        """ Get the information needed to load an OCB file using instrument
        specific formatting, also update the boundary latitude for a given
        instrument type.

        Returns
        -------
        hlines : (int)
            Number of header lines
        ocb_cols : (str)
            String containing the names for each data column
        datetime_fmt : (str)
            String containing the datetime format
        """

        if self.instrument == "image":
            hlines = 0
            ocb_cols = "year soy num_sectors phi_cent r_cent r a r_err"
            datetime_fmt = ""
            self.boundary_lat = self.hemisphere * 74.0
        elif self.instrument == "ampere":
            hlines = 0
            ocb_cols = "date time r x y j_mag"
            datetime_fmt = "%Y%m%d %H:%M"
            self.boundary_lat = self.hemisphere * 72.0
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
        -----------
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

        Returns
        --------
        self
        """
        import datetime as dt
        import ocbpy.ocb_time as ocbt
        
        cols = ocb_cols.split()
        dflag = -1
        ldtype = [(k,float) if k != "num_sectors" else (k,int) for k in cols]
        
        if "soy" in cols and "year" in cols:
            dflag = 0
            ldtype[cols.index('year')] = ('year',int)
        elif "date" in cols and "time" in cols:
            dflag = 1
            ldtype[cols.index('date')] = ('date','|U50')
            ldtype[cols.index('time')] = ('time','|U50')

        if dflag < 0:
            logging.error("missing time columns in [{:s}]".format(ocb_cols))
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
                
            dtime = ocbt.convert_time(year=year, soy=soy, date=date, tod=tod,
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

        # Load the attributes saved in odata
        for nn in oname:
            setattr(self, nn, getattr(odata, nn)[itime])

        return

    def get_next_good_ocb_ind(self, min_sectors=7, rcent_dev=8.0, max_r=23.0,
                              min_r=10.0, min_j=0.15):
        """read in the next usuable OCB record from the data file.  Only uses
        the available parameters.

        Parameters
        -----------
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
        min_j : (float)
            Minimum unitless current magnitude scale difference (default=0.15)

        Returns
        ---------
        self
            updates self.rec_ind to the index of next good OCB record or a value
            greater than self.records if there aren't any more good records
            available after the starting point

        Comments
        ---------
        Checks:
        - more than 6 MLT boundary values have contributed to OCB circle
        - that the OCB 'pole' is with 8 degrees of the AACGM pole
        - that the OCB 'radius' is greater than 10 and less than 23 degrees
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
            elif hasattr(self, "j_mag") and self.j_mag[self.rec_ind] < min_j:
                good = False

            # Evaluate the current boundary for quality, using non-optional
            # parameters
            if(good and self.r_cent[self.rec_ind] <= rcent_dev and
               self.r[self.rec_ind] >= min_r and self.r[self.rec_ind] <= max_r):
                return

            # Cycle to next boundary
            self.rec_ind += 1

        return

    def normal_coord(self, aacgm_lat, aacgm_mlt):
        """converts the position of a measurement in AACGM co-ordinates to
        normalised co-ordinates relative to the OCB

        Parameters
        -----------
        aacgm_lat : (float)
            Input magnetic latitude (degrees)
        aacgm_mlt : (float)
            Input magnetic local time (hours)

        Returns
        --------
        ocb_lat : (float)
            Magnetic latitude relative to OCB (degrees)
        ocb_mlt : (float)
            Magnetic local time relative to OCB (hours)
 
        Comments
        ---------
        Approximation - Conversion assumes a planar surface
        """
        if self.rec_ind < 0 or self.rec_ind >= self.records:
            return np.nan, np.nan

        if np.sign(aacgm_lat) != self.hemisphere:
            return np.nan, np.nan

        phi_cent_rad = np.radians(self.phi_cent[self.rec_ind])
        xc = self.r_cent[self.rec_ind] * np.cos(phi_cent_rad)
        yc = self.r_cent[self.rec_ind] * np.sin(phi_cent_rad)

        scalep = 90.0 - self.hemisphere * aacgm_lat
        xp = scalep * np.cos(np.radians(aacgm_mlt * 15.0))
        yp = scalep * np.sin(np.radians(aacgm_mlt * 15.0))

        scalen = (90.0 - abs(self.boundary_lat)) / self.r[self.rec_ind]
        xn = (xp - xc) * scalen
        yn = (yp - yc) * scalen

        ocb_lat = self.hemisphere * (90.0 - np.sqrt(xn**2 + yn**2))
        ocb_mlt = np.degrees(np.arctan2(yn, xn)) / 15.0

        if ocb_mlt < 0.0:
            ocb_mlt += 24.0

        return ocb_lat, ocb_mlt

    def revert_coord(self, ocb_lat, ocb_mlt):
        """Converts the position of a measurement in normalised co-ordinates
        relative to the OCB into AACGM co-ordinates

        Parameters
        -----------
        ocb_lat : (float)
            Input OCB latitude (degrees)
        ocb_mlt : (float)
            Input OCB local time (hours)

        Returns
        --------
        aacgm_lat : (float)
            AACGM latitude (degrees)
        aacgm_mlt : (float)
            AACGM magnetic local time (hours)
 
        Comments
        ---------
        Approximation - Conversion assumes a planar surface
        """
        if self.rec_ind < 0 or self.rec_ind >= self.records:
            return np.nan, np.nan

        if np.sign(ocb_lat) != self.hemisphere:
            return np.nan, np.nan

        phi_cent_rad = np.radians(self.phi_cent[self.rec_ind])
        xc = self.r_cent[self.rec_ind] * np.cos(phi_cent_rad)
        yc = self.r_cent[self.rec_ind] * np.sin(phi_cent_rad)

        rn = 90.0 - self.hemisphere * ocb_lat

        thetan = ocb_mlt * np.pi / 12.0
        xn = rn * np.cos(thetan)
        yn = rn * np.sin(thetan)

        scale_ocb = self.r[self.rec_ind] / (90.0 - self.hemisphere *
                                            self.boundary_lat)
        xp = xn * scale_ocb + xc
        yp = yn * scale_ocb + yc

        aacgm_lat = self.hemisphere * (90.0 - np.sqrt(xp**2 + yp**2))
        aacgm_mlt = np.degrees(np.arctan2(yp, xp)) / 15.0

        if aacgm_mlt < 0.0:
            aacgm_mlt += 24.0

        return aacgm_lat, aacgm_mlt

def match_data_ocb(ocb, dat_dtime, idat=0, max_tol=600, min_sectors=7,
                   rcent_dev=8.0, max_r=23.0, min_r=10.0, min_j=0.15):
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
        Maximum radius for open-closed field line boundary in degrees.
        (default=23.0)
    min_r : (float)
        Minimum radius for open-closed field line boundary in degrees
        (default=10.0)
    min_j : (float)
        Minimum unitless current magnitude scale difference (default=0.15)

    Returns
    ---------
    idat : (int or NoneType)
        Data index for match value.  None if all of the data have been searched.

    Notes
    --------
    Updates OCBoundary.rec_ind for matched value.  None if all of the
    boundaries have been searched.
    """
    import ocbpy.ocboundary as ocboundary
    import datetime as dt

    dat_records = len(dat_dtime)

    # Ensure that the indices are good
    if idat >= dat_records:
        return idat
    if ocb.rec_ind >= ocb.records:
        return idat

    # Get the first reliable circle boundary estimate if none was provided
    if ocb.rec_ind < 0:
        ocb.get_next_good_ocb_ind(min_sectors=min_sectors, rcent_dev=rcent_dev,
                                  max_r=max_r, min_r=min_r, min_j=min_j)
        if ocb.rec_ind >= ocb.records:
            estr = "unable to find a good OCB record in "
            estr = "{:s}{:s}".format(estr, ocb.filename)
            logging.error(estr)
            return idat
        else:
            estr = "found first good OCB record at "
            estr = "{:s}{:}".format(estr, ocb.dtime[ocb.rec_ind])
            logging.info(estr)

        # Cycle past data occuring before the specified OC boundary point
        first_ocb = ocb.dtime[ocb.rec_ind] - dt.timedelta(seconds=max_tol)
        while dat_dtime[idat] < first_ocb:
            idat += 1

            if idat >= dat_records:
                logging.error("no input data close enough to first record")
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
                                      min_r=min_r, min_j=min_j)
        elif sdiff > max_tol:
            # Cycle to the next vorticity value if no OCB values were close
            # enough to grid this one
            estr = "no OCB data available within [{:d} s] of".format(max_tol)
            estr = "{:s} input measurement at ".format(estr)
            estr = "{:s}[{:}]".format(estr, dat_dtime[idat])
            logging.info(estr)
            idat += 1
        else:
            # Make sure this is the OCB value closest to the input record
            last_sdiff = sdiff
            last_iocb = ocb.rec_ind
            ocb.get_next_good_ocb_ind(min_sectors=min_sectors,
                                      rcent_dev=rcent_dev, max_r=max_r,
                                      min_r=min_r, min_j=min_j)

            if ocb.rec_ind < ocb.records:
                sdiff = (ocb.dtime[ocb.rec_ind] -
                         dat_dtime[idat]).total_seconds()

                while abs(sdiff) < abs(last_sdiff):
                    last_sdiff = sdiff
                    last_iocb = ocb.rec_ind
                    ocb.get_next_good_ocb_ind(min_sectors=min_sectors,
                                              rcent_dev=rcent_dev, max_r=max_r,
                                              min_r=min_r, min_j=min_j)
                    if ocb.rec_ind < ocb.records:
                        sdiff = (ocb.dtime[ocb.rec_ind] -
                                 dat_dtime[idat]).total_seconds()

            sdiff = last_sdiff
            ocb.rec_ind = last_iocb

            # Return the requested indices
            return idat

    # Return from the last loop
    return idat
