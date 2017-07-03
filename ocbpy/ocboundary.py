#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
"""Hold, manipulate, and load the open-closed field line boundary data

Routines
-------------------------------------------------------------------------------
year_soy_to_datetime    Converts from seconds of year to datetime
match_data_ocb          Match data with open-closed field line boundaries
-------------------------------------------------------------------------------

Classes
-------------------------------------------------------------------------------
OCBoundary    Loads, holds, and cycles the open-closed field line boundary data.
              Calculates magnetic coordinates relative to OCB (setting OCB at
              74 degrees latitude) given an AACGM location.
-------------------------------------------------------------------------------

Moduleauthor
-------------------------------------------------------------------------------
Angeline G. Burrell (AGB), 15 April 2017, University of Texas, Dallas (UTDallas)
-------------------------------------------------------------------------------

References
-------------------------------------------------------------------------------
Chisham, G. (2016), A new methodology for the development of high-latitude
 ionospheric climatologies and empirical models, Journal of Geophysical
 Research: Space Physics, 122, doi:10.1002/2016JA023235.
-------------------------------------------------------------------------------
"""
import logging
import numpy as np

class OCBoundary(object):
    """ Object containing open-closed field-line boundary (OCB) data

    Parameters
    ----------
    filename : (str or NoneType)
        file containing the required open-closed circle boundary data sorted by
        time.  If NoneType, the recommended default file is used (if available)

    Returns
    ---------
    self : OCBoundary class object containing OCB file data

    Attributes
    -----------
    filename : (str or NoneType)
        OCBoundary filename or None, if problem loading default
    records : (int)
        Number of OCB records (default=0)
    rec_ind : (int)
        Current OCB record index (default=0; initialised=-1)
    dtime : (numpy.ndarray or NoneType)
        Numpy array of OCB datetimes (default=None)
    num_sectors : (numpy.ndarray or NoneType)
        Numpy array of int indicating number of MLT sectors used to find
        the OCB for each record (default=None)
    phi_cent : (numpy.ndarray or NoneType)
        Numpy array of floats that give the angle from AACGM midnight
        of the OCB pole in degrees (default=None)
    r_cent : (numpy.ndarray or NoneType)
        Numpy array of floats that give the AACGM co-latitude of the OCB
        pole in degrees (default=None)
    r : (numpy.ndarray or NoneType)
        Numpy array of floats that give the radius of the OCBoundary
        in degrees (default=None)
    r_err : (numpy.ndarray or NoneType)
        Numpy array of floats that give the error of the OCBoundary
        radius in degrees (default=None)
    area : (numpy.ndarray or NoneType)
       Numpy array of floats that give the area of the circle defined by
       the OCBoundary in degrees (default=None)

    Functions
    ----------
    load : Load the data from the OCB file specified by self.filename
    get_next_good_ocb_ind : Cycle to the next good OCB index
    normal_coord : Calculate the OCB coordinates of an AACGM location
    """

    def __init__(self, filename=None):
        """Object containing OCB data

        Parameters
        ----------
        filename : (str or NoneType)
            file containing OCB data
        """
        import ocbpy

        if filename is None:
            self.filename = None
        elif not isinstance(filename, str):
            logging.warning("OCB file is not a string [{:s}]".format(filename))
            self.filename = None
        elif not ocbpy.instruments.test_file(filename):
            logging.warning("cannot open OCB file [{:s}]".format(filename))
            self.filename = None
        else:
            self.filename = filename

        if self.filename is None:
            ocb_dir = ocbpy.__file__.split("/")
            self.filename = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                              ocbpy.__default_file__)
            if not ocbpy.instruments.test_file(self.filename):
                logging.warning("problem with default OC Boundary file")
                self.filename = None

        self.records = 0
        self.rec_ind = 0
        self.dtime = None
        self.num_sectors = None
        self.phi_cent = None
        self.r_cent = None
        self.r = None
        self.r_err = None
        self.area = None

        if self.filename is not None:
            self.load()

        return

    def __repr__(self):
        """ Provide readable representation of the OCBoundary object
        """
    
        if self.filename is None:
            out = "No Open-Closed Boundary file specified\n"
        else:
            out = "Open-Closed Boundary file: {:s}\n\n".format(self.filename)

            if self.records == 0:
                out = "{:s}No data loaded\n".format(out)
            else:
                out = "{:s}{:d} records from {:}".format(out, self.records,
                                                         self.dtime[0])
                out = "{:s} to {:}\n\n".format(out, self.dtime[-1])

                irep = sorted(set([0, 1, self.records - 2, self.records - 1]))
                while irep[0] < 0:
                    irep.pop(0)

                head = "YYYY-MM-DD HH:MM:SS NumSectors Phi_Centre R_Centre R "
                out = "{:s}{:s} R_Err Area\n{:-<77s}\n".format(out, head, "")
                for i in irep:
                    out = "{:s}{:} {:d} ".format(out, self.dtime[i],
                                                 self.num_sectors[i])
                    out = "{:s}{:.2f} {:.2f} ".format(out, self.phi_cent[i],
                                                      self.r_cent[i])
                    out = "{:s}{:.2f} {:.2f} {:.4g}\n".format(out, self.r[i],
                                                              self.r_err[i],
                                                              self.area[i])

        return out

    def __str__(self):
        """ Provide readable representation of the OCBoundary object
        """

        out = self.__repr__()
        return out

    def load(self, ocb_cols="YEAR SOY NB PHICENT RCENT R A R_ERR", hlines=0):
        """Load the data from the specified Open-Closed Boundary file

        Parameters
        -----------
        ocb_cols : (str)
            String specifying format of OCB file.  All but the first two 
            columns must be included in the string, additional data values will
            be ignored.  If "YEAR SOY" aren't used, expects
            "DATE TIME" in "YYYY-MM-DD HH:MM:SS" format.
            (default="YEAR SOY NB PHICENT RCENT R A R_ERR")
        hlines : (int)
            Number of header lines preceeding data in the OCB file (default=0)

        Returns
        --------
        self
        """
        import datetime as dt

        cols = ocb_cols.split()
        dflag = -1
        ldtype = [(k,float) if k != "NB" else (k,int) for k in cols]
        
        if "SOY" in cols and "YEAR" in cols:
            dflag = 0
            ldtype[cols.index('YEAR')] = ('YEAR',int)
        elif "DATE" in cols and "TIME" in cols:
            dflag = 1
            ldtype[cols.index('DATE')] = ('DATE','|S50')
            ldtype[cols.index('TIME')] = ('TIME','|S50')

        if dflag < 0:
            logging.error("missing time columns in [{:s}]".format(ocb_cols))
            return

        # Read the OCB data
        odata = np.genfromtxt(self.filename, skip_header=hlines, dtype=ldtype)

        # Load the data into the OCBoundary object
        self.records = odata.shape[0]
        self.rec_ind = -1

        dt_list = list()
        for i in range(self.records):
            try:
                if dflag == 0:
                    dtime = year_soy_to_datetime(odata['YEAR'][i],
                                                 odata['SOY'][i])

                else:
                    stime = "{:} {:}".format(odata['DATE'][i], odata['TIME'][i])
                    dtime = dt.datetime.strptime(stime, "%Y-%m-%d %H:%M:%S")

                dt_list.append(dtime)
            except ValueError as v:
                if(len(v.args) > 0 and
                   v.args[0].startswith('unconverted data remains: ')):
                    vsplit = v.args[0].split(" ")
                    dtime = dt.datetime.strptime(dtstring[:-(len(vsplit[-1]))],
                                                 "%Y-%m-%d %H:%M:%S")
                else:
                    raise v

        self.dtime = np.array(dt_list)
        self.num_sectors = odata['NB']
        self.phi_cent = odata['PHICENT']
        self.r_cent = odata['RCENT']
        self.r = odata['R']
        self.r_err = odata['R_ERR']
        self.area = odata['A']

        return

    def get_next_good_ocb_ind(self, min_sectors=7, rcent_dev=8.0, max_r=23.0,
                              min_r=10.0):
        """read in the next usuable OCB record from the data file

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
            # Evaluate current boundary for quality
            if(self.num_sectors[self.rec_ind] >= min_sectors and
               self.r_cent[self.rec_ind] <= rcent_dev and
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

        xc = self.r_cent[self.rec_ind] * \
             np.cos(np.radians(self.phi_cent[self.rec_ind]))
        yc = self.r_cent[self.rec_ind] * \
             np.sin(np.radians(self.phi_cent[self.rec_ind]))

        xp = (90.0 - aacgm_lat) * np.cos(np.radians(aacgm_mlt * 15.0))
        yp = (90.0 - aacgm_lat) * np.sin(np.radians(aacgm_mlt * 15.0))

        xn = (xp - xc) * (16.0 / self.r[self.rec_ind])
        yn = (yp - yc) * (16.0 / self.r[self.rec_ind])

        ocb_lat = 90.0 - np.sqrt(xn**2 + yn**2)
        ocb_mlt = np.degrees(np.arctan2(yn, xn)) / 15.0

        if ocb_mlt < 0.0:
            ocb_mlt += 24.0

        return ocb_lat, ocb_mlt

def year_soy_to_datetime(yyyy, soy):
    """Converts year and soy to datetime

    Parameters
    -----------
    yyyy : (int)
        4 digit year
    soy : (float)
        seconds of year

    Returns
    ---------
    dtime : (dt.datetime)
        datetime object
    """
    import datetime as dt
                
    # Calcuate doy, hour, min, seconds of day
    ss = soy / 86400.0
    ddd = np.floor(ss)

    ss = (soy - ddd * 86400.0) / 3600.0
    hh = np.floor(ss)

    ss = (soy - ddd * 86400.0 - hh * 3600.0) / 60.0
    mm = np.floor(ss)

    ss = soy - ddd * 86400.0 - hh * 3600.0 - mm * 60.0
    
    # Define format
    stime = "{:d}-{:.0f}-{:.0f}-{:.0f}-{:.0f}".format(yyyy, ddd + 1, hh, mm, ss)

    # Convert to datetime
    dtime = dt.datetime.strptime(stime, "%Y-%j-%H-%M-%S")

    return dtime

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
         Maximum radius for open-closed field line boundary in degrees.
        (default=23.0)
    min_r : (float)
        Minimum radius for open-closed field line boundary in degrees
        (default=10.0)

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
                                  max_r=max_r, min_r=min_r)
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
                                      min_r=min_r)
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
    return idat
