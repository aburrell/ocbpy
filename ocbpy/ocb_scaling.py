#!/usr/bin/env python# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
"""Scale data affected by magnetic field direction or electric field

Routines
-------------------------------------------------------------------------------
normal_evar(evar, unscaled_r, scaled_r)
    Normalise a variable proportaional to the electric field (such as velocity)
normal_curl_evar(curl_evar, unscaled_r, scaled_r)
    Normalise a variable proportional to the curl of the electric field (such
    as vorticity)

Classes
-------------------------------------------------------------------------------
VectorData(object)
    Holds vector data in AACGM N-E-Z coordinates along with location
    information.  Converts vector from AACGM to OCB coordinates.

Moduleauthor
-------------------------------------------------------------------------------
Angeline G. Burrell (AGB), 12 May 2017, University of Texas, Dallas (UTDallas)

References
-------------------------------------------------------------------------------
Chisham, G. (2017), A new methodology for the development of high-latitude
 ionospheric climatologies and empirical models, Journal of Geophysical
 Research: Space Physics, 122, doi:10.1002/2016JA023235.
"""
import logbook as logging
import numpy as np

def normal_evar(evar, unscaled_r, scaled_r):
    """ Normalise a variable proportional to the electric field

    Parameters
    -----------
    evar : (float)
        Variable related to electric field (e.g. velocity)
    unscaled_r : (float)
        Radius of polar cap in degrees
    scaled_r : (float)
        Radius of normalised OCB polar cap in degrees

    Returns
    --------
    nvar : (float)
        Normalised variable

    Notes
    -------
    Assumes that the cross polar cap potential is fixed across the polar cap
    regardless of the radius of the Open Closed field line Boundary.  This is
    commonly assumed when looking at statistical patterns that control the IMF
    (which accounts for dayside reconnection) and assume that the nightside
    reconnection influence is averaged out over the averaged period.

    References
    -----------
    Chisham, G. (2017), A new methodology for the development of high‐latitude
    ionospheric climatologies and empirical models, Journal of Geophysical
    Research: Space Physics, doi:10.1002/2016JA023235.
    """

    nvar = evar * unscaled_r / scaled_r

    return nvar

def normal_curl_evar(curl_evar, unscaled_r, scaled_r):
    """ Normalise a variable proportional to the curl of the electric field

    Parameters
    -----------
    curl_evar : (float)
        Variable related to electric field (e.g. vorticity)
    unscaled_r : (float)
        Radius of polar cap in degrees
    scaled_r : (float)
        Radius of normalised OCB polar cap in degrees

    Returns
    --------
    nvar : (float)
        Normalised variable

    Notes
    -------
    Assumes that the cross polar cap potential is fixed across the polar cap
    regardless of the radius of the Open Closed field line Boundary.  This is
    commonly assumed when looking at statistical patterns that control the IMF
    (which accounts for dayside reconnection) and assume that the nightside
    reconnection influence is averaged out over the averaged period.

    References
    -----------
    Chisham, G. (2017), A new methodology for the development of high‐latitude
    ionospheric climatologies and empirical models, Journal of Geophysical
    Research: Space Physics, doi:10.1002/2016JA023235.
    """

    nvar = curl_evar * (unscaled_r / scaled_r)**2

    return nvar

class VectorData(object):
    """ Object containing a vector data point

    Parameters
    -----------
    dat_ind : (int)
        Data index (zero offset)
    ocb_ind : (int)
        OCBoundary record index matched to this data index (zero offset)
    aacgm_lat : (float)
        Vector AACGM latitude (degrees)
    aacgm_mlt : (float)
        Vector AACGM MLT (hours)
    ocb_lat : (float)
        Vector OCB latitude (degrees) (default=np.nan)
    ocb_mlt : (float)
        Vector OCB MLT (hours) (default=np.nan)
    aacgm_n : (float)
        AACGM North pointing vector (positive towards North) (default=0.0)
    aacgm_e : (float)
        AACGM East pointing vector (completes right-handed coordinate system
        (default = 0.0)
    aacgm_z : (float)
        AACGM Vertical pointing vector (positive down) (default=0.0)
    aacgm_mag : (float)
        Vector magnitude (default=np.nan)
    scale_func : (function)
        Function for scaling AACGM magnitude with arguements:
        [measurement value, mesurement AACGM latitude (degrees),
        mesurement OCB latitude (degrees)]
        (default=None)
    dat_name : (str)
        Data name (default=None)
    dat_units : (str)
        Data units (default=None)

    Attributes
    ------------
    dat_name : (str or NoneType)
        Name of data
    dat_units : (str or NoneType)
        Units of data
    dat_ind : (int)
        Vector data index in external data array
    ocb_ind : (int)
        OCBoundary rec_ind value that matches dat_ind
    unscaled_r : (float)
        Radius of polar cap in degrees
    scaled_r : (float)
        Radius of normalised OCB polar cap in degrees
    aacgm_n : (float)
        AACGM north component of data vector (default=0.0)
    aacgm_e : (float)
        AACGM east component of data vector (default=0.0)
    aacgm_z : (float)
        AACGM vertical component of data vector (default=0.0)
    aacgm_mag : (float)
        Magnitude of data vector in AACGM coordinates (default=np.nan)
    aacgm_lat : (float)
        AACGM latitude of data vector in degrees
    aacgm_mlt : (float)
        AACGM MLT of data vector in hours
    ocb_n : (float)
        OCB north component of data vector (default=np.nan)
    ocb_e : (float)
        OCB east component of data vector (default=np.nan)
    ocb_z : (float)
        OCB vertical component of data vector (default=np.nan)
    ocb_mag : (float)
        OCB magnitude of data vector (default=np.nan)
    ocb_lat : (float)
        OCB latitude of data vector in degrees (default=np.nan)
    ocb_mlt : (float)
        OCB MLT of data vector in hours (default=np.nan)
    ocb_quad : (int)
        AACGM quadrant of OCB pole (default=0)
    vec_quad : (int)
        AACGM quadrant of Vector (default=0)
    pole_angle : (float)
        Angle at vector location appended by AACGM and OCB poles in degrees
        (default=np.nan)
    aacgm_naz : (float)
        AACGM north azimuth of data vector in degrees (default=np.nan)
    ocb_aacgm_lat : (float)
        AACGM latitude of OCB pole in degrees (default=np.nan)
    ocb_aacgm_mlt : (float)
        AACGM MLT of OCB pole in hours (default=np.nan)
    scale_func : (function or NoneType)
        Funciton that scales the magnitude of the data vector from AACGM
        polar cap coverage to OCB polar cap coverage

    Methods
    -----------
    set_ocb(ocb, scale_func=None)
        Set the ocb coordinates and vector values
    define_quadrants()
        Define the OCB pole and vector AACGM quadrant
    scale_vector()
        Scale the data vector into OCB coordinates
    calc_ocb_polar_angle()
        calculate the OCB north azimuth angle
    calc_ocb_vec_sign(north=False, east=False, quads=dict())
        calculate the signs of the OCB scaled vector components
    calc_vec_pole_angle(angular_res=1.0e-3)
        calculate the vector angle of the vector-poles triangle
    """

    def __init__(self, dat_ind, ocb_ind, aacgm_lat, aacgm_mlt, ocb_lat=np.nan,
                 ocb_mlt=np.nan, aacgm_n=0.0, aacgm_e=0.0, aacgm_z=0.0,
                 aacgm_mag=np.nan, dat_name=None, dat_units=None,
                 scale_func=None):
        """ Initialize VectorData object

        Parameters
        -----------
        dat_ind : (int)
            Data index (zero offset)
        ocb_ind : (int)
            OCBoundary record index matched to this data index (zero offset)
        aacgm_lat : (float)
            Vector AACGM latitude (degrees)
        aacgm_mlt : (float)
            Vector AACGM MLT (hours)
        ocb_lat : (float)
            Vector OCB latitude (degrees) (default=np.nan)
        ocb_mlt : (float)
            Vector OCB MLT (hours) (default=np.nan)
        aacgm_n : (float)
            AACGM North pointing vector (positive towards North) (default=0.0)
        aacgm_e : (float)
            AACGM East pointing vector (completes right-handed coordinate system
            (default = 0.0)
        aacgm_z : (float)
            AACGM Vertical pointing vector (positive down) (default=0.0)
        aacgm_mag : (float)
            Vector magnitude (default = np.nan)
        dat_name : (str)
            Data name (default=None)
        dat_units : (str)
            Data units (default=None)
        scale_func : (function)
            Function for scaling AACGM magnitude with arguements:
            [measurement value, mesurement AACGM latitude (degrees),
            mesurement OCB latitude (degrees)]
            Not necessary if no magnitude scaling is needed. (default=None)

        Returns
        --------
            self : Initialised VectorData class object by setting AACGM values
        """
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

        if np.isnan(aacgm_mag):
            self.aacgm_mag = np.sqrt(aacgm_n**2 + aacgm_e**2 + aacgm_z**2)
        else:
            self.aacgm_mag = aacgm_mag

        # Assign the OCB vector default values and location
        self.ocb_n = np.nan
        self.ocb_e = np.nan
        self.ocb_z = np.nan
        self.ocb_mag = np.nan
        self.ocb_lat = ocb_lat
        self.ocb_mlt = ocb_mlt

        # Assign the default pole locations, relative angles, and quadrants
        self.ocb_quad = 0
        self.vec_quad = 0
        self.pole_angle = np.nan
        self.aacgm_naz = np.nan
        self.ocb_aacgm_lat = np.nan
        self.ocb_aacgm_mlt = np.nan

        # Assign the vector scaling function
        self.scale_func = scale_func

        return

    def __repr__(self):
        """ Provide readable representation of the DataVector object
        """
        
        out = "Vector data:"
        if self.dat_name is not None:
            out = "{:s} {:s}".format(out, self.dat_name)
        if self.dat_units is not None:
            out = "{:s} ({:s})".format(out, self.dat_units)
        out = "{:s}\nData Index {:d}\tOCB Index {:d}\n".format(out,
                                                               self.dat_ind,
                                                               self.ocb_ind)
        out = "{:s}-------------------------------------------\n".format(out)
        out = "{:s}Location: [Mag. Lat. (degrees), MLT (hours)]\n".format(out)
        out = "{:s}   AACGM: [{:.3f}, {:.3f}]\n".format(out, self.aacgm_lat,
                                                       self.aacgm_mlt)
        if not np.isnan(self.ocb_lat) and not np.isnan(self.ocb_mlt):
            out = "{:s}     OCB: [{:.3f}, {:.3f}]\n".format(out, self.ocb_lat,
                                                           self.ocb_mlt)

        out = "\n{:s}-------------------------------------------\n".format(out)
        out = "{:s}Value: Magnitude [N, E, Z]\n".format(out)
        out = "{:s}AACGM: {:.3g} [".format(out, self.aacgm_mag)
        out = "{:s}{:.3g}, {:.3g}, {:.3g}]\n".format(out, self.aacgm_n,
                                                     self.aacgm_e, self.aacgm_z)
        if not np.isnan(self.ocb_mag):
            out = "{:s}  OCB: {:.3g} [".format(out, self.ocb_mag)
            out = "{:s}{:.3g}, {:.3g}, {:.3g}]\n".format(out, self.ocb_n,
                                                         self.ocb_e, self.ocb_z)

        out = "\n{:s}-------------------------------------------\n".format(out)
        if self.scale_func is None:
            out = "{:s}No magnitude scaling function provided\n".format(out)
        else:
            out = "{:s}Scaling function: ".format(out)
            out = "{:s}{:s}\n".format(out, self.scale_func.__name__)

        return out

    def __str__(self):
        """ Provide readable representation of the DataVector object
        """
        
        out = self.__repr__()
        return out

    def set_ocb(self, ocb, scale_func=None):
        """ Set the OCBoundary values for this data point

        Parameters
        ------------
        ocb : (OCBoundary)
            Open Closed Boundary class object
        scale_func : (function)
            Function for scaling AACGM magnitude with arguements:
            [measurement value, mesurement AACGM latitude (degrees),
            mesurement OCB latitude (degrees)]
            Not necessary if defined earlier or no scaling is needed.
            (default=None)

        Updates
        ---------
        self.unscaled_r : (float)
            Radius of polar cap in degrees
        self.scaled_r : (float)
            Radius of normalised OCB polar cap in degrees
        self.ocb_n : (float)
            Vector OCB North component
        self.ocb_e : (float)
            Vector OCB East component
        self.ocb_z : (float)
            Vector OCB vertical component (positive downward)
        self.ocb_mag : (float)
            Vector OCB magnitude
        self.ocb_lat : (float)
            Vector OCB latitude, if not updated already (degrees)
        self.ocb_mlt : (float)
            Vector OCB MLT, if not updated already (hours)
        self.ocb_quad : (int)
            OCB pole AACGM quadrant
        self.vec_quad : (int)
            Vector AACGM quadrant
        self.pole_angle : (float)
            Angle at the vector in the triangle formed by the poles and vector
            (degrees)
        self.aacgm_naz : (float)
            AACGM north azimuth angle (degrees)
        self.ocb_aacgm_lat : (float)
            AACGM latitude of the OCB pole (degrees)
        self.ocb_aacgm_mlt : (float)
            AACGM MLT of the OCB pole (hours)
        self.scale_func : (function)
            Function for scaling AACGM magnitude with arguements:
            [measurement value, unscaled polar cap radius (degrees),
            scaled polar cap radius (degrees)]
            Not necessary if defined earlier or if no scaling is needed.
        """

        # Set the AACGM coordinates of the OCB pole
        self.unscaled_r = ocb.r[self.ocb_ind]
        self.scaled_r = 90.0 - abs(ocb.boundary_lat)
        self.ocb_aacgm_mlt = ocb.phi_cent[self.ocb_ind] / 15.0
        self.ocb_aacgm_lat = 90.0 - ocb.r_cent[self.ocb_ind]

        # If the OCB vector coordinates weren't included in the initial info,
        # update them here
        if np.isnan(self.ocb_lat) or np.isnan(self.ocb_mlt):
            self.ocb_lat, self.ocb_mlt = ocb.normal_coord(self.aacgm_lat,
                                                          self.aacgm_mlt)

        # Get the angle at the data vector appended by the AACGM and OCB poles
        self.calc_vec_pole_angle()

        # Set the OCB and Vector quadrants
        if not np.isnan(self.pole_angle):
            self.define_quadrants()
        
            # Set the scaling function
            if self.scale_func is None:
                if scale_func is None:
                    # This is not necessarily a bad thing, if the value does not
                    # need to be scaled.
                    logging.info("no scaling function provided")
                else:
                    self.scale_func = scale_func

            # Assign the OCB vector default values and location.  Will also
            # update the AACGM north azimuth of the vector.
            self.scale_vector()
        return

    def define_quadrants(self):
        """ Determine which quadrants (in AACGM coordinates) the OCB pole
        and data vector lie in

        Requires
        ---------
        self.ocb_aacgm_mlt : (float)
            OCB pole MLT in AACGM coordinates in hours
        self.aacgm_mlt : (float)
            Vector AACGM MLT in hours
        self.pole_angle : (float)
            vector angle in poles-vector triangle in degrees

        Updates
        --------
        self.ocb_quad : (int)
            OCB pole quadrant
        self.vec_quad : (int)
            Vector quadrant

        Notes
        ------
        North (N) and East (E) are defined by the AACGM directions centred on
        the data vector location, assuming vertical is positive downwards
        Quadrants: 1 [N, E]; 2 [N, W]; 3 [S, W]; 4 [S, E]
        """

        # Test input
        assert(not np.isnan(self.ocb_aacgm_mlt)), \
            logging.error("OCB pole location required")
        assert(not np.isnan(self.aacgm_mlt)), \
            logging.error("Vector AACGM location required")
        assert(not np.isnan(self.pole_angle)), \
            logging.error("vector angle in poles-vector triangle required")

        # Determine where the OCB pole is relative to the data vector
        ocb_adj_mlt = self.ocb_aacgm_mlt - self.aacgm_mlt
        while ocb_adj_mlt < 0.0:
            ocb_adj_mlt += 24.0
        if abs(ocb_adj_mlt) >= 24.0:
            ocb_adj_mlt -= 24.0 * np.sign(ocb_adj_mlt)

        if self.pole_angle < 90.0:
            # OCB pole lies in quadrant 1 or 2
            self.ocb_quad = 1 if ocb_adj_mlt < 12.0 else 2
        else:
            # OCB poles lies in quadrant 3 or 4
            self.ocb_quad = 3 if ocb_adj_mlt < 24.0 else 4
       
        # Now determine which quadrant the vector is pointed into
        if self.aacgm_n >= 0.0:
            self.vec_quad = 1 if self.aacgm_e >= 0.0 else 2
        else:
            self.vec_quad = 4 if self.aacgm_e >= 0.0 else 3

        return
        
    def scale_vector(self):
        """ Normalise a variable proportional to the curl of the electric field.

        Requires
        ---------
        self.ocb_lat : (float)
            OCB latitude in degrees
        self.ocb_mlt : (float)
            OCB MLT in hours
        self.ocb_aacgm_mlt : (float)
            OCB pole MLT in AACGM coordinates in hours
        self.pole_angle : (float)
            vector angle in poles-vector triangle

        Updates
        --------
        ocb_n : (float)
            OCB scaled north component
        ocb_e : (float)
            OCB scaled east component
        ocb_z : (float)
            OCB scaled vertical component
        ocb_mag : (float)
            OCB scaled magnitude
        """

        # Test input
        assert(not np.isnan(self.ocb_lat) and not np.isnan(self.ocb_mlt)), \
            logging.error("OCB coordinates required")
        assert(not np.isnan(self.ocb_aacgm_mlt)), \
            logging.error("OCB pole location required")
        assert(not np.isnan(self.pole_angle)), \
            logging.error("vector angle in poles-vector triangle required")

        # Scale vertical component
        if self.scale_func is None or self.aacgm_z != 0.0:
            self.ocb_z = self.aacgm_z
        else:
            self.ocb_z = self.scale_func(self.aacgm_z, self.unscaled_r,
                                         self.scaled_r)

        if self.aacgm_n == 0.0 and self.aacgm_e == 0.0:
            # There's no magnitude, so nothing to adjust
            self.ocb_n = 0.0
            self.ocb_e = 0.0
        elif self.pole_angle == 0.0 or self.pole_angle == 180.0:
            # The measurement is aligned with the AACGM and OCB poles
            if self.scale_func is None:
                self.ocb_n = self.aacgm_n
                self.ocb_e = self.aacgm_e
            else:
                self.ocb_n = self.scale_func(self.aacgm_n, self.unscaled_r,
                                             self.scaled_r)
                self.ocb_e = self.scale_func(self.aacgm_e, self.unscaled_r,
                                             self.scaled_r)
        
            if self.pole_angle == 0.0 and self.aacgm_lat >= self.ocb_aacgm_lat:
                # The measurement is on or between the poles
                self.ocb_n *= -1.0
                self.ocb_e *= -1.0
        else:
            # If not defined, get the OCB and vector quadrants
            if(self.ocb_quad == 0 or self.vec_quad == 0):
                self.define_quadrants()

            if(self.ocb_quad == 0 or self.vec_quad == 0):
                logging.error("unable to define OCB and vector quadrants")
                return
    
            # Get the unscaled 2D vector magnitude
            vmag = np.sqrt(self.aacgm_n**2 + self.aacgm_e**2)

            # Calculate the AACGM north azimuth in degrees
            self.aacgm_naz = np.degrees(np.arccos(self.aacgm_n / vmag))

            # Get the OCB north azimuth in radians
            ocb_angle = np.radians(self.calc_ocb_polar_angle())

            # Get the sign of the North and East components
            vsigns = self.calc_ocb_vec_sign(north=True, east=True)
        
            # Scale the vector along the OCB north and account for
            # any changes associated with adjusting the size of the polar cap
            if self.scale_func is not None:
                vmag = self.scale_func(vmag, self.unscaled_r, self.scaled_r)

            self.ocb_n = vsigns['north'] * vmag * np.cos(ocb_angle)
            self.ocb_e = vsigns['east'] * vmag * np.sin(ocb_angle)

        # Calculate the scaled OCB vector magnitude
        self.ocb_mag = np.sqrt(self.ocb_n**2 + self.ocb_e**2 + self.ocb_z**2)

        return

    def calc_ocb_polar_angle(self):
        """ Calculate the OCB north azimuth angle

        Requires
        ---------
        self.ocb_quad : (int)
            OCB quadrant
        self.vec_quad : (int)
            Vector quadrant
        self.aacgm_naz : (float)
            AACGM polar angle
        self.pole_angle : (float)
            Vector angle between AACGM pole, vector origin, and OCB pole

        Returns
        --------
        ocb_naz : (float)
            Angle between measurement vector and OCB pole in degrees
        """
        quad_range = np.arange(1, 5)

        assert self.ocb_quad in quad_range, \
            logging.error("OCB quadrant undefined")
        assert self.vec_quad in quad_range, \
            logging.error("Vector quadrant undefined")
        assert not np.isnan(self.aacgm_naz), \
            logging.error("AACGM polar angle undefined")
        assert not np.isnan(self.pole_angle), \
            logging.error("Vector angle undefined")

        # Initialise the output and set the quadrant dictionary
        ocb_naz = np.nan
        quads = {o:{v:True if self.ocb_quad == o and self.vec_quad == v
                    else False for v in quad_range} for o in quad_range}

        # Calculate OCB polar angle based on quadrants and other angles
        if(((quads[2][4] or quads[2][2]) and self.aacgm_naz > self.pole_angle)
           or (self.aacgm_naz > self.pole_angle and quads[1][1]) or
           (quads[1][4] and self.aacgm_naz <= self.pole_angle + 90.0)):
            ocb_naz = self.aacgm_naz - self.pole_angle
        elif((self.aacgm_naz <= self.pole_angle and quads[1][1]) or
             (self.aacgm_naz <= self.pole_angle and
              (quads[2][4] or quads[2][2])) or
             (self.aacgm_naz > self.pole_angle - 90.0 and
              (quads[4][1] or quads[4][3] or quads[3][4] or quads[3][2]))):
            ocb_naz = self.pole_angle - self.aacgm_naz
        elif(self.aacgm_naz <= 90.0 - self.pole_angle and
             (quads[1][2] or quads[2][1] or quads[2][3])):
            ocb_naz = self.aacgm_naz + self.pole_angle
        elif((self.aacgm_naz > 90.0 - self.pole_angle and
              (quads[1][2] or quads[2][1] or quads[2][3])) or
             ((quads[4][4] or quads[4][2] or quads[3][1] or quads[3][3] or
               quads[1][3]) and self.aacgm_naz <= 180.0 - self.pole_angle)):
            ocb_naz = 180.0 - self.aacgm_naz - self.pole_angle
        elif(((quads[3][1] or quads[3][3] or quads[4][4] or quads [4][2] or
               quads[1][3]) and self.aacgm_naz > 180.0 - self.pole_angle) or
             (quads[1][4] and self.aacgm_naz > self.pole_angle + 90.0)):
            ocb_naz = self.aacgm_naz - 180.0 + self.pole_angle
        elif(self.aacgm_naz <= self.pole_angle - 90.0 and
             (quads[3][4] or quads[3][2] or quads[4][1] or quads[4][3])):
            ocb_naz = 180.0 - self.pole_angle + self.aacgm_naz

        return ocb_naz
    
    def calc_ocb_vec_sign(self, north=False, east=False, quads=dict()):
        """ Get the sign of the North and East components

        Parameters
        ------------
        north : (boolian)
            Get the sign of the north component (default=False)
        east : (boolian)
            Get the sign of the east component (default=False)
        quads : (dictionary)
            Dictionary of boolian values for OCB and Vector quadrants
            (default=dict())

        Requires
        ----------
        self.ocb_quad : (int)
            OCB pole quadrant
        self.vec_quad : (int)
            Vector quadrant
        self.aacgm_naz : (float)
            AACGM polar angle in degrees
        self.pole_angle : (float)
            Vector angle in degrees

        Returns
        ---------
        vsigns : (dict)
            Dictionary with keys 'north' and 'east' containing the desired signs
        """
        quad_range = np.arange(1, 5)

        # Test input
        assert north or east, logging.warning("must set at least one direction")
        assert self.ocb_quad in quad_range, \
            logging.error("OCB quadrant undefined")
        assert self.vec_quad in quad_range, \
            logging.error("Vector quadrant undefined")
        assert not np.isnan(self.aacgm_naz), \
            logging.error("AACGM polar angle undefined")
        assert not np.isnan(self.pole_angle), \
            logging.error("Vector angle undefined")

        # Initialise output
        vsigns = {"north":0, "east":0}

        # If necessary, initialise quadrant dictionary
        if not np.all(quads.keys() == quad_range):
            quads = {o:{v:True if self.ocb_quad == o and self.vec_quad == v
                        else False for v in quad_range} for o in quad_range}

        if north:
            pole_minus = self.pole_angle - 90.0
            minus_pole = 90.0 - self.pole_angle
            pole_plus = self.pole_angle + 90.0
            
            if(quads[1][1] or quads[2][2] or quads[3][3] or quads[4][4] or
               (quads[1][4] and self.aacgm_naz <= pole_plus) or
               (quads[1][2] and self.aacgm_naz > minus_pole) or
               (quads[2][1] and self.aacgm_naz <= minus_pole) or 
               ((quads[3][4] or quads[4][3]) and self.aacgm_naz <= pole_minus)
               or ((quads[3][2] or quads[4][1]) and self.aacgm_naz > pole_minus)
               or (quads[2][3] and self.aacgm_naz > minus_pole)):
                vsigns["north"] = 1
            elif((quads[1][2] and self.aacgm_naz > minus_pole) or
                 (quads[1][4] and self.aacgm_naz > pole_plus) or
                 (quads[2][1] and self.aacgm_naz > minus_pole) or
                 ((quads[4][1] or quads[3][2]) and self.aacgm_naz <= pole_minus)
                 or (quads[2][3] and self.aacgm_naz <= minus_pole) or
                 ((quads[4][3] or quads[3][4]) and self.aacgm_naz > pole_minus)
                 or quads[1][3] or quads[2][4] or quads[3][1] or quads[4][2]):
                vsigns["north"] = -1

        if east:
            minus_pole = 180.0 - self.pole_angle

            if(quads[1][4] or quads[2][1] or quads[3][2] or quads[4][3] or
               (quads[1][1] and self.aacgm_naz > self.pole_angle) or
               (quads[1][3] and self.aacgm_naz > minus_pole) or
               ((quads[4][4] or quads[3][1]) and self.aacgm_naz <= minus_pole)
               or (quads[2][4] and self.aacgm_naz > self.pole_angle) or
               ((quads[4][2] or quads[3][3]) and self.aacgm_naz > minus_pole)
               or (quads[2][2] and self.aacgm_naz <= self.pole_angle)):
                vsigns["east"] = 1
            elif(quads[1][2] or quads[2][3] or quads[3][4] or quads[4][1] or
                 ((quads[4][4] or quads[3][1]) and self.aacgm_naz > minus_pole)
                 or (quads[2][2] and self.aacgm_naz > self.pole_angle) or
                 ((quads[4][2] or quads[3][3])and self.aacgm_naz <= minus_pole)
                 or (quads[1][3] and self.aacgm_naz <= minus_pole) or
                 ((quads[1][1] or quads[2][4])
                  and self.aacgm_naz <= self.pole_angle)):
                vsigns["east"] = -1

        return vsigns

    def calc_vec_pole_angle(self):
        """calculates the angle between the AACGM pole, a measurement, and the
        OCB pole using spherical triginometry

        Requires
        ---------
        self.aacgm_mlt : (float)
            AACGM MLT of vector origin in hours
        self.aacgm_lat : (float)
            AACGM latitude of vector origin in degrees
        self.ocb_aacgm_mlt : (float)
            AACGM MLT of the OCB pole in hours
        self.ocb_aacgm_lat : (float)
            AACGM latitude of the OCB pole in degrees

        Updates
        --------
        self.pole_angle : (float)
            Angle in degrees between AACGM north, a measurement, and OCB north
        """
        # Test input
        assert(not np.isnan(self.aacgm_mlt)), \
            logging.error("AACGM MLT of Vector undefinded")
        assert(not np.isnan(self.ocb_aacgm_mlt)), \
            logging.error("AACGM MLT of OCB pole is undefined")
        assert(not np.isnan(self.ocb_aacgm_lat)), \
            logging.error("AACGM latitude of OCB pole is undefined")
        assert(not np.isnan(self.aacgm_lat)), \
            logging.error("AACGM latitude of Vector undefined")

        # Convert the AACGM MLT of the observation and OCB pole to radians,
        # then calculate the difference between them.
        del_long = (self.ocb_aacgm_mlt - self.aacgm_mlt) * np.pi / 12.0

        if del_long < 0.0:
            del_long += 2.0 * np.pi

        if del_long == 0.0:
            self.pole_angle = 0.0
            return

        if del_long == np.pi:
            self.pole_angle = 180.0
            return

        # Find the distance in radians between the two poles
        hemisphere = np.sign(self.ocb_aacgm_lat)
        rad_pole = hemisphere * np.pi * 0.5
        del_pole = hemisphere * (rad_pole - np.radians(self.ocb_aacgm_lat))

        # Get the distance in radians between the AACGM pole and the data point
        del_vect = hemisphere * (rad_pole - np.radians(self.aacgm_lat))

        # Use the law of haversines, which goes to the spherical trigonometric
        # cosine rule for sides at large angles, but is more robust at small
        # angles, to find the length of the last side of the spherical triangle.
        del_ocb = archav(hav(del_pole - del_vect) +
                         np.sin(del_pole) * np.sin(del_vect) * hav(del_long))

        # Again use law of haversines, this time to find the polar angle
        hav_pole_angle = (hav(del_pole) - hav(del_vect - del_ocb)) \
                         / (np.sin(del_vect) * np.sin(del_ocb))

        self.pole_angle = np.degrees(archav(hav_pole_angle))

        return

def hav(alpha):
    """ Formula for haversine

    Parameters
    ----------
    alpha : (float)
        Angle in radians

    Returns
    --------
    hav_alpha : (float)
        Haversine of alpha, equal to the square of the sine of half-alpha
    """
    hav_alpha = np.sin(alpha * 0.5)**2

    return hav_alpha

def archav(hav):
    """ Formula for the inverse haversine

    Parameters
    -----------
    hav : (float)
        Haversine of an angle

    Returns
    ---------
    alpha : (float)
        Angle in radians
    """
    alpha = 2.0 * np.arcsin(np.sqrt(hav))

    return alpha
