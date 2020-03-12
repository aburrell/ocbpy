#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:47:16 2018

@author: jone
"""

#rotate aacgm/apex vector components into the frame centered at the oval centre location
#code is adapted from ocbpy from gritub

import numpy as np

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

    Notes
    -----
    The input must be positive.  However, any number with a magnitude below
    10-16 will be rounded to zero.        
    """
    
    alpha = np.zeros(len(hav))
    alpha[:] = np.nan
    
    # Propagate NaNs
    nans = np.isnan(hav)
    alpha[nans] = np.nan
    
    # The number is positive, calculate the angle
    positive = (hav>=1.0e-16)
    alpha[positive] = 2.0 * np.arcsin(np.sqrt(hav[positive]))

    # The number is small enough that machine precision may have changed
    # the sign, but it's a single-precission zero
    small = (abs(hav) < 1.0e-16)    
    alpha[small] = 0
    
    # The input is negative
    negs = (hav < -1.0e-16)
    alpha[negs] = np.nan

    return alpha


def calc_vec_pole_angle(mlat, mlt, center_mlat, center_mlt):
    """calculates the angle between the AACGM pole, a measurement, and the
    OCB pole using spherical triginometry
    
    input:
    mlat: position of vector (aacgm/apex) in degrees
    mlt: position of vector (aacgm/apex) in hours
    oval_mlat: apex mlat position of oval center in degrees
    oval_phi: apex mlt position of oval center in degrees [hrs*15]

    return: 
    pole_angle: the angle between the AACGM pole, a measurement, and the
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

    # Convert the AACGM MLT of the observation and OCB pole to radians,
    # then calculate the difference between them.
    
    pole_angle = np.zeros(len(mlat)) - 1
    
    del_long = (center_mlt - mlt) * np.pi / 12.0

    negs = (del_long < 0.0)
    del_long[negs] = del_long[negs] + 2.0 * np.pi

    zeros = (del_long == 0.)
    pole_angle[zeros] = 0.
    
    pis = (del_long == np.pi)
    pole_angle[pis] = 180.
    use = (pole_angle < 0.)


    # Find the distance in radians between the two poles
    hemisphere = np.sign(center_mlat[use])
    rad_pole = hemisphere * np.pi * 0.5
    del_pole = hemisphere * (rad_pole - np.radians(center_mlat[use]))

    # Get the distance in radians between the AACGM pole and the data point
    del_vect = hemisphere * (rad_pole - np.radians(mlat[use]))

    # Use the law of haversines, which goes to the spherical trigonometric
    # cosine rule for sides at large angles, but is more robust at small
    # angles, to find the length of the last side of the spherical triangle.
    del_ocb = archav(hav(del_pole - del_vect) +
                     np.sin(del_pole) * np.sin(del_vect) * hav(del_long[use]))

    # Again use law of haversines, this time to find the polar angle
    hav_pole_angle = (hav(del_pole) - hav(del_vect - del_ocb)) \
                     / (np.sin(del_vect) * np.sin(del_ocb))

    pole_angle[use] = np.degrees(archav(np.abs(hav_pole_angle)))    #-180,180 range. 2020-03-03 Jone added the abs constraint. This line really only give an output on the range 0,180

    return pole_angle

def define_quadrants(oval_mlt, mlt, pole_angle, apex_e, apex_n):
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


    # Determine where the OCB pole is relative to the data vector
    ocb_quad = np.zeros(len(mlt))  #to hold the result
    vec_quad = np.zeros(len(mlt))  #to hold the result
    
    ocb_adj_mlt = oval_mlt - mlt
    
    negs = (ocb_adj_mlt < 0.)
    while sum(negs) >= 1:
        ocb_adj_mlt[negs] = ocb_adj_mlt[negs] + 24.
        negs = (ocb_adj_mlt < 0.)
    large = (ocb_adj_mlt) >= 24.
    ocb_adj_mlt[large] = ocb_adj_mlt[large] - 24.
    
    ocbq1 = (pole_angle < 90.) & (ocb_adj_mlt < 12.)
    ocbq2 = (pole_angle < 90.) & (ocb_adj_mlt > 12.)
    ocbq3 = (pole_angle > 90.) & (ocb_adj_mlt < 24.)
    ocbq4 = (pole_angle > 90.) & (ocb_adj_mlt > 24.)
    ocb_quad[ocbq1] = 1
    ocb_quad[ocbq2] = 2
    ocb_quad[ocbq3] = 3
    ocb_quad[ocbq4] = 4
    
    vecq1 = (apex_n >= 0.) & (apex_e >= 0)
    vecq2 = (apex_n >= 0.) & (apex_e < 0)
    vecq3 = (apex_n < 0.) & (apex_e < 0)
    vecq4 = (apex_n < 0.) & (apex_e >= 0)
    vec_quad[vecq1] = 1
    vec_quad[vecq2] = 2
    vec_quad[vecq3] = 3
    vec_quad[vecq4] = 4
    
    return [ocb_quad, vec_quad]

def calc_ocb_vec_sign(ocb_quad, vec_quad, aacgm_naz, pole_angle):
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


    # Initialise output
    vsign_n = np.zeros(len(pole_angle))
    vsign_e = np.zeros(len(pole_angle))


    #Make new names so one can overwriter the quadrant information
    oq = np.copy(ocb_quad)
    vq = np.copy(vec_quad)

    #north
    pole_minus = pole_angle - 90.0
    minus_pole = 90.0 - pole_angle
    pole_plus = pole_angle + 90.0
    
    
    case1 = (((oq==1) & (vq==1)) | ((oq==2) & (vq==2)) | ((oq==3) & (vq==3)) | ((oq==4) & (vq==4)) | \
             (((oq==1) & (vq==4)) & (aacgm_naz<=pole_plus)) | \
             (((oq==1) & (vq==2)) & (aacgm_naz<minus_pole)) | #2020-03-03 Jone changed to "<" from ">"\
             (((oq==2) & (vq==1)) & (aacgm_naz<=minus_pole)) | \
             ((((oq==3) & (vq==4)) | ((oq==4) & (vq==3))) & (aacgm_naz<=pole_minus)) | \
             ((((oq==3) & (vq==2)) | ((oq==4) & (vq==1))) & (aacgm_naz>pole_minus)) | \
             (((oq==2) & (vq==3)) & (aacgm_naz>minus_pole)))
    vsign_n[case1] = 1
    oq[case1] = 5
    vq[case1] = 5

#check if one have to use elif approach, i.e. if both case1 and case2 can be true
    
    case2 = ((((oq==1) & (vq==2)) & (aacgm_naz>minus_pole)) | \
             (((oq==1) & (vq==4)) & (aacgm_naz>pole_plus)) | \
             (((oq==2) & (vq==1)) & (aacgm_naz>minus_pole)) | \
             ((((oq==4) & (vq==1)) | ((oq==3) & (vq==2))) & (aacgm_naz<=pole_minus)) | \
             (((oq==2) & (vq==3)) & (aacgm_naz<=minus_pole)) | \
             ((((oq==4) & (vq==3)) | ((oq==3) & (vq==4))) & (aacgm_naz>pole_minus)) | \
             ((oq==1) & (vq==3)) | ((oq==2) & (vq==4)) | ((oq==3) & (vq==1)) | \
             ((oq==4) & (vq==2)))
    vsign_n[case2] = -1
        
    #Redefine names
    oq = np.copy(ocb_quad)
    vq = np.copy(vec_quad)
    #east
    minus_pole = 180.0 - pole_angle

    case1 = ((((oq==1) & (vq==4)) | ((oq==2) & (vq==1)) | ((oq==3) & (vq==2)) | \
              ((oq==4) & (vq==3))) | \
                (((oq==1) & (vq==1)) & (aacgm_naz>pole_angle)) | \
                (((oq==1) & (vq==3)) & (aacgm_naz>minus_pole)) | \
                ((((oq==4) & (vq==4)) | ((oq==3) & (vq==1))) & (aacgm_naz<=minus_pole)) | \
                (((oq==2) & (vq==4)) & (aacgm_naz>pole_angle)) | \
                ((((oq==4) & (vq==2)) | ((oq==3) & (vq==3))) & (aacgm_naz>minus_pole)) | \
                (((oq==2) & (vq==2)) & (aacgm_naz<=pole_angle)))
    vsign_e[case1] = 1
    oq[case1] = 5
    vq[case1] = 5    
    
    case2 = ((((oq==1) & (vq==2)) | ((oq==2) & (vq==3)) | ((oq==3) & (vq==4)) | \
              ((oq==4) & (vq==1))) | \
                ((((oq==4) & (vq==4)) | ((oq==3) & (vq==1))) & (aacgm_naz>minus_pole)) | \
                (((oq==2) & (vq==2)) & (aacgm_naz>pole_angle)) | \
                ((((oq==4) & (vq==2)) | ((oq==3) & (vq==3))) & (aacgm_naz<=minus_pole)) | \
                (((oq==1) & (vq==3)) & (aacgm_naz<=minus_pole)) | \
                ((((oq==1) & (vq==1)) | ((oq==2) & (vq==4))) & (aacgm_naz<=pole_angle)))
    vsign_e[case2] = -1

    return [vsign_e, vsign_n]

def calc_ocb_polar_angle(ocb_quad, vec_quad, aacgm_naz, pole_angle):
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


    # Initialise the output and set the quadrant dictionary
    ocb_naz = np.zeros(len(pole_angle))
#    quads = [{o:{v:True if ocb_quad[i] == o and vec_quad[i] == v else False for v in quad_range} \
#      for o in quad_range} for i in range(len(ocb_quad))]
    
    case1 = (((((ocb_quad==2) & (vec_quad==4)) | ((ocb_quad==2) & (vec_quad==2))) & (aacgm_naz>pole_angle)) \
             | ((aacgm_naz>pole_angle) & (ocb_quad==1) & (vec_quad==1)) | ((ocb_quad==1) & (vec_quad==4) & (aacgm_naz<pole_angle+90)))
    ocb_naz[case1] = aacgm_naz[case1] - pole_angle[case1]
    
    case2 = (((aacgm_naz<=pole_angle) & (ocb_quad==1) & (vec_quad==1)) | \
             ((aacgm_naz<=pole_angle) & (((ocb_quad==2) & (vec_quad==4)) | ((ocb_quad==2) & (vec_quad==2)))) | \
             ((aacgm_naz>pole_angle-90.) & \
                  (((ocb_quad==4) & (vec_quad==1)) | ((ocb_quad==4) & (vec_quad==3)) | 
                          ((ocb_quad==3) & (vec_quad==4)) | ((ocb_quad==3) & (vec_quad==2)))))
    ocb_naz[case2] = pole_angle[case2] - aacgm_naz[case2]

    case3 = ((aacgm_naz<=90.-pole_angle) & \
             ((ocb_quad==1) & (vec_quad==2) | ((ocb_quad==2) & (vec_quad==1)) | ((ocb_quad==2) & (vec_quad==3))))
    ocb_naz[case3] = aacgm_naz[case3] + pole_angle[case3]

    case4 = ((aacgm_naz>90.-pole_angle) & (((ocb_quad==1) & (vec_quad==2)) | ((ocb_quad==2) & (vec_quad==1)) | ((ocb_quad==2) & (vec_quad==3))) | \
             ((((ocb_quad==4) & (vec_quad==4)) | ((ocb_quad==4) & (vec_quad==2)) | ((ocb_quad==3) & (vec_quad==1)) | 
                     ((ocb_quad==3) & (vec_quad==3)) | ((ocb_quad==1) & (vec_quad==3)))
                        & (aacgm_naz<=180.-pole_angle)))
    ocb_naz[case4] = 180. - aacgm_naz[case4] - pole_angle[case4]
    
    case5 = (((((ocb_quad==3) & (vec_quad==1)) | ((ocb_quad==3) & (vec_quad==3)) | ((ocb_quad==4) & (vec_quad==4)) | \
               ((ocb_quad==4) & (vec_quad==2)) | ((ocb_quad==1) & (vec_quad==3))) & (aacgm_naz>180.-pole_angle)) | \
                (((ocb_quad==1) & (vec_quad==4)) & (aacgm_naz>pole_angle+90.)))
    ocb_naz[case5] = aacgm_naz[case5] - 180. + pole_angle[case5]

    case6 = ((aacgm_naz<=pole_angle-90.) & (((ocb_quad==3) & (vec_quad==4)) | ((ocb_quad==3) & (vec_quad==2)) | \
              ((ocb_quad==4) & (vec_quad==1)) | ((ocb_quad==4) & (vec_quad==3))))
    ocb_naz[case6] = 180. - pole_angle[case6] + aacgm_naz[case6]

    return ocb_naz


def normal_coord(aacgm_lat, aacgm_mlt, phi_cent, r_cent, r, boundary_lat, hemi = 1):
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

    phi_cent_rad = np.radians(phi_cent)
    xc = r_cent * np.cos(phi_cent_rad)
    yc = r_cent * np.sin(phi_cent_rad)

    scalep = 90.0 - np.abs(aacgm_lat)
    xp = scalep * np.cos(np.radians(aacgm_mlt * 15.0))
    yp = scalep * np.sin(np.radians(aacgm_mlt * 15.0))

    scalen = (90.0 - abs(boundary_lat)) / r
    xn = (xp - xc) * scalen
    yn = (yp - yc) * scalen

    ocb_lat = hemi * (90.0 - np.sqrt(xn**2 + yn**2))
    ocb_mlt = np.degrees(np.arctan2(yn, xn)) / 15.0

    negs = ocb_mlt < 0
    ocb_mlt[negs] += 24.0

    return ocb_lat, ocb_mlt
    

def rotate_ocb(mlat, mlt, oval_r, center_r, center_mlt, apex_e, apex_n, boundary=None):
    """
    This function with the above dependensies are copied from the OCBpy library
    https://github.com/aburrell/ocbpy/
    developed and maintained by A. Burrell. J.P. More specifically, the 
    ocb_scaling.py file. Reistad has only implemented the support for array 
    input/output to increase speed.
    
    inputs:
        mlat: apex/aacgm mlat location of vector to be converted [degrees]
        mlt: mlt location of vector to ve converted [hrs]
        oval_r: polar cap radius, in degrees
        center_r: center of oval radial coordinate [degrees]
        center_mlt: center of oval longitudinal coordinate [hrs]
        apex_n: magnitude of vector to be converted, along apex_north direction
        apex_e: magnitude of vector to be converted, along apex_east direction
        boundary: typical polar cap size for the array values provided in 
            the other input variables, scalar [degrees]. If this keyword is set
            the magnitude of the converted vector will be scaled according to 
            E3. 3 in Chisham 2017. If None, no scaling is performed.

    output:
        [ocb_e, ocb_n] being the corresponding arrays of the east and north 
            components of the converted and scaled vectors
    """
        
    ocb_e = np.zeros(len(apex_e))   #To hold the result
    ocb_n = np.zeros(len(apex_e))   #To hold the result
    
    #Vectors of zero length
    zeros = (apex_e == 0.) & (apex_n == 0)   #No magnitude, nothing to adjust
    ocb_e[zeros] = 0.
    ocb_n[zeros] = 0.
    
    #Vectors along the line between apex and ocb pole
    center_mlat = 90. - center_r    #mlat of the center of the oval
    pole_angle = calc_vec_pole_angle(mlat, mlt, center_mlat, center_mlt)
    aligned = (pole_angle == 0.0) | (pole_angle == 180.)    # The measurement is aligned with the AACGM and OCB poles
    ocb_e[aligned] = oval_r[aligned]/(90.-boundary) * apex_e[aligned]      #scale magnitude with size of present polar cap to its typical value (boundary)
    ocb_n[aligned] = oval_r[aligned]/(90.-boundary) * apex_n[aligned]
    between = (pole_angle == 0) & (mlat >= center_mlat) # The measurement is on or between the poles
    ocb_e[between] = -1.*ocb_e[between]
    ocb_n[between] = -1.*ocb_n[between]
    
    convert = (zeros==False) & (aligned==False) & (between==False)
    
    #Get the ocb and vec quadrants
    [ocb_quad, vec_quad] = define_quadrants(center_mlt[convert], mlt[convert], pole_angle[convert], apex_e[convert], apex_n[convert])
        
    # Get the unscaled 2D vector magnitude
    vmag = np.sqrt(apex_n[convert]**2 + apex_e[convert]**2)

    # Calculate the AACGM north azimuth in degrees [0,180]
    aacgm_naz = np.degrees(np.arccos(apex_n[convert] / vmag))

    # Get the OCB north azimuth in radians
    ocb_angle = np.radians(calc_ocb_polar_angle(ocb_quad, vec_quad, aacgm_naz, pole_angle[convert]))

    # Get the sign of the North and East components
    [vsigns_e, vsigns_n] = calc_ocb_vec_sign(ocb_quad, vec_quad, aacgm_naz, pole_angle[convert])

    # Scale the vector along the OCB north and account for
    # any changes associated with adjusting the size of the polar cap
    if boundary != None:
        vmag =  vmag * oval_r[convert]/(90.-boundary)
    
    ocb_n[convert] = vsigns_n * vmag * np.cos(ocb_angle)
    ocb_e[convert] = vsigns_e * vmag * np.sin(ocb_angle)

    return [ocb_e, ocb_n]
    