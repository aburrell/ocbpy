#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocb_scaling class and functions
"""

from io import StringIO
import logging
import numpy as np
from os import path
from sys import version_info
import unittest

import ocbpy

class TestOCBScalingLogFailure(unittest.TestCase):
    def setUp(self):
        """ Initialize the test class"""
        # Initialize the logging info
        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.INFO)

        # Initialize the testing variables
        test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                              "test_data", "test_north_circle")
        self.assertTrue(path.isfile(test_file))
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=test_file,
                                               instrument='image')
        self.ocb.rec_ind = 27
        self.vdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind, 75.0,
                                                  22.0, aacgm_n=50.0,
                                                  aacgm_e=86.5, aacgm_z=5.0,
                                                  dat_name="Test",
                                                  dat_units="$m s^{-1}$")

    def tearDown(self):
        """ Tear down the test case"""
        del self.lwarn, self.lout, self.log_capture, self.ocb, self.vdata

    def test_no_scale_func(self):
        """ Test OCBScaling initialization with no scaling function """
        self.lwarn = u"no scaling function provided"

        # Initialize the OCBScaling class without a scaling function
        self.vdata.set_ocb(self.ocb)
        self.assertIsNone(self.vdata.scale_func)

        self.lout = self.log_capture.getvalue()
        # Test logging error message for each bad initialization
        self.assertTrue(self.lout.find(self.lwarn) >= 0)


class TestOCBScalingMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """

        test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                              "test_data", "test_north_circle")
        self.assertTrue(path.isfile(test_file))
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=test_file,
                                               instrument='image')
        self.ocb.rec_ind = 27
        self.vdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind, 75.0,
                                                  22.0, aacgm_n=50.0,
                                                  aacgm_e=86.5, aacgm_z=5.0,
                                                  dat_name="Test",
                                                  dat_units="$m s^{-1}$")
        self.wdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind, 75.0,
                                                  22.0, aacgm_n=50.0,
                                                  aacgm_e=86.5, aacgm_z=5.0,
                                                  aacgm_mag=100.036243432,
                                                  dat_name="Test",
                                                  dat_units="$m s^{-1}$")
        self.zdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind, 87.2,
                                                  21.22, aacgm_n=0.0,
                                                  aacgm_e=0.0,
                                                  dat_name="Test Zero",
                                                  dat_units="$m s^{-1}$")

        if version_info.major == 2:
            self.assertRegex = self.assertRegexpMatches
            self.assertRaisesRegex = self.assertRaisesRegexp

    def tearDown(self):
        del self.ocb, self.vdata, self.wdata, self.zdata

    def test_init_nez(self):
        """ Test the initialisation of the VectorData object without magnitude
        """
        self.assertAlmostEqual(self.vdata.aacgm_mag, 100.036243432)
        self.assertAlmostEqual(self.zdata.aacgm_mag, 0.0)

    def test_init_mag(self):
        """ Test the initialisation of the VectorData object with magnitude
        """
        self.assertAlmostEqual(self.wdata.aacgm_mag, 100.036243432)

    def test_init_failure(self):
        """ Test the initialisation of the VectorData object with inconsistent
        AACGM components
        """
        with self.assertRaisesRegexp(ValueError, "inconsistent AACGM"):
            self.wdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind, 75.0,
                                                      22.0, aacgm_mag=100.0,
                                                      dat_name="Test",
                                                      dat_units="$m s^{-1}$")

    def test_vector_repr_str(self):
        """ Test the VectorData print statement using repr and str """
        self.assertTrue(self.vdata.__repr__() == self.vdata.__str__())

    def test_vector_repr_no_scaling(self):
        """ Test the VectorData print statement without a scaling function """
        out = self.vdata.__repr__()

        self.assertRegex(out, "Vector data:")
        self.assertRegex(out, "No magnitude scaling function")
        del out

    def test_vector_repr_with_scaling(self):
        """ Test the VectorData print statement with a scaling function """
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        out = self.vdata.__repr__()

        print(out)

        self.assertRegex(out, "Vector data:")
        self.assertRegex(out, "Scaling function")

    def test_vector_bad_lat(self):
        """ Test the VectorData output with data from the wrong hemisphere """
        self.vdata.aacgm_lat *= -1.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)

        self.assertTrue(np.isnan(self.vdata.ocb_lat))
        self.assertTrue(np.isnan(self.vdata.ocb_mlt))
        self.assertTrue(np.isnan(self.vdata.r_corr))
        self.assertTrue(np.isnan(self.vdata.ocb_n))
        self.assertTrue(np.isnan(self.vdata.ocb_e))
        self.assertTrue(np.isnan(self.vdata.ocb_z))

    def test_calc_large_pole_angle(self):
        """ Test to see that the OCB polar angle calculation is performed
        properly when the angle is greater than 90 degrees
        """
        self.zdata.ocb_aacgm_mlt = 1.260677777
        self.zdata.ocb_aacgm_lat = 83.99
        self.zdata.ocb_lat = 84.838777192
        self.zdata.ocb_mlt = 15.1110383783

        self.zdata.calc_vec_pole_angle()
        self.assertAlmostEqual(self.zdata.pole_angle, 91.72024697182087)
        
    def test_calc_vec_pole_angle(self):
        """ Test to see that the polar angle calculation is performed properly
        """
        self.vdata.ocb_aacgm_mlt = self.ocb.phi_cent[self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.r_cent[self.vdata.ocb_ind]
        (self.vdata.ocb_lat, self.vdata.ocb_mlt,
         self.vdata.r_corr) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                    self.vdata.aacgm_mlt)

        # Test the calculation of the test pole angle
        self.vdata.calc_vec_pole_angle()
        self.assertAlmostEqual(self.vdata.pole_angle, 8.67527923044)
        
        # If the measurement has the same MLT as the OCB pole, the angle is
        # zero.  This includes the AACGM pole and the OCB pole.
        self.vdata.aacgm_mlt = self.vdata.ocb_aacgm_mlt
        self.vdata.calc_vec_pole_angle()
        self.assertEqual(self.vdata.pole_angle, 0.0)

        # If the measurement is on the opposite side of the AACGM pole as the
        # OCB pole, then the angle will be 180 degrees
        self.vdata.aacgm_mlt = self.vdata.ocb_aacgm_mlt + 12.0
        self.vdata.calc_vec_pole_angle()
        self.assertEqual(self.vdata.pole_angle, 180.0)

        # Set the distance between the data point and the OCB is equal to the
        # distance between the AACGM pole and the OCB so that the triangles
        # we're examining are isosceles triangles.  If the triangles were flat,
        # the angle would be 46.26 degrees
        self.vdata.aacgm_mlt = 0.0
        self.vdata.aacgm_lat = self.vdata.ocb_aacgm_lat
        self.vdata.calc_vec_pole_angle()
        self.assertAlmostEqual(self.vdata.pole_angle, 46.2932179019)

        # Return to the default AACGM MLT value
        self.vdata.aacgm_mlt = 22.0

    def test_define_quadrants(self):
        """ Test the assignment of quadrants
        """
        # Set the initial values
        self.vdata.ocb_aacgm_mlt = self.ocb.phi_cent[self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.r_cent[self.vdata.ocb_ind]
        (self.vdata.ocb_lat, self.vdata.ocb_mlt,
         self.vdata.r_corr) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                     self.vdata.aacgm_mlt)
        self.vdata.calc_vec_pole_angle()
        
        # Get the test quadrants
        self.vdata.define_quadrants()
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 1)

    def test_define_quadrants_neg_adj_mlt(self):
        """ Test the quadrant assignment with a negative AACGM MLT """
        self.vdata.aacgm_mlt = -22.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertGreater(self.vdata.ocb_aacgm_mlt-self.vdata.aacgm_mlt, 24)
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 1)

    def test_define_quadrants_neg_north(self):
        """ Test the quadrant assignment with a vector pointing south """
        self.vdata.aacgm_n *= -1.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 4)

    def test_define_quadrants_noon_north(self):
        """ Test the quadrant assignment with a vector pointing north from noon
        """
        self.vdata.aacgm_mlt = 12.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 2)
        self.assertEqual(self.vdata.vec_quad, 1)

    def test_define_quadrants_opposite_south(self):
        """ Test the quadrant assignment with a vector pointing south from the
        opposite sector
        """
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.vdata.aacgm_mlt = self.vdata.ocb_aacgm_mlt + 12.0
        self.vdata.aacgm_n = -10.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 3)
        self.assertEqual(self.vdata.vec_quad, 4)

    def test_define_quadrants_ocb_south(self):
        """ Test the quadrant assignment with the OCB pole in a southern quad"""
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.vdata.ocb_aacgm_mlt = 10.0
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()
        self.assertEqual(self.vdata.ocb_quad, 3)
        self.assertEqual(self.vdata.vec_quad, 1)

    def test_undefinable_quadrants(self):
        """ Test OCBScaling initialization for undefinable quadrants """
        self.vdata.aacgm_lat = 0.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 0)
        self.assertEqual(self.vdata.vec_quad, 0)

    def test_lost_ocb_quadrant(self):
        """ Test OCBScaling initialization for unset quadrants """
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 1)
        self.vdata.ocb_quad = 0
        self.vdata.scale_vector()
        self.assertEqual(self.vdata.ocb_quad, 1)

    def test_lost_vec_quadrant(self):
        """ Test OCBScaling initialization for unset quadrants """
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 1)
        self.vdata.vec_quad = 0
        self.vdata.scale_vector()
        self.assertEqual(self.vdata.vec_quad, 1)

    def test_calc_ocb_vec_sign(self):
        """ Test the calculation of the OCB vector signs
        """

        # Set the initial values
        self.vdata.ocb_aacgm_mlt = self.ocb.phi_cent[self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.r_cent[self.vdata.ocb_ind]
        (self.vdata.ocb_lat, self.vdata.ocb_mlt,
         self.vdata.r_corr) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                    self.vdata.aacgm_mlt)
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()

        vmag = np.sqrt(self.vdata.aacgm_n**2 + self.vdata.aacgm_e**2)
        self.vdata.aacgm_naz = np.degrees(np.arccos(self.vdata.aacgm_n / vmag))

        # Calculate the vector data signs
        vsigns = self.vdata.calc_ocb_vec_sign(north=True, east=True)
        self.assertTrue(vsigns['north'])
        self.assertTrue(vsigns['east'])

        del vmag, vsigns

    def test_scale_vec(self):
        """ Test the calculation of the OCB vector signs
        """

        # Set the initial values
        self.vdata.ocb_aacgm_mlt = self.ocb.phi_cent[self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.r_cent[self.vdata.ocb_ind]
        (self.vdata.ocb_lat, self.vdata.ocb_mlt,
         self.vdata.r_corr) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                    self.vdata.aacgm_mlt)
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()

        vmag = np.sqrt(self.vdata.aacgm_n**2 + self.vdata.aacgm_e**2)
        self.vdata.aacgm_naz = np.degrees(np.arccos(self.vdata.aacgm_n / vmag))

        # Scale the data vector
        self.vdata.scale_vector()

        # Test the North and East components
        self.assertAlmostEqual(self.vdata.ocb_n, 62.4751208491)
        self.assertAlmostEqual(self.vdata.ocb_e, 77.9686428950)
        
        # Test to see that the magnitudes and z-components are the same
        self.assertEqual(self.vdata.aacgm_mag, self.vdata.ocb_mag)
        self.assertEqual(self.vdata.ocb_z, self.vdata.aacgm_z)

        del vmag

    def test_scale_vec_z_zero(self):
        """ Test the calculation of the OCB vector sign with no vertical aacgm_z
        """
        # Re-assing the necessary variable
        self.vdata.aacgm_z = 0.0

        # Run the scale_vector routine
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)

        # Assess the ocb_z component
        self.assertEqual(self.vdata.ocb_z,
                         self.vdata.scale_func(0.0, self.vdata.unscaled_r,
                                               self.vdata.scaled_r))

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_scale_vec_pole_angle_zero(self):
        """ Test the calculation of the OCB vector sign with no pole angle
        """
        self.vdata.set_ocb(self.ocb)
        self.vdata.pole_angle = 0.0

        nscale = ocbpy.ocb_scaling.normal_evar(self.vdata.aacgm_n,
                                               self.vdata.unscaled_r,
                                               self.vdata.scaled_r)
        escale = ocbpy.ocb_scaling.normal_evar(self.vdata.aacgm_e,
                                               self.vdata.unscaled_r,
                                               self.vdata.scaled_r)
        
        # Cycle through all the possible options for a pole angle of zero/180
        for tset in [('scale_func', None, self.vdata.aacgm_n,
                      self.vdata.aacgm_e),
                     ('scale_func', ocbpy.ocb_scaling.normal_evar, nscale,
                      escale),
                          ('ocb_aacgm_lat', self.vdata.aacgm_lat, -1.0*nscale,
                           -1.0*escale)]:
            with self.subTest(tset=tset):
                setattr(self.vdata, tset[0], tset[1])

                # Run the scale_vector routine with the new attributes
                self.vdata.scale_vector()
        
                # Assess the ocb north and east components
                self.assertEqual(self.vdata.ocb_n, tset[2])
                self.assertEqual(self.vdata.ocb_e, tset[3])

        del nscale, escale, tset

    @unittest.skipIf(version_info.major > 2, 'Already tested')
    def test_scale_vec_pole_angle_zero_none(self):
        """ Test the OCB vector sign routine with no pole angle or scaling
        """
        self.vdata.set_ocb(self.ocb)
        self.vdata.pole_angle = 0.0

        # Run the scale_vector routine with the new attributes
        self.vdata.scale_vector()
        
        # Assess the ocb north and east components
        self.assertEqual(self.vdata.ocb_n, self.vdata.aacgm_n)
        self.assertEqual(self.vdata.ocb_e, self.vdata.aacgm_e)


    @unittest.skipIf(version_info.major > 2, 'Already tested')
    def test_scale_vec_pole_angle_zero_scale_at_pole(self):
        """ Test the OCB vector sign routine with no pole angle and data at pole
        """
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.vdata.pole_angle = 0.0
        self.vdata.ocb_aacgm_lat = self.vdata.aacgm_lat

        nscale = -1.0 * ocbpy.ocb_scaling.normal_evar(self.vdata.aacgm_n,
                                                      self.vdata.unscaled_r,
                                                      self.vdata.scaled_r)
        escale = -1.0 * ocbpy.ocb_scaling.normal_evar(self.vdata.aacgm_e,
                                                      self.vdata.unscaled_r,
                                                      self.vdata.scaled_r)
        
        # Run the scale_vector routine with the new attributes
        self.vdata.scale_vector()
        
        # Assess the ocb north and east components
        self.assertEqual(self.vdata.ocb_n, nscale)
        self.assertEqual(self.vdata.ocb_e, escale)

        del nscale, escale

    def test_set_ocb_zero(self):
        """ Test setting of OCB values for the VectorData object without any
        magnitude
        """
        # Set the OCB values without any E-field scaling, test to see that the
        # AACGM and OCB vector magnitudes are the same
        self.zdata.set_ocb(self.ocb)
        self.assertEqual(self.zdata.ocb_mag, 0.0)

    def test_set_ocb_none(self):
        """ Test setting of OCB values for the VectorData object
        """

        # Set the OCB values without any E-field scaling, test to see that the
        # AACGM and OCB vector magnitudes are the same
        self.vdata.set_ocb(self.ocb)
        self.assertEqual(self.vdata.aacgm_mag, self.vdata.ocb_mag)

    def test_set_ocb_evar(self):
        """ Test setting of OCB values for the VectorData object
        """

        # Set the OCB values with scaling for a variable proportional to
        # the electric field
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertAlmostEqual(self.vdata.ocb_mag, 88.1262660863)
        
    def test_set_ocb_curl_evar(self):
        """ Test setting of OCB values for the VectorData object
        """

        # Set the OCB values with scaling for a variable proportional to
        # the curl of the electric field
        self.vdata.set_ocb(self.ocb,
                           scale_func=ocbpy.ocb_scaling.normal_curl_evar)
        self.assertAlmostEqual(self.vdata.ocb_mag, 77.6423447186)

    def test_scaled_r(self):
        """ Test that the scaled radius is correct
        """
        self.vdata.set_ocb(self.ocb, None)
        self.assertEqual(self.vdata.scaled_r, 16.0)

    def test_unscaled_r(self):
        """ Test that the scaled radius is correct
        """
        self.vdata.set_ocb(self.ocb, None)
        self.assertEqual(self.vdata.unscaled_r, 14.09)

    def test_no_ocb_lat(self):
        """ Test failure when OCB latitude is not available"""

        self.vdata.ocb_lat = np.nan
        
        with self.assertRaisesRegex(ValueError, 'OCB coordinates required'):
            self.vdata.scale_vector()

    def test_no_ocb_mlt(self):
        """ Test failure when OCB latitude is not available"""

        self.vdata.ocb_mlt = np.nan
        
        with self.assertRaisesRegex(ValueError, 'OCB coordinates required'):
            self.vdata.scale_vector()

    def test_no_ocb_pole_location(self):
        """ Test failure when OCB latitude is not available"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_mlt = np.nan
        
        with self.assertRaisesRegex(ValueError, "OCB pole location required"):
            self.vdata.scale_vector()

    def test_no_ocb_pole_angle(self):
        """ Test failure when pole angle is not available"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = np.nan
        
        with self.assertRaisesRegex(ValueError,
                            "vector angle in poles-vector triangle required"):
            self.vdata.scale_vector()

    def test_bad_ocb_quad(self):
        """ Test failure when OCB quadrant is wrong"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_quad = -1
        
        with self.assertRaisesRegex(ValueError, "OCB quadrant undefined"):
            self.vdata.calc_ocb_polar_angle()

    def test_bad_vec_quad(self):
        """ Test failure when vector quadrant is wrong"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.vec_quad = -1
        
        with self.assertRaisesRegex(ValueError, "Vector quadrant undefined"):
            self.vdata.calc_ocb_polar_angle()

    def test_bad_quad_polar_angle(self):
        """ Test failure when quadrant polar angle is bad"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_naz = np.nan
        
        with self.assertRaisesRegex(ValueError, "AACGM polar angle undefined"):
            self.vdata.calc_ocb_polar_angle()

    def test_bad_quad_pole_angle(self):
        """ Test failure when quandrant pole angle is bad"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = np.nan
        
        with self.assertRaisesRegex(ValueError, "Vector angle undefined"):
            self.vdata.calc_ocb_polar_angle()

    def test_bad_calc_vec_sign_direction(self):
        """ Test calc_vec_sign failure when no direction is provided"""

        self.vdata.set_ocb(self.ocb, None)
        
        with self.assertRaisesRegex(ValueError,
                                    "must set at least one direction"):
            self.vdata.calc_ocb_vec_sign()

    def test_bad_calc_sign_ocb_quad(self):
        """ Test calc_vec_sign failure with bad OCB quadrant"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_quad = -1
        
        with self.assertRaisesRegex(ValueError, "OCB quadrant undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)

    def test_bad_calc_sign_vec_quad(self):
        """ Test calc_vec_sign failure with bad vector quadrant"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.vec_quad = -1
        
        with self.assertRaisesRegex(ValueError, "Vector quadrant undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)

    def test_bad_calc_sign_polar_angle(self):
        """ Test calc_vec_sign failure with bad polar angle"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_naz = np.nan
        
        with self.assertRaisesRegex(ValueError,
                                    "AACGM polar angle(s) undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)

    def test_bad_calc_sign_pole_angle(self):
        """ Test calc_vec_sign failure with bad pole angle"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = np.nan
        
        with self.assertRaisesRegex(ValueError, "Vector angle undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)

    def test_bad_calc_vec_pole_angle_mlt(self):
        """Test calc_vec_pole_angle failure with bad AACGM MLT"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_mlt = np.nan
        
        with self.assertRaisesRegex(ValueError,
                                    "AACGM MLT of Vector(s) undefinded"):
            self.vdata.calc_vec_pole_angle()

    def test_bad_calc_vec_pole_angle_ocb_mlt(self):
        """Test calc_vec_pole_angle failure with bad OCB MLT"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_mlt = np.nan
        
        with self.assertRaisesRegex(ValueError,
                                    "AACGM MLT of OCB pole(s) undefined"):
            self.vdata.calc_vec_pole_angle()

    def test_bad_calc_vec_pole_angle_ocb_mlat(self):
        """Test calc_vec_pole_angle failure with bad OCB latitude"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_lat = np.nan
        
        with self.assertRaisesRegex(ValueError,
                                    "AACGM latitude of OCB pole(s) undefined"):
            self.vdata.calc_vec_pole_angle()

    def test_bad_calc_vec_pole_angle_vec_mlat(self):
        """Test calc_vec_pole_angle failure with bad vector latitude"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_lat = np.nan
        
        with self.assertRaisesRegex(ValueError,
                                    "AACGM latitude of Vector(s) undefined"):
            self.vdata.calc_vec_pole_angle()

    def test_bad_define_quandrants_pole_mlt(self):
        """Test define_quadrants failure with bad pole MLT"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_mlt = np.nan
        
        with self.assertRaisesRegex(ValueError, "OCB pole location required"):
            self.vdata.define_quadrants()

    def test_bad_define_quandrants_vec_mlt(self):
        """Test define_quadrants failure with bad vector MLT"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_mlt = np.nan
        
        with self.assertRaisesRegex(ValueError,
                                    "Vector AACGM location required"):
            self.vdata.define_quadrants()

    def test_bad_define_quandrants_pole_angle(self):
        """Test define_quadrants failure with bad pole angle"""

        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = np.nan
        
        with self.assertRaisesRegex(ValueError,
                            "vector angle in poles-vector triangle required"):
            self.vdata.define_quadrants()


class TestHaversine(unittest.TestCase):

    def setUp(self):
        """ Initialize the tests for the haversine and archaversine functions
        """
        self.input_angles = np.linspace(-2.0*np.pi, 2.0*np.pi, 5)
        self.hav_out = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        self.iarchav = [2, 3]
        self.out = None

    def tearDown(self):
        del self.input_angles, self.hav_out, self.out, self.iarchav

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_haversine(self):
        """ Test implimentation of the haversine
        """
        
        # Cycle through all the possible input options
        for i, tset in enumerate([(self.input_angles[0], self.hav_out[0]),
                                  (list(self.input_angles), self.hav_out),
                                  (self.input_angles, self.hav_out)]):
            with self.subTest(tset=tset):
                self.out = ocbpy.ocb_scaling.hav(tset[0])
        
                # Assess the output
                if i == 0:
                    self.assertAlmostEqual(self.out, tset[1])
                else:
                    for j, hout in enumerate(self.out):
                        self.assertAlmostEqual(hout, tset[1][j])

        del tset

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_haversine_float(self):
        """ Test implimentation of the haversine with float inputs
        """

        for i, alpha in enumerate(self.input_angles):
            self.assertAlmostEqual(ocbpy.ocb_scaling.hav(alpha),
                                   self.hav_out[i])

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_haversine_list(self):
        """ Test implimentation of the haversine with a list input
        """
        self.out = ocbpy.ocb_scaling.hav(list(self.input_angles))
        for i, hout in enumerate(self.out):
            self.assertAlmostEqual(hout, self.hav_out[i])

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_haversine_array(self):
        """ Test implimentation of the haversine with a array input
        """
        self.out = ocbpy.ocb_scaling.hav(self.input_angles)
        for i, hout in enumerate(self.out):
            self.assertAlmostEqual(hout, self.hav_out[i])

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_inverse_haversine(self):
        """ Test the implemenation of the inverse haversine
        """
        
        # Cycle through all the possible input options
        for i, tset in enumerate([(self.hav_out[self.iarchav[0]],
                                   self.input_angles[self.iarchav[0]]),
                                  ([self.hav_out[j] for j in self.iarchav],
                                  self.input_angles[self.iarchav]),
                                  (self.hav_out[self.iarchav],
                                   self.input_angles[self.iarchav])]):
            with self.subTest(tset=tset):
                self.out = ocbpy.ocb_scaling.archav(tset[0])
        
                # Assess the output
                if i == 0:
                    self.assertEqual(self.out, tset[1])
                else:
                    for k, hout in enumerate(self.out):
                        self.assertAlmostEqual(hout, tset[1][k])

        del tset

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_inverse_haversine_float(self):
        """ Test implimentation of the inverse haversine with float input
        """
        for i in self.iarchav:
            self.assertEqual(ocbpy.ocb_scaling.archav(self.hav_out[i]),
                             self.input_angles[i])

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_inverse_haversine_list(self):
        """ Test implimentation of the inverse haversine with list input
        """
        self.out = ocbpy.ocb_scaling.archav([self.hav_out[i]
                                             for i in self.iarchav])
        for i, hout in enumerate(self.out):
            self.assertEqual(hout, self.input_angles[self.iarchav[i]])

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_inverse_haversine_array(self):
        """ Test implimentation of the inverse haversine with array input
        """
        self.out = ocbpy.ocb_scaling.archav(self.hav_out[self.iarchav])
        for i, hout in enumerate(self.out):
            self.assertEqual(hout, self.input_angles[self.iarchav[i]])

    def test_inverse_haversine_small_float(self):
        """ Test implimentation of the inverse haversine with very small numbers
        """
        self.assertEqual(ocbpy.ocb_scaling.archav(1.0e-17), 0.0)
        self.assertEqual(ocbpy.ocb_scaling.archav(-1.0e-17), 0.0)

    def test_inverse_haversine_nan_float(self):
        """ Test implimentation of the inverse haversine with NaN
        """
        self.assertTrue(np.isnan(ocbpy.ocb_scaling.archav(np.nan)))

    def test_inverse_haversine_negative_float(self):
        """ Test implimentation of the inverse haversine with negative input
        """
        self.assertTrue(np.isnan(ocbpy.ocb_scaling.archav(-1.0)))

    def test_inverse_haversine_mixed(self):
        """ Test the inverse haversine with array input of good and bad values
        """
        # Update the test input and output
        j = [i for i in range(len(self.hav_out)) if i not in self.iarchav]
        self.hav_out[j[0]] = 1.0e-17
        self.input_angles[j[0]] = 0.0
        self.hav_out[j[1]] = np.nan
        self.input_angles[j[1]] = np.nan
        self.hav_out[j[2]] = -1.0
        self.input_angles[j[2]] = np.nan

        self.out = ocbpy.ocb_scaling.archav(self.hav_out)

        for i, hout in enumerate(self.out):
            if np.isnan(hout):
                self.assertTrue(np.isnan(self.input_angles[i]))
            else:
                self.assertEqual(hout, self.input_angles[i])


if __name__ == '__main__':
    unittest.main()

