#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
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
        """ Initialize the test class
        """
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
        """ Tear down the test case
        """
        del self.lwarn, self.lout, self.log_capture, self.ocb, self.vdata

    def test_no_scale_func(self):
        """ Test OCBScaling initialization with no scaling function
        """
        self.lwarn = u"no scaling function provided"

        # Initialize the OCBScaling class without a scaling function
        self.vdata.set_ocb(self.ocb)
        self.assertIsNone(self.vdata.scale_func)

        self.lout = self.log_capture.getvalue()
        # Test logging error message for each bad initialization
        self.assertTrue(self.lout.find(self.lwarn) >= 0)


class TestOCBScalingMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary and VectorData objects
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

    def test_vector_repr_str(self):
        """ Test the VectorData print statement using repr and str
        """
        self.assertTrue(self.vdata.__repr__() == self.vdata.__str__())

    def test_vector_repr_no_scaling(self):
        """ Test the VectorData print statement without a scaling function
        """
        out = self.vdata.__repr__()

        self.assertRegex(out, "Vector data:")
        self.assertRegex(out, "No magnitude scaling function")
        del out

    def test_vector_repr_with_scaling(self):
        """ Test the VectorData print statement with a scaling function
        """
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        out = self.vdata.__repr__()

        self.assertRegex(out, "Vector data:")
        self.assertRegex(out, "Scaling function")

    def test_vector_bad_lat(self):
        """ Test the VectorData output with data from the wrong hemisphere
        """
        self.vdata.aacgm_lat *= -1.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)

        self.assertTrue(np.isnan(self.vdata.ocb_lat))
        self.assertTrue(np.isnan(self.vdata.ocb_mlt))
        self.assertTrue(np.isnan(self.vdata.r_corr))
        self.assertTrue(np.isnan(self.vdata.ocb_n))
        self.assertTrue(np.isnan(self.vdata.ocb_e))
        self.assertTrue(np.isnan(self.vdata.ocb_z))

    def test_calc_large_pole_angle(self):
        """ Test the OCB polar angle calculation with angles > 90 deg
        """
        self.zdata.ocb_aacgm_mlt = 1.260677777
        self.zdata.ocb_aacgm_lat = 83.99
        self.zdata.ocb_lat = 84.838777192
        self.zdata.ocb_mlt = 15.1110383783

        self.zdata.calc_vec_pole_angle()
        self.assertAlmostEqual(self.zdata.pole_angle, 91.72024697182087)

    def test_calc_vec_pole_angle_acute(self):
        """ Test the polar angle calculation with an acute angle
        """
        self.vdata.set_ocb(self.ocb)
        self.assertAlmostEqual(self.vdata.pole_angle, 8.67527923)

    def test_calc_vec_pole_angle_zero(self):
        """ Test the polar angle calculation with an angle of zero
        """
        self.vdata.set_ocb(self.ocb)
        self.vdata.aacgm_mlt = self.vdata.ocb_aacgm_mlt
        self.vdata.calc_vec_pole_angle()
        self.assertEqual(self.vdata.pole_angle, 0.0)

    def test_calc_vec_pole_angle_flat(self):
        """ Test the polar angle calculation with an angle of 180 deg
        """
        self.vdata.set_ocb(self.ocb)
        self.vdata.ocb_aacgm_mlt = 6.0
        self.vdata.aacgm_mlt = 6.0
        self.vdata.aacgm_lat = 45.0 + 0.5 * self.vdata.ocb_aacgm_lat
        self.vdata.calc_vec_pole_angle()
        self.assertEqual(self.vdata.pole_angle, 180.0)

    def test_calc_vec_pole_angle_right_isosceles(self):
        """ Test the polar angle calculation with a right isosceles triangle
        """
        # Set the distance between the data point and the OCB is equal to the
        # distance between the AACGM pole and the OCB so that the triangles
        # we're examining are isosceles triangles.  If the triangles were flat,
        # the angle would be 45 degrees
        self.vdata.set_ocb(self.ocb)
        self.vdata.ocb_aacgm_mlt = 0.0
        self.vdata.aacgm_mlt = 6.0
        self.vdata.aacgm_lat = self.vdata.ocb_aacgm_lat
        self.vdata.calc_vec_pole_angle()
        self.assertAlmostEqual(self.vdata.pole_angle, 45.03325090532819)

    def test_calc_vec_pole_angle_oblique(self):
        """ Test the polar angle calculation with an isosceles triangle
        """
        self.vdata.set_ocb(self.ocb)
        self.vdata.aacgm_mlt = self.vdata.ocb_aacgm_mlt - 1.0
        self.vdata.aacgm_lat = 45.0 + 0.5 * self.vdata.ocb_aacgm_lat
        self.vdata.calc_vec_pole_angle()
        self.assertAlmostEqual(self.vdata.pole_angle, 150.9561733411)

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

    def test_define_quadrants_neg_adj_mlt_west(self):
        """ Test the quadrant assignment with a negative AACGM MLT and W vect
        """
        self.vdata.aacgm_mlt = -22.0
        self.vdata.aacgm_e *= -1.0
        self.vdata.set_ocb(self.ocb)
        self.assertGreater(self.vdata.ocb_aacgm_mlt-self.vdata.aacgm_mlt, 24)
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 2)

    def test_define_quadrants_neg_north(self):
        """ Test the quadrant assignment with a vector pointing south
        """
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

    def test_define_quadrants_aligned_poles_southwest(self):
        """ Test quad assignment w/vector pointing SW and both poles aligned
        """
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.vdata.aacgm_mlt = self.vdata.ocb_aacgm_mlt + 12.0
        self.vdata.aacgm_n = -10.0
        self.vdata.aacgm_e = -10.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 2)
        self.assertEqual(self.vdata.vec_quad, 3)

    def test_define_quadrants_ocb_south_night(self):
        """ Test the quadrant assignment with the OCB pole in a south/night quad
        """
        self.vdata.aacgm_mlt = 0.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.vdata.ocb_aacgm_mlt = 23.0
        self.vdata.ocb_aacgm_lat = self.vdata.aacgm_lat - 2.0
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()
        self.assertEqual(self.vdata.ocb_quad, 3)
        self.assertEqual(self.vdata.vec_quad, 1)

    def test_define_quadrants_ocb_south_day(self):
        """ Test the quadrant assignment with the OCB pole in a south/day quad
        """
        self.vdata.aacgm_mlt = 0.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.vdata.ocb_aacgm_mlt = 1.0
        self.vdata.ocb_aacgm_lat = self.vdata.aacgm_lat - 2.0
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()
        self.assertEqual(self.vdata.ocb_quad, 4)
        self.assertEqual(self.vdata.vec_quad, 1)

    def test_undefinable_quadrants(self):
        """ Test OCBScaling initialization for undefinable quadrants
        """
        self.vdata.aacgm_lat = 0.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 0)
        self.assertEqual(self.vdata.vec_quad, 0)

    def test_lost_ocb_quadrant(self):
        """ Test OCBScaling initialization for unset quadrants
        """
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 1)
        self.vdata.ocb_quad = 0
        self.vdata.scale_vector()
        self.assertEqual(self.vdata.ocb_quad, 1)

    def test_lost_vec_quadrant(self):
        """ Test OCBScaling initialization for unset quadrants
        """
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
        self.assertAlmostEqual(self.vdata.aacgm_mag, self.vdata.ocb_mag)
        self.assertAlmostEqual(self.vdata.ocb_z, self.vdata.aacgm_z)

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
                     ('ocb_aacgm_lat', self.vdata.aacgm_lat, -1.0 * nscale,
                      -1.0 * escale)]:
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
        """ Test setting of OCB values in VectorData without any magnitude
        """
        # Set the OCB values without any E-field scaling, test to see that the
        # AACGM and OCB vector magnitudes are the same
        self.zdata.set_ocb(self.ocb)
        self.assertEqual(self.zdata.ocb_mag, 0.0)

    def test_set_ocb_none(self):
        """ Test setting of OCB values without scaling
        """

        # Set the OCB values without any E-field scaling, test to see that the
        # AACGM and OCB vector magnitudes are the same
        self.vdata.set_ocb(self.ocb)
        self.assertAlmostEqual(self.vdata.aacgm_mag, self.vdata.ocb_mag)

    def test_set_ocb_evar(self):
        """ Test setting of OCB values with E field scaling
        """

        # Set the OCB values with scaling for a variable proportional to
        # the electric field
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertAlmostEqual(self.vdata.ocb_mag, 88.094416872365)

    def test_set_ocb_curl_evar(self):
        """ Test setting of OCB values with Curl E scaling
        """
        # Set the OCB values with scaling for a variable proportional to
        # the curl of the electric field
        self.vdata.set_ocb(self.ocb,
                           scale_func=ocbpy.ocb_scaling.normal_curl_evar)
        self.assertAlmostEqual(self.vdata.ocb_mag, 77.57814585822645)

    def test_scaled_r(self):
        """ Test that the scaled radius is correct
        """
        self.vdata.set_ocb(self.ocb, None)
        self.assertEqual(self.vdata.scaled_r, 16.0)

    def test_unscaled_r(self):
        """ Test that the unscaled radius is correct
        """
        self.vdata.set_ocb(self.ocb, None)
        self.assertEqual(self.vdata.unscaled_r, 14.09)


class TestVectorDataRaises(unittest.TestCase):
    def setUp(self):
        """ Initialize the tests for calc_vec_pole_angle
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
        self.input_attrs = list()
        self.bad_input = [np.nan, np.full(shape=2, fill_value=np.nan)]
        self.raise_out = list()
        self.hold_val = None

        if version_info.major == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp

    def tearDown(self):
        del self.ocb, self.vdata, self.input_attrs, self.bad_input
        del self.raise_out, self.hold_val

    def test_init_failure(self):
        """ Test init failure with inconsistent AACGM components
        """
        with self.assertRaisesRegex(ValueError, "inconsistent AACGM"):
            self.vdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind,
                                                      75.0, 22.0,
                                                      aacgm_mag=100.0,
                                                      dat_name="Test",
                                                      dat_units="$m s^{-1}$")

    def test_init_ocb_array_failure(self):
        """ Test init failure with mismatched OCB and input array input
        """
        self.input_attrs = [0, [27, 31], 75.0, 22.0]
        self.bad_input = {'aacgm_n': 100.0, 'aacgm_e': 100.0,
                          'aacgm_z': 10.0, 'ocb_lat': 81.0,
                          'ocb_mlt': [2.0, 5.8, 22.5]}

        with self.assertRaisesRegex(ValueError, "OCB index and input shapes"):
            self.vdata = ocbpy.ocb_scaling.VectorData(*self.input_attrs,
                                                      **self.bad_input)

    def test_init_ocb_vector_failure(self):
        """ Test init failure with mismatched OCB and data array input
        """
        self.input_attrs = [[3, 6, 0], [27, 31], [75.0, 87.2, 65.0],
                            [22.0, 21, 22]]
        self.bad_input = {'aacgm_n': [100.0, 110.0, 30.0],
                          'aacgm_e': [100.0, 110.0, 30.0],
                          'aacgm_z': [10.0, 10.0, 3.0]}

        with self.assertRaisesRegex(ValueError,
                                    "Mismatched OCB and Vector input shapes"):
            self.vdata = ocbpy.ocb_scaling.VectorData(*self.input_attrs,
                                                      **self.bad_input)

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_init_vector_failure(self):
        """ Test init failure with a bad mix of vector and scalar input
        """
        self.input_attrs = [[0, self.ocb.rec_ind, [75.0, 70.0], [22.0, 20.0]],
                            [[0, 1], self.ocb.rec_ind, [75.0, 70.0], 22.0],
                            [[0, 1], self.ocb.rec_ind, [75.0, 70.0],
                             [22.0, 20.0, 23.0]]]
        self.bad_input = [{'aacgm_n': 10.0},
                          {'aacgm_n': [100.0, 110.0, 30.0]},
                          {'aacgm_n': [100.0, 110.0, 30.0]}]
        self.raise_out = ['data index shape must match vector shape',
                          'mismatched VectorData input shapes',
                          'mismatched VectorData input shapes']

        for i, iattrs in enumerate(self.input_attrs):
            tset = [iattrs, self.bad_input[i], self.raise_out[i]]
            with self.subTest(tset=tset):
                with self.assertRaisesRegex(ValueError, tset[2]):
                    self.vdata = ocbpy.ocb_scaling.VectorData(*tset[0],
                                                              **tset[1])

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_init_vector_failure_dat_ind(self):
        """ Test init failure with a bad data index shape
        """
        self.input_attrs = [0, self.ocb.rec_ind, [75.0, 70.0], [22.0, 20.0]]
        self.bad_input = {'aacgm_n': 10.0}

        with self.assertRaisesRegex(
                ValueError, "data index shape must match vector shape"):
            self.vdata = ocbpy.ocb_scaling.VectorData(*self.input_attrs,
                                                      **self.bad_input)

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_init_vector_failure_many_array_size(self):
        """ Test init failure with a bad vector lengths
        """
        self.input_attrs = [[0, 1], self.ocb.rec_ind, [75.0, 70.0], 20.0]
        self.bad_input = {'aacgm_n': [100.0, 110.0, 30.0]}

        with self.assertRaisesRegex(ValueError,
                                    "mismatched VectorData input shapes"):
            self.vdata = ocbpy.ocb_scaling.VectorData(*self.input_attrs,
                                                      **self.bad_input)

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_init_vector_failure_bad_lat_array_size(self):
        """ Test init failure with a bad vector lengths
        """
        self.input_attrs = [[0, 1, 3], self.ocb.rec_ind, [75.0, 70.0],
                            [22.0, 20.0, 23.0]]
        self.bad_input = {'aacgm_n': [100.0, 110.0, 30.0]}

        with self.assertRaisesRegex(ValueError,
                                    "mismatched VectorData input shapes"):
            self.vdata = ocbpy.ocb_scaling.VectorData(*self.input_attrs,
                                                      **self.bad_input)

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_bad_calc_vec_pole_angle(self):
        """Test calc_vec_pole_angle failure with bad input"""
        self.input_attrs = ['aacgm_mlt', 'ocb_aacgm_mlt', 'aacgm_lat',
                            'ocb_aacgm_lat']
        self.raise_out = ["AACGM MLT of Vector", "AACGM MLT of OCB pole",
                          "AACGM latitude of Vector",
                          "AACGM latitude of OCB pole"]
        tsets = [(iattrs, bi, self.raise_out[i])
                 for i, iattrs in enumerate(self.input_attrs)
                 for bi in self.bad_input]
        self.vdata.set_ocb(self.ocb, None)

        for tset in tsets:
            with self.subTest(tset=tset):
                self.hold_val = getattr(self.vdata, tset[0])
                setattr(self.vdata, tset[0], tset[1])

                with self.assertRaisesRegex(ValueError, tset[2]):
                    self.vdata.calc_vec_pole_angle()

                setattr(self.vdata, tset[0], self.hold_val)

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_bad_calc_vec_pole_angle_mlt_float(self):
        """Test calc_vec_pole_angle failure with bad AACGM MLT
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_mlt = self.bad_input[0]

        with self.assertRaisesRegex(ValueError, "AACGM MLT of Vector"):
            self.vdata.calc_vec_pole_angle()

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_bad_calc_vec_pole_angle_ocb_mlt_float_small(self):
        """Test calc_vec_pole_angle failure with small bad OCB MLT
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_mlt = self.bad_input[0]

        with self.assertRaisesRegex(ValueError, "AACGM MLT of OCB pole"):
            self.vdata.calc_vec_pole_angle()

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_bad_calc_vec_pole_angle_vec_mlat_float(self):
        """Test calc_vec_pole_angle failure with bad vector latitude
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_lat = self.bad_input[0]

        with self.assertRaisesRegex(ValueError, "AACGM latitude of Vector"):
            self.vdata.calc_vec_pole_angle()

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_bad_calc_vec_pole_angle_mlt_array(self):
        """Test calc_vec_pole_angle failure with bad AACGM MLT
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_mlt = self.bad_input[1]

        with self.assertRaisesRegex(ValueError, "AACGM MLT of Vector"):
            self.vdata.calc_vec_pole_angle()

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_bad_calc_vec_pole_angle_ocb_mlt_float_big(self):
        """Test calc_vec_pole_angle failure with bad OCB MLT
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_mlt = self.bad_input[1]

        with self.assertRaisesRegex(ValueError, "AACGM MLT of OCB pole"):
            self.vdata.calc_vec_pole_angle()

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_bad_calc_vec_pole_angle_ocb_mlat_float_small(self):
        """Test calc_vec_pole_angle failure with barely bad OCB latitude
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_lat = self.bad_input[0]

        with self.assertRaisesRegex(ValueError, "AACGM latitude of OCB pole"):
            self.vdata.calc_vec_pole_angle()

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_bad_calc_vec_pole_angle_ocb_mlat_float_big(self):
        """Test calc_vec_pole_angle failure with bad OCB latitude
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_lat = self.bad_input[1]

        with self.assertRaisesRegex(ValueError, "AACGM latitude of OCB pole"):
            self.vdata.calc_vec_pole_angle()

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_bad_calc_vec_aacgm_lat_float(self):
        """Test calc_vec_pole_angle failure with bad vector latitude
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_lat = self.bad_input[1]

        with self.assertRaisesRegex(ValueError, "AACGM latitude of Vector"):
            self.vdata.calc_vec_pole_angle()

    def test_no_ocb_lat(self):
        """ Test failure when OCB latitude is not available
        """
        self.vdata.ocb_lat = np.nan

        with self.assertRaisesRegex(ValueError, 'OCB coordinates required'):
            self.vdata.scale_vector()

    def test_no_ocb_mlt(self):
        """ Test failure when OCB MLT is not available
        """
        self.vdata.ocb_mlt = np.nan

        with self.assertRaisesRegex(ValueError, 'OCB coordinates required'):
            self.vdata.scale_vector()

    def test_no_ocb_pole_location(self):
        """ Test failure when OCB pole location is not available
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_mlt = np.nan

        with self.assertRaisesRegex(ValueError, "OCB pole location required"):
            self.vdata.scale_vector()

    def test_no_ocb_pole_angle(self):
        """ Test failure when pole angle is not available
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = np.nan

        with self.assertRaisesRegex(
                ValueError, "vector angle in poles-vector triangle required"):
            self.vdata.scale_vector()

    def test_bad_ocb_quad(self):
        """ Test failure when OCB quadrant is wrong
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_quad = -1

        with self.assertRaisesRegex(ValueError, "OCB quadrant undefined"):
            self.vdata.calc_ocb_polar_angle()

    def test_bad_vec_quad(self):
        """ Test failure when vector quadrant is wrong
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.vec_quad = -1

        with self.assertRaisesRegex(ValueError, "Vector quadrant undefined"):
            self.vdata.calc_ocb_polar_angle()

    def test_bad_quad_polar_angle(self):
        """ Test failure when quadrant polar angle is bad
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_naz = np.nan

        with self.assertRaisesRegex(ValueError,
                                    "AACGM polar angle undefined"):
            self.vdata.calc_ocb_polar_angle()

    def test_bad_quad_vector_angle(self):
        """ Test failure when quandrant vector angle is bad
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = np.nan

        with self.assertRaisesRegex(ValueError, "Vector angle undefined"):
            self.vdata.calc_ocb_polar_angle()

    def test_bad_calc_vec_sign_direction(self):
        """ Test calc_vec_sign failure when no direction is provided
        """
        self.vdata.set_ocb(self.ocb, None)

        with self.assertRaisesRegex(ValueError,
                                    "must set at least one direction"):
            self.vdata.calc_ocb_vec_sign()

    def test_bad_calc_sign_ocb_quad(self):
        """ Test calc_vec_sign failure with bad OCB quadrant
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_quad = -1

        with self.assertRaisesRegex(ValueError, "OCB quadrant undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)

    def test_bad_calc_sign_vec_quad(self):
        """ Test calc_vec_sign failure with bad vector quadrant
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.vec_quad = -1

        with self.assertRaisesRegex(ValueError, "Vector quadrant undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)

    def test_bad_calc_sign_polar_angle(self):
        """ Test calc_vec_sign failure with bad polar angle
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_naz = np.nan

        with self.assertRaisesRegex(ValueError,
                                    "AACGM polar angle undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)

    def test_bad_calc_sign_pole_angle(self):
        """ Test calc_vec_sign failure with bad pole angle
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = np.nan

        with self.assertRaisesRegex(ValueError, "Vector angle undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)

    def test_bad_define_quandrants_pole_mlt(self):
        """Test define_quadrants failure with bad pole MLT
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_mlt = np.nan

        with self.assertRaisesRegex(ValueError, "OCB pole location required"):
            self.vdata.define_quadrants()

    def test_bad_define_quandrants_vec_mlt(self):
        """Test define_quadrants failure with bad vector MLT
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_mlt = np.nan

        with self.assertRaisesRegex(ValueError,
                                    "Vector AACGM location required"):
            self.vdata.define_quadrants()

    def test_bad_define_quandrants_pole_angle(self):
        """Test define_quadrants failure with bad pole angle
        """
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = np.nan

        with self.assertRaisesRegex(
                ValueError, "vector angle in poles-vector triangle required"):
            self.vdata.define_quadrants()


class TestHaversine(unittest.TestCase):

    def setUp(self):
        """ Initialize the tests for the haversine and archaversine functions
        """
        self.input_angles = np.linspace(-2.0*np.pi, 2.0*np.pi, 9)
        self.hav_out = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0])
        # archaversine is confinded to 0-pi
        self.ahav_out = abs(np.array([aa - np.sign(aa) * 2.0 * np.pi
                                      if abs(aa) > np.pi
                                      else aa for aa in self.input_angles]))
        self.out = None

    def tearDown(self):
        del self.input_angles, self.hav_out, self.out, self.ahav_out

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
                    self.assertTrue(np.all(abs(self.out - tset[1]) < 1.0e-7))

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
        self.assertTrue(np.all(abs(self.out - self.hav_out) < 1.0e-7))

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_haversine_array(self):
        """ Test implimentation of the haversine with a array input
        """
        self.out = ocbpy.ocb_scaling.hav(self.input_angles)
        self.assertTrue(np.all(abs(self.out - self.hav_out) < 1.0e-7))

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_inverse_haversine(self):
        """ Test the implemenation of the inverse haversine
        """
        # Cycle through all the possible input options
        for i, tset in enumerate([(self.hav_out[0], self.ahav_out[0]),
                                  (list(self.hav_out), self.ahav_out),
                                  (self.hav_out, self.ahav_out)]):
            with self.subTest(tset=tset):
                self.out = ocbpy.ocb_scaling.archav(tset[0])

                # Assess the output
                if i == 0:
                    self.assertEqual(self.out, abs(tset[1]))
                else:
                    self.assertTrue(np.all(self.out - tset[1] < 1.0e-7))

        del tset

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_inverse_haversine_float(self):
        """ Test implimentation of the inverse haversine with float input
        """
        for i, self.out in enumerate(self.ahav_out):
            self.assertAlmostEqual(ocbpy.ocb_scaling.archav(self.hav_out[i]),
                                   self.out)

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_inverse_haversine_list(self):
        """ Test implimentation of the inverse haversine with list input
        """
        self.out = ocbpy.ocb_scaling.archav(list(self.hav_out))
        self.assertTrue(np.all(abs(self.out - self.ahav_out)
                               < 1.0e-7))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_inverse_haversine_array(self):
        """ Test implimentation of the inverse haversine with array input
        """
        self.out = ocbpy.ocb_scaling.archav(self.hav_out)
        self.assertTrue(np.all(abs(self.out - self.ahav_out) < 1.0e-7))

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
        self.hav_out[0] = 1.0e-17
        self.ahav_out[0] = 0.0
        self.hav_out[1] = np.nan
        self.ahav_out[1] = np.nan
        self.hav_out[2] = -1.0
        self.ahav_out[2] = np.nan

        self.out = ocbpy.ocb_scaling.archav(self.hav_out)

        for i, hout in enumerate(self.out):
            if np.isnan(hout):
                self.assertTrue(np.isnan(self.ahav_out[i]))
            else:
                self.assertAlmostEqual(hout, self.ahav_out[i])


class TestOCBScalingArrayMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary and array VectorData objects
        """
        test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                              "test_data", "test_north_circle")
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=test_file,
                                               instrument='image')
        self.vargs = [[3, 6], 27, np.array([75.0, 87.2]),
                      np.array([22.0, 21.22])]
        self.vkwargs = {'aacgm_n': np.array([50.0, 0.0]),
                        'aacgm_e': np.array([86.5, 0.0]),
                        'aacgm_z': np.array([5.0, 0.0]),
                        'dat_name': 'Test', 'dat_units': 'm/s'}
        self.vdata = None
        self.out = None

        self.aacgm_mag = [100.03624343, 0.0]

        if version_info.major == 2:
            self.assertRegex = self.assertRegexpMatches
            self.assertNotRegex = self.assertNotRegexpMatches

    def tearDown(self):
        del self.ocb, self.vargs, self.vkwargs, self.out, self.vdata
        del self.aacgm_mag

    def test_array_vector_repr_not_calc(self):
        """ Test the VectorData print statement with uncalculated array input
        """
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.out = self.vdata.__repr__()
        self.assertRegex(self.out, "Index")
        self.assertRegex(self.out, "nan, nan, {:d}".format(self.vargs[1]))

    def test_array_vector_repr_calc(self):
        """ Test the VectorData print statement with calculated array input
        """
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)
        self.out = self.vdata.__repr__()
        self.assertRegex(self.out, "Index")
        self.assertNotRegex(self.out, "nan, nan")

    def test_array_vector_repr_calc_ocb_vec_array(self):
        """ Test the VectorData print statement with calculated ocb/vec arrays
        """
        self.vargs[1] = [27, 31]
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)
        self.out = self.vdata.__repr__()
        self.assertRegex(self.out, "Index")
        self.assertNotRegex(self.out, "nan, nan")

    def test_array_vector_repr_calc_ocb_array(self):
        """ Test the VectorData print statement with calculated ocb arrays
        """
        self.vargs[0] = self.vargs[0][0]
        self.vargs[1] = [27, 31]
        self.vargs[2] = self.vargs[2][0]
        self.vargs[3] = self.vargs[3][0]
        self.vkwargs['aacgm_n'] = self.vkwargs['aacgm_n'][0]
        self.vkwargs['aacgm_e'] = self.vkwargs['aacgm_e'][0]
        self.vkwargs['aacgm_z'] = self.vkwargs['aacgm_z'][0]
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)
        self.out = self.vdata.__repr__()
        self.assertRegex(self.out, "Index")
        self.assertNotRegex(self.out, "nan, nan")

    def test_init_nez_vec_array(self):
        """ Test VectorData initialisation  with vector array components
        """
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.assertEqual(len(self.vdata.aacgm_mag), len(self.vargs[0]))
        self.assertEqual(len(self.vdata.ocb_mag), len(self.vargs[0]))
        for i, self.out in enumerate(self.vdata.aacgm_mag):
            self.assertAlmostEqual(self.out, self.aacgm_mag[i])

    def test_init_nez_ocb_vec_array(self):
        """ Test VectorData initialisation with ocb and vector array components
        """
        self.vargs[1] = [27, 31]
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.assertEqual(len(self.vdata.aacgm_mag), len(self.vargs[0]))
        self.assertEqual(len(self.vdata.ocb_mag), len(self.vargs[0]))
        for i, self.out in enumerate(self.vdata.aacgm_mag):
            self.assertAlmostEqual(self.out, self.aacgm_mag[i])

    def test_init_nez_ocb_array(self):
        """ Test VectorData initialisation with ocb array components
        """
        self.vargs[0] = self.vargs[0][0]
        self.vargs[1] = [27, 31]
        self.vargs[2] = self.vargs[2][0]
        self.vargs[3] = self.vargs[3][0]
        self.vkwargs['aacgm_n'] = self.vkwargs['aacgm_n'][0]
        self.vkwargs['aacgm_e'] = self.vkwargs['aacgm_e'][0]
        self.vkwargs['aacgm_z'] = self.vkwargs['aacgm_z'][0]

        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.assertEqual(len(self.vdata.ocb_mag), len(self.vargs[1]))
        self.assertAlmostEqual(self.vdata.aacgm_mag, self.aacgm_mag[0])

    def test_init_mag(self):
        """ Test the initialisation of the VectorData array input with magnitude
        """
        self.vkwargs['aacgm_mag'] = self.aacgm_mag
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.assertEqual(len(self.vdata.aacgm_mag), len(self.vargs[0]))
        for i, self.out in enumerate(self.vdata.aacgm_mag):
            self.assertAlmostEqual(self.out, self.aacgm_mag[i])

    def test_vector_all_bad_lat(self):
        """ Test the VectorData output with all data from the wrong hemisphere
        """
        self.vargs[2] *= -1.0
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertTrue(len(self.vdata.ocb_lat), len(self.vargs[2]))
        self.assertTrue(np.all(np.isnan(self.vdata.ocb_lat)))
        self.assertTrue(np.all(np.isnan(self.vdata.ocb_mlt)))
        self.assertTrue(np.all(np.isnan(self.vdata.ocb_n)))
        self.assertTrue(np.all(np.isnan(self.vdata.ocb_e)))
        self.assertTrue(np.all(np.isnan(self.vdata.ocb_z)))

        # Ensure that input is not overwritten
        for vkey in self.vkwargs.keys():
            self.out = getattr(self.vdata, vkey)
            if vkey.find('aacgm_') == 0:
                for i, val in enumerate(self.vkwargs[vkey]):
                    if np.isnan(val):
                        self.assertTrue(np.isnan(self.out[i]))
                    else:
                        self.assertEqual(self.out[i], val)
            else:
                self.assertRegex(self.out, self.vkwargs[vkey])

    def test_vector_some_bad_lat(self):
        """ Test the VectorData output with mixed hemisphere input
        """
        self.vargs[2][0] *= -1.0
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertTrue(len(self.vdata.ocb_lat), len(self.vargs[2]))

        # Ensure the wrong hemisphere is NaN
        self.assertTrue(np.isnan(self.vdata.ocb_lat[0]))
        self.assertTrue(np.isnan(self.vdata.ocb_mlt[0]))
        self.assertTrue(np.isnan(self.vdata.ocb_n[0]))
        self.assertTrue(np.isnan(self.vdata.ocb_e[0]))
        self.assertTrue(np.isnan(self.vdata.ocb_z[0]))

        # Ensure that input is not overwritten
        for vkey in self.vkwargs.keys():
            self.out = getattr(self.vdata, vkey)
            if vkey.find('aacgm_') == 0:
                for i, val in enumerate(self.vkwargs[vkey]):
                    if np.isnan(val):
                        self.assertTrue(np.isnan(self.out[i]))
                    else:
                        self.assertEqual(self.out[i], val)
            else:
                self.assertRegex(self.out, self.vkwargs[vkey])

        # Ensure the right hemisphere is good
        self.assertAlmostEqual(self.vdata.aacgm_mag[1], self.aacgm_mag[1])
        self.assertAlmostEqual(self.vdata.ocb_mag[1], self.aacgm_mag[1])

    def test_calc_one_large_pole_angle(self):
        """ Test the OCB polar angle calculation with one angle > 90 deg
        """
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.vdata.ocb_aacgm_mlt = np.asarray(1.260677777)
        self.vdata.ocb_aacgm_lat = np.asarray(83.99)
        self.vdata.ocb_lat[1] = 84.838777192
        self.vdata.ocb_mlt[1] = 15.1110383783

        self.vdata.calc_vec_pole_angle()
        self.assertAlmostEqual(self.vdata.pole_angle[0], 22.45577128)
        self.assertAlmostEqual(self.vdata.pole_angle[1], 91.72024697)

    def test_calc_vec_pole_angle_flat(self):
        """ Test the polar angle calculation with angles of 0 and 180 deg
        """
        self.vargs[3] = np.array([6.0, 6.0])
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.vdata.ocb_aacgm_mlt = np.asarray(6.0)
        self.vdata.aacgm_lat = np.array([45.0 + 0.5 * self.vdata.ocb_aacgm_lat,
                                         self.vdata.ocb_aacgm_lat - 0.5])
        self.vdata.calc_vec_pole_angle()
        self.assertEqual(self.vdata.pole_angle[0], 0.0)
        self.assertEqual(self.vdata.pole_angle[1], 180.0)

    def test_array_vec_quad(self):
        """ Test the assignment of vector quadrants with array input
        """
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertEqual(len(self.vargs[0]), len(self.vdata.vec_quad))
        self.assertTrue(np.all(self.vdata.vec_quad == 1.0))

    def test_array_ocb_quad(self):
        """ Test the assignment of OCB quadrants with array input
        """
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        print(self.vdata.ocb_quad)
        self.assertEqual(len(self.vargs[0]), len(self.vdata.ocb_quad))
        self.assertTrue(np.all(self.vdata.ocb_quad == 1.0))

    def test_one_undefinable_ocb_quadrant(self):
        """ Test VectorData array initialization for a undefinable OCB quadrant
        """
        self.vargs[2][1] = 0.0
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertEqual(self.vdata.ocb_quad[0], 1)
        self.assertEqual(self.vdata.ocb_quad[1], 0)

    def test_one_undefinable_vec_quadrant(self):
        """ Test VectorData array initialization for a undefinable vec quadrant
        """
        self.vkwargs['aacgm_n'][1] = np.nan
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertEqual(self.vdata.vec_quad[0], 1)
        self.assertEqual(self.vdata.vec_quad[1], 0)

    def test_define_quadrants_neg_adj_mlt(self):
        """ Test the quadrant assignment with a negative AACGM MLT
        """
        self.vargs[3][0] = -22.0
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)

        self.assertGreater(self.vdata.ocb_aacgm_mlt-self.vargs[3][0], 24)
        self.assertTrue(np.all(self.vdata.ocb_quad == 1))
        self.assertTrue(np.all(self.vdata.vec_quad == 1))

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_scale_vec_pole_angle_zero(self):
        """ Test the calculation of the OCB vector sign with no pole angle
        """
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)
        self.vdata.pole_angle = np.zeros(shape=self.vargs[2].shape)

        nscale = ocbpy.ocb_scaling.normal_evar(self.vdata.aacgm_n,
                                               self.vdata.unscaled_r,
                                               self.vdata.scaled_r)
        escale = ocbpy.ocb_scaling.normal_evar(self.vdata.aacgm_e,
                                               self.vdata.unscaled_r,
                                               self.vdata.scaled_r)

        # Cycle through all the possible options for a pole angle of zero/180
        for tset in [('scale_func', None, self.vkwargs['aacgm_n'],
                      self.vkwargs['aacgm_e']),
                     ('scale_func', ocbpy.ocb_scaling.normal_evar, nscale,
                      escale),
                     ('ocb_aacgm_lat', self.vargs[2][0], -1.0 * nscale,
                      -1.0 * escale)]:
            with self.subTest(tset=tset):
                setattr(self.vdata, tset[0], tset[1])

                # Run the scale_vector routine with the new attributes
                self.vdata.scale_vector()

                # Assess the ocb north and east components
                self.assertTrue(np.all(self.vdata.ocb_n == tset[2]))
                self.assertTrue(np.all(self.vdata.ocb_e == tset[3]))

        del nscale, escale, tset

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_scale_vec_pole_angle_zero_noscale(self):
        """ Test the OCB vector sign calc with no pole angle or scaling
        """
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)
        self.vdata.pole_angle = np.zeros(shape=self.vargs[2].shape)

        # Run the scale_vector routine with the new attributes
        self.vdata.scale_vector()

        # Assess the ocb north and east components
        self.assertTrue(np.all(self.vdata.ocb_n == self.vkwargs['aacgm_n']))
        self.assertTrue(np.all(self.vdata.ocb_e == self.vkwargs['aacgm_e']))

    @unittest.skipIf(version_info.major > 2, 'Already tested with subTest')
    def test_scale_vec_pole_angle_zero_scale(self):
        """  Test the OCB vector sign calc with scaling but no pole angle
        """
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.vdata.pole_angle = np.zeros(shape=self.vargs[2].shape)

        # Run the scale_vector routine with the new attributes
        self.vdata.scale_vector()

        # Assess the ocb north and east components
        self.out = ocbpy.ocb_scaling.normal_evar(self.vkwargs['aacgm_n'],
                                                 self.vdata.unscaled_r,
                                                 self.vdata.scaled_r)
        self.assertTrue(np.all(self.vdata.ocb_n == self.out))

        self.out = ocbpy.ocb_scaling.normal_evar(self.vkwargs['aacgm_e'],
                                                 self.vdata.unscaled_r,
                                                 self.vdata.scaled_r)
        self.assertTrue(np.all(self.vdata.ocb_e == self.out))
