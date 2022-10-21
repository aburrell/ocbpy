#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the ocb_scaling class and functions."""

from io import StringIO
import logging
import numpy
from numpy import nan
from os import path
import unittest

import ocbpy


class TestOCBScalingLogFailure(unittest.TestCase):
    """Unit tests for logging messages in ocb_scaling module."""

    def setUp(self):
        """Initialize the test class."""
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
        self.ocb = ocbpy.OCBoundary(filename=test_file, instrument='image')
        self.ocb.rec_ind = 27
        self.vdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind, 75.0,
                                                  22.0, aacgm_n=50.0,
                                                  aacgm_e=86.5, aacgm_z=5.0,
                                                  dat_name="Test",
                                                  dat_units="$m s^{-1}$")
        return

    def tearDown(self):
        """Tear down the test case."""
        del self.lwarn, self.lout, self.log_capture, self.ocb, self.vdata
        return

    def test_no_scale_func(self):
        """Test OCBScaling initialization with no scaling function."""
        self.lwarn = u"no scaling function provided"

        # Initialize the VectorData class without a scaling function
        self.vdata.set_ocb(self.ocb)
        self.assertIsNone(self.vdata.scale_func)

        self.lout = self.log_capture.getvalue()
        # Test logging error message for each bad initialization
        self.assertTrue(self.lout.find(self.lwarn) >= 0)
        return

    def test_inconsistent_vector_warning(self):
        """Test init failure with inconsistent AACGM components."""
        self.lwarn = u"inconsistent AACGM"

        # Initalize the VectorData class with inconsistent vector magnitudes
        self.vdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind,
                                                  75.0, 22.0,
                                                  aacgm_mag=100.0,
                                                  dat_name="Test",
                                                  dat_units="$m s^{-1}$")

        self.lout = self.log_capture.getvalue()
        # Test logging error message for each bad initialization
        self.assertTrue(self.lout.find(self.lwarn) >= 0)
        return


class TestOCBScalingMethods(unittest.TestCase):
    """Unit tests for the VectorData class."""

    def setUp(self):
        """Initialize the OCBoundary and VectorData objects."""

        test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                              "test_data", "test_north_circle")
        self.assertTrue(path.isfile(test_file))
        self.ocb = ocbpy.OCBoundary(filename=test_file, instrument='image')
        self.ocb.rec_ind = 27
        self.ocb_attrs = ['ocb_lat', 'ocb_mlt', 'r_corr', 'ocb_n', 'ocb_e',
                          'ocb_z']

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
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.ocb, self.ocb_attrs, self.vdata, self.wdata, self.zdata
        return

    def test_init_nez(self):
        """Test the set up of the VectorData object without magnitude."""
        self.assertAlmostEqual(self.vdata.aacgm_mag, 100.036243432)
        self.assertAlmostEqual(self.zdata.aacgm_mag, 0.0)
        return

    def test_init_mag(self):
        """Test the initialisation of the VectorData object with magnitude."""
        self.assertAlmostEqual(self.wdata.aacgm_mag, 100.036243432)
        return

    def test_repr_string(self):
        """Test __repr__ method string. """
        for val in [None, ocbpy.ocb_correction.circular]:
            with self.subTest(val=val):
                self.vdata.scale_func = val
                self.assertRegex(repr(self.vdata),
                                 "ocbpy.ocb_scaling.VectorData")

                if val is not None:
                    self.assertRegex(repr(self.vdata), val.__name__)
        return

    def test_repr_eval(self):
        """Test __repr__ method's ability to reproduce a class."""
        self.wdata = eval(repr(self.vdata))
        self.assertEqual(repr(self.wdata), repr(self.vdata))
        return

    def test_vector_str_no_scaling(self):
        """Test the VectorData print statement without a scaling function."""
        self.assertRegex(str(self.vdata), "Vector data:")
        self.assertRegex(str(self.vdata), "No magnitude scaling function")
        return

    def test_vector_str_with_scaling(self):
        """Test the VectorData print statement with a scaling function."""
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)

        self.assertRegex(str(self.vdata), "Vector data:")
        self.assertRegex(str(self.vdata), "Scaling function")
        return

    def test_vector_mult_ocb_ind(self):
        """Test the VectorData performance with multiple OCB indices."""
        # Update the VectorData attribute to contain a list of all good indices
        self.vdata.ocb_ind = ocbpy.cycle_boundary.retrieve_all_good_indices(
            self.ocb)

        # Set the VectorData OCB attributes
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)

        # Evaluate values are realistic and appropriately shaped
        for attr in self.ocb_attrs:
            with self.subTest(attr=attr):
                val = getattr(self.vdata, attr)
                self.assertTupleEqual(self.vdata.ocb_ind.shape, val.shape)
                self.assertTrue(numpy.isfinite(val).all())
        return

    def test_vector_mult_dat_ind(self):
        """Test the VectorData performance with multiple OCB indices."""
        # Update the VectorData attribute to contain a list of all good indices
        test_shape = (3,)
        self.vdata.aacgm_lat = numpy.full(shape=test_shape,
                                          fill_value=self.vdata.aacgm_lat)
        self.vdata.aacgm_mlt = numpy.full(shape=test_shape,
                                          fill_value=self.vdata.aacgm_mlt)
        self.vdata.dat_ind = [i * 2 for i in range(test_shape[0])]

        # Set the VectorData OCB attributes
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)

        # Evaluate values are realistic and appropriately shaped
        for attr in self.ocb_attrs:
            with self.subTest(attr=attr):
                val = getattr(self.vdata, attr)
                self.assertTupleEqual(val.shape, test_shape)
                self.assertTrue(numpy.isfinite(val).all())
        return

    def test_vector_clear_data(self):
        """Test the VectorData.clear_data method."""
        # Set the VectorData OCB attributes
        self.vdata.set_ocb(self.ocb)

        # Evaluate values are realistic
        for attr in self.ocb_attrs:
            with self.subTest(attr=attr):
                val = getattr(self.vdata, attr)
                self.assertTrue(numpy.isfinite(val).all())

        # Clear data
        self.vdata.clear_data()

        # Evaluate values are NaN
        self.ocb_attrs = ['ocb_n', 'ocb_e', 'ocb_z', 'ocb_mag', 'pole_angle',
                          'aacgm_naz', 'ocb_aacgm_lat', 'ocb_aacgm_mlt']
        for attr in self.ocb_attrs:
            with self.subTest(attr=attr):
                val = getattr(self.vdata, attr)
                self.assertTrue(numpy.isnan(val).all(),
                                msg="{:} is not NaN".format(val))

        # Evaluate values are zero
        self.ocb_attrs = ['ocb_quad', 'vec_quad']
        for attr in self.ocb_attrs:
            with self.subTest(attr=attr):
                val = getattr(self.vdata, attr)
                self.assertEqual(val, 0)
        return

    def test_vector_bad_lat(self):
        """Test the VectorData output with data from the wrong hemisphere."""
        self.vdata.aacgm_lat *= -1.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)

        for attr in self.ocb_attrs:
            with self.subTest(attr=attr):
                val = getattr(self.vdata, attr)
                self.assertTrue(numpy.isnan(val),
                                msg="{:} is not NaN".format(val))
        return

    def test_calc_large_pole_angle(self):
        """Test the OCB polar angle calculation with angles > 90 deg."""
        self.zdata.ocb_aacgm_mlt = 1.260677777
        self.zdata.ocb_aacgm_lat = 83.99
        self.zdata.ocb_lat = 84.838777192
        self.zdata.ocb_mlt = 15.1110383783

        self.zdata.calc_vec_pole_angle()
        self.assertAlmostEqual(self.zdata.pole_angle, 91.72024697182087)
        return

    def test_calc_polar_angle_ocb_south_night(self):
        """Test `calc_polar_angle` with the OCB pole in a south/night quad."""
        # Set a useful vector locaiton and intialise with current boundary
        self.vdata.aacgm_mlt = 0.0
        self.vdata.aacgm_n = -10.0
        self.vdata.aacgm_e = -10.0
        self.vdata.set_ocb(self.ocb)

        # Change the location of the boundary center
        self.vdata.ocb_aacgm_mlt = 1.0
        self.vdata.ocb_aacgm_lat = self.vdata.aacgm_lat - 2.0

        # Update the quandrants
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()

        # Get the polar angle
        self.assertAlmostEqual(self.vdata.calc_ocb_polar_angle(), 116.52904962)
        return

    def test_calc_polar_angle_ocb_south_day(self):
        """Test `calc_polar_angle` with the OCB pole in a south/day quad."""
        # Set a useful vector locaiton and intialise with current boundary
        self.vdata.aacgm_mlt = 0.0
        self.vdata.set_ocb(self.ocb)

        # Change the location of the boundary center
        self.vdata.ocb_aacgm_mlt = 1.0
        self.vdata.ocb_aacgm_lat = self.vdata.aacgm_lat - 2.0

        # Update the quandrants
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()

        # Get the polar angle
        self.assertAlmostEqual(self.vdata.calc_ocb_polar_angle(), 48.500352141)
        return

    def test_big_pole_angle_mlt_west(self):
        """Test `calc_ocb_polar_angle` with a neg MLT, W vect, and big angle."""
        # Get the original angle
        self.vdata.aacgm_mlt = -22.0
        self.vdata.aacgm_e *= -1.0
        self.vdata.set_ocb(self.ocb)

        # Increase the pole angle enough to require an adjustment
        self.vdata.pole_angle += 90.0
        self.assertAlmostEqual(self.vdata.calc_ocb_polar_angle(), 159.83429474)
        return

    def test_calc_vec_pole_angle_acute(self):
        """Test the polar angle calculation with an acute angle."""
        self.vdata.set_ocb(self.ocb)
        self.assertAlmostEqual(self.vdata.pole_angle, 8.67527923)
        return

    def test_calc_vec_pole_angle_zero(self):
        """Test the polar angle calculation with an angle of zero."""
        self.vdata.set_ocb(self.ocb)
        self.vdata.aacgm_mlt = self.vdata.ocb_aacgm_mlt
        self.vdata.calc_vec_pole_angle()
        self.assertEqual(self.vdata.pole_angle, 0.0)
        return

    def test_calc_vec_pole_angle_flat(self):
        """Test the polar angle calculation with an angle of 180 deg."""
        self.vdata.set_ocb(self.ocb)
        self.vdata.ocb_aacgm_mlt = 6.0
        self.vdata.aacgm_mlt = 6.0
        self.vdata.aacgm_lat = 45.0 + 0.5 * self.vdata.ocb_aacgm_lat
        self.vdata.calc_vec_pole_angle()
        self.assertEqual(self.vdata.pole_angle, 180.0)
        return

    def test_calc_vec_pole_angle_right_isosceles(self):
        """Test the polar angle calculation with a right isosceles triangle."""
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
        return

    def test_calc_vec_pole_angle_oblique(self):
        """Test the polar angle calculation with an isosceles triangle."""
        self.vdata.set_ocb(self.ocb)
        self.vdata.aacgm_mlt = self.vdata.ocb_aacgm_mlt - 1.0
        self.vdata.aacgm_lat = 45.0 + 0.5 * self.vdata.ocb_aacgm_lat
        self.vdata.calc_vec_pole_angle()
        self.assertAlmostEqual(self.vdata.pole_angle, 150.9561733411)
        return

    def test_define_quadrants(self):
        """Test the assignment of quadrants."""
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
        return

    def test_define_quadrants_neg_adj_mlt_west(self):
        """Test quadrant assignment with a negative AACGM MLT and W vect."""
        self.vdata.aacgm_mlt = -22.0
        self.vdata.aacgm_e *= -1.0
        self.vdata.set_ocb(self.ocb)
        self.assertGreater(self.vdata.ocb_aacgm_mlt - self.vdata.aacgm_mlt, 24)
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 2)
        return

    def test_define_quadrants_neg_north(self):
        """Test the quadrant assignment with a vector pointing south."""
        # Adjust the vector quadrant
        self.vdata.aacgm_n *= -1.0
        self.vdata.set_ocb(self.ocb)

        # Evaluate the output quadrants
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 4)
        return

    def test_define_quadrants_noon_north(self):
        """Test quadrant assignment with a vector pointing north from noon."""
        self.vdata.aacgm_mlt = 12.0
        self.vdata.set_ocb(self.ocb)
        self.assertEqual(self.vdata.ocb_quad, 2)
        self.assertEqual(self.vdata.vec_quad, 1)
        return

    def test_define_quadrants_aligned_poles_southwest(self):
        """Test quad assignment w/vector pointing SW and both poles aligned."""
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.vdata.aacgm_mlt = self.vdata.ocb_aacgm_mlt + 12.0
        self.vdata.aacgm_n = -10.0
        self.vdata.aacgm_e = -10.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 2)
        self.assertEqual(self.vdata.vec_quad, 3)
        return

    def test_define_quadrants_ocb_south_night(self):
        """Test quadrant assignment with the OCB pole in a south/night quad."""
        self.vdata.aacgm_mlt = 0.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.vdata.ocb_aacgm_mlt = 23.0
        self.vdata.ocb_aacgm_lat = self.vdata.aacgm_lat - 2.0
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()
        self.assertEqual(self.vdata.ocb_quad, 3)
        self.assertEqual(self.vdata.vec_quad, 1)
        return

    def test_define_quadrants_ocb_south_day(self):
        """Test quadrant assignment with the OCB pole in a south/day quad."""
        self.vdata.aacgm_mlt = 0.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.vdata.ocb_aacgm_mlt = 1.0
        self.vdata.ocb_aacgm_lat = self.vdata.aacgm_lat - 2.0
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()
        self.assertEqual(self.vdata.ocb_quad, 4)
        self.assertEqual(self.vdata.vec_quad, 1)
        return

    def test_undefinable_quadrants(self):
        """Test OCBScaling initialization for undefinable quadrants."""
        self.vdata.aacgm_lat = 0.0
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 0)
        self.assertEqual(self.vdata.vec_quad, 0)
        return

    def test_lost_ocb_quadrant(self):
        """Test OCBScaling initialization for unset quadrants."""
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 1)
        self.vdata.ocb_quad = 0
        self.vdata.scale_vector()
        self.assertEqual(self.vdata.ocb_quad, 1)
        return

    def test_lost_vec_quadrant(self):
        """Test OCBScaling initialization for unset quadrants."""
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 1)
        self.vdata.vec_quad = 0
        self.vdata.scale_vector()
        self.assertEqual(self.vdata.vec_quad, 1)
        return

    def test_calc_ocb_vec_sign(self):
        """Test the calculation of the OCB vector signs."""

        # Set the initial values
        self.vdata.ocb_aacgm_mlt = self.ocb.phi_cent[self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.r_cent[self.vdata.ocb_ind]
        (self.vdata.ocb_lat, self.vdata.ocb_mlt,
         self.vdata.r_corr) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                    self.vdata.aacgm_mlt)
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()

        vmag = numpy.sqrt(self.vdata.aacgm_n**2 + self.vdata.aacgm_e**2)
        self.vdata.aacgm_naz = numpy.degrees(numpy.arccos(
            self.vdata.aacgm_n / vmag))

        # Calculate the vector data signs
        vsigns = self.vdata.calc_ocb_vec_sign(north=True, east=True)
        self.assertTrue(vsigns['north'])
        self.assertTrue(vsigns['east'])

        return

    def test_scale_vec(self):
        """Test the calculation of the OCB vector signs."""

        # Set the initial values
        self.vdata.ocb_aacgm_mlt = self.ocb.phi_cent[self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.r_cent[self.vdata.ocb_ind]
        (self.vdata.ocb_lat, self.vdata.ocb_mlt,
         self.vdata.r_corr) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                    self.vdata.aacgm_mlt)
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()

        vmag = numpy.sqrt(self.vdata.aacgm_n**2 + self.vdata.aacgm_e**2)
        self.vdata.aacgm_naz = numpy.degrees(numpy.arccos(
            self.vdata.aacgm_n / vmag))

        # Scale the data vector
        self.vdata.scale_vector()

        # Test the North and East components
        self.assertAlmostEqual(self.vdata.ocb_n, 62.4751208491)
        self.assertAlmostEqual(self.vdata.ocb_e, 77.9686428950)

        # Test to see that the magnitudes and z-components are the same
        self.assertAlmostEqual(self.vdata.aacgm_mag, self.vdata.ocb_mag)
        self.assertAlmostEqual(self.vdata.ocb_z, self.vdata.aacgm_z)

        return

    def test_scale_vec_z_zero(self):
        """Test the calc of the OCB vector sign with no vertical aacgm_z."""
        # Re-assing the necessary variable
        self.vdata.aacgm_z = 0.0

        # Run the scale_vector routine
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)

        # Assess the ocb_z component
        self.assertEqual(self.vdata.ocb_z,
                         self.vdata.scale_func(0.0, self.vdata.unscaled_r,
                                               self.vdata.scaled_r))
        return

    def test_scale_vec_pole_angle_zero(self):
        """Test the calculation of the OCB vector sign with no pole angle."""
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

        return

    def test_set_ocb_zero(self):
        """Test setting of OCB values in VectorData without any magnitude."""
        # Set the OCB values without any E-field scaling, test to see that the
        # AACGM and OCB vector magnitudes are the same
        self.zdata.set_ocb(self.ocb)
        self.assertEqual(self.zdata.ocb_mag, 0.0)
        return

    def test_set_ocb_none(self):
        """Test setting of OCB values without scaling."""

        # Set the OCB values without any E-field scaling, test to see that the
        # AACGM and OCB vector magnitudes are the same
        self.vdata.set_ocb(self.ocb)
        self.assertAlmostEqual(self.vdata.aacgm_mag, self.vdata.ocb_mag)
        return

    def test_set_ocb_evar(self):
        """Test setting of OCB values with E field scaling."""

        # Set the OCB values with scaling for a variable proportional to
        # the electric field
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertAlmostEqual(self.vdata.ocb_mag, 88.094416872365)
        return

    def test_set_ocb_curl_evar(self):
        """Test setting of OCB values with Curl E scaling."""
        # Set the OCB values with scaling for a variable proportional to
        # the curl of the electric field
        self.vdata.set_ocb(self.ocb,
                           scale_func=ocbpy.ocb_scaling.normal_curl_evar)
        self.assertAlmostEqual(self.vdata.ocb_mag, 77.57814585822645)
        return

    def test_scaled_r(self):
        """Test that the scaled radius is correct."""
        self.vdata.set_ocb(self.ocb, None)
        self.assertEqual(self.vdata.scaled_r, 16.0)
        return

    def test_unscaled_r(self):
        """Test that the unscaled radius is correct."""
        self.vdata.set_ocb(self.ocb, None)
        self.assertEqual(self.vdata.unscaled_r, 14.09)
        return


class TestDualScalingMethods(TestOCBScalingMethods):
    """Unit tests for the VectorData class."""

    def setUp(self):
        """Initialize the DualBoundary and VectorData objects."""

        test_dir = path.join(path.dirname(ocbpy.__file__), "tests",
                             "test_data")
        self.assertTrue(path.isdir(test_dir))
        self.ocb = ocbpy.DualBoundary(
            eab_filename=path.join(test_dir, "test_north_eab"),
            eab_instrument='image', ocb_instrument='image', hemisphere=1,
            ocb_filename=path.join(test_dir, "test_north_circle"))
        self.ocb_attrs = ['ocb_lat', 'ocb_mlt', 'r_corr', 'ocb_n', 'ocb_e',
                          'ocb_z']

        self.vdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind,
                                                  75.0, 22.0, aacgm_n=50.0,
                                                  aacgm_e=86.5, aacgm_z=5.0,
                                                  dat_name="Test",
                                                  dat_units="$m s^{-1}$")
        self.wdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind,
                                                  75.0, 22.0, aacgm_n=50.0,
                                                  aacgm_e=86.5, aacgm_z=5.0,
                                                  aacgm_mag=100.036243432,
                                                  dat_name="Test",
                                                  dat_units="$m s^{-1}$")
        self.zdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind,
                                                  87.2, 21.22, aacgm_n=0.0,
                                                  aacgm_e=0.0,
                                                  dat_name="Test Zero",
                                                  dat_units="$m s^{-1}$")
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.ocb, self.vdata, self.wdata, self.zdata, self.ocb_attrs
        return

    def test_vector_mult_ocb_ind(self):
        """Test the VectorData performance with multiple OCB indices."""
        # Update the VectorData attribute to contain a list of all good indices
        self.vdata.ocb_ind = numpy.arange(0, self.ocb.records, 1)

        # Set the VectorData OCB attributes
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)

        # Evaluate here
        for attr in self.ocb_attrs:
            with self.subTest(attr=attr):
                val = getattr(self.vdata, attr)
                self.assertTupleEqual(self.vdata.ocb_ind.shape, val.shape)
                self.assertTrue(numpy.isfinite(val).all())
        return

    def test_calc_ocb_vec_sign(self):
        """Test the calculation of the OCB vector signs."""

        # Set the initial values
        self.vdata.ocb_aacgm_mlt = self.ocb.ocb.phi_cent[
            self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.ocb.r_cent[
            self.vdata.ocb_ind]
        (self.vdata.ocb_lat, self.vdata.ocb_mlt, _,
         self.vdata.r_corr) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                    self.vdata.aacgm_mlt)
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()

        vmag = numpy.sqrt(self.vdata.aacgm_n**2 + self.vdata.aacgm_e**2)
        self.vdata.aacgm_naz = numpy.degrees(numpy.arccos(
            self.vdata.aacgm_n / vmag))

        # Calculate the vector data signs
        vsigns = self.vdata.calc_ocb_vec_sign(north=True, east=True)
        self.assertTrue(vsigns['north'])
        self.assertTrue(vsigns['east'])

        return

    def test_scale_vec(self):
        """Test the calculation of the OCB vector signs."""

        # Set the initial values
        self.vdata.ocb_aacgm_mlt = self.ocb.ocb.phi_cent[
            self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.ocb.r_cent[
            self.vdata.ocb_ind]
        (self.vdata.ocb_lat, self.vdata.ocb_mlt, _,
         self.vdata.r_corr) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                    self.vdata.aacgm_mlt)
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()

        vmag = numpy.sqrt(self.vdata.aacgm_n**2 + self.vdata.aacgm_e**2)
        self.vdata.aacgm_naz = numpy.degrees(numpy.arccos(
            self.vdata.aacgm_n / vmag))

        # Scale the data vector
        self.vdata.scale_vector()

        # Test the North and East components
        self.assertAlmostEqual(self.vdata.ocb_n, 85.52621517,
                               msg="northern vector coordinates differ")
        self.assertAlmostEqual(self.vdata.ocb_e, 51.64800594,
                               msg="eastern vector coordinates differe")

        # Test to see that the magnitudes and z-components are the same
        self.assertAlmostEqual(self.vdata.aacgm_mag, self.vdata.ocb_mag,
                               msg="vector magnitudes differ")
        self.assertAlmostEqual(self.vdata.ocb_z, self.vdata.aacgm_z,
                               msg="vector vertical commonents differ")

        return

    def test_define_quadrants(self):
        """Test the assignment of quadrants."""
        # Set the initial values
        self.vdata.ocb_aacgm_mlt = self.ocb.ocb.phi_cent[
            self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.ocb.r_cent[
            self.vdata.ocb_ind]
        (self.vdata.ocb_lat, self.vdata.ocb_mlt, _,
         self.vdata.r_corr) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                    self.vdata.aacgm_mlt)
        self.vdata.calc_vec_pole_angle()

        # Get the test quadrants
        self.vdata.define_quadrants()
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 1)
        return


class TestVectorDataRaises(unittest.TestCase):
    """Unit tests for the VectorData errors."""

    def setUp(self):
        """Initialize the tests for calc_vec_pole_angle."""
        test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                              "test_data", "test_north_circle")
        self.assertTrue(path.isfile(test_file))
        self.ocb = ocbpy.OCBoundary(filename=test_file, instrument='image')
        self.ocb.rec_ind = 27
        self.vdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind, 75.0,
                                                  22.0, aacgm_n=50.0,
                                                  aacgm_e=86.5, aacgm_z=5.0,
                                                  dat_name="Test",
                                                  dat_units="$m s^{-1}$")
        self.input_attrs = list()
        self.bad_input = [numpy.nan, numpy.full(shape=2, fill_value=numpy.nan)]
        self.raise_out = list()
        self.hold_val = None
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.ocb, self.vdata, self.input_attrs, self.bad_input
        del self.raise_out, self.hold_val
        return

    def test_init_ocb_array_failure(self):
        """Test init failure with mismatched OCB and input array input."""
        self.input_attrs = [0, [27, 31], 75.0, 22.0]
        self.bad_input = {'aacgm_n': 100.0, 'aacgm_e': 100.0,
                          'aacgm_z': 10.0, 'ocb_lat': 81.0,
                          'ocb_mlt': [2.0, 5.8, 22.5]}

        with self.assertRaisesRegex(ValueError, "OCB index and input shapes"):
            self.vdata = ocbpy.ocb_scaling.VectorData(*self.input_attrs,
                                                      **self.bad_input)
        return

    def test_init_ocb_vector_failure(self):
        """Test init failure with mismatched OCB and data array input."""
        self.input_attrs = [[3, 6, 0], [27, 31], [75.0, 87.2, 65.0],
                            [22.0, 21, 22]]
        self.bad_input = {'aacgm_n': [100.0, 110.0, 30.0],
                          'aacgm_e': [100.0, 110.0, 30.0],
                          'aacgm_z': [10.0, 10.0, 3.0]}

        with self.assertRaisesRegex(ValueError,
                                    "Mismatched OCB and Vector input shapes"):
            self.vdata = ocbpy.ocb_scaling.VectorData(*self.input_attrs,
                                                      **self.bad_input)
        return

    def test_reinit_ocb_vector_failure(self):
        """Test init failure with mismatched OCB and data array input."""
        self.vdata.dat_ind = [3, 6, 0]

        with self.assertRaisesRegex(ValueError,
                                    "Mismatched OCB and Vector input shapes"):
            self.vdata.ocb_ind = [27, 31]
        return

    def test_init_vector_failure(self):
        """Test init failure with a bad mix of vector and scalar input."""
        self.input_attrs = [[0, self.ocb.rec_ind, [75.0, 70.0], [22.0, 20.0]],
                            [[0, 1], self.ocb.rec_ind, [75.0, 70.0], 22.0],
                            [[0, 1], self.ocb.rec_ind, [75.0, 70.0],
                             [22.0, 20.0, 23.0]]]
        self.bad_input = [{'aacgm_n': 10.0},
                          {'aacgm_n': [100.0, 110.0, 30.0]},
                          {'aacgm_n': [100.0, 110.0, 30.0]}]
        self.raise_out = ['data index shape must match vector shape',
                          'mismatched dimensions for VectorData inputs',
                          'mismatched dimensions for VectorData inputs']

        for i, iattrs in enumerate(self.input_attrs):
            tset = [iattrs, self.bad_input[i], self.raise_out[i]]
            with self.subTest(tset=tset):
                with self.assertRaisesRegex(ValueError, tset[2]):
                    self.vdata = ocbpy.ocb_scaling.VectorData(*tset[0],
                                                              **tset[1])
        return

    def test_bad_calc_vec_pole_angle(self):
        """Test calc_vec_pole_angle failure with bad input."""
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
        return

    def test_no_ocb_lat(self):
        """Test failure when OCB latitude is not available."""
        self.vdata.ocb_lat = numpy.nan

        with self.assertRaisesRegex(ValueError, 'OCB coordinates required'):
            self.vdata.scale_vector()
        return

    def test_no_ocb_mlt(self):
        """Test failure when OCB MLT is not available."""
        self.vdata.ocb_mlt = numpy.nan

        with self.assertRaisesRegex(ValueError, 'OCB coordinates required'):
            self.vdata.scale_vector()
        return

    def test_no_ocb_pole_location(self):
        """Test failure when OCB pole location is not available."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_mlt = numpy.nan

        with self.assertRaisesRegex(ValueError, "OCB pole location required"):
            self.vdata.scale_vector()
        return

    def test_no_ocb_pole_angle(self):
        """Test failure when pole angle is not available."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = numpy.nan

        with self.assertRaisesRegex(
                ValueError, "vector angle in poles-vector triangle required"):
            self.vdata.scale_vector()
        return

    def test_bad_ocb_quad(self):
        """Test failure when OCB quadrant is wrong."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_quad = -1

        with self.assertRaisesRegex(ValueError, "OCB quadrant undefined"):
            self.vdata.calc_ocb_polar_angle()
        return

    def test_bad_vec_quad(self):
        """Test failure when vector quadrant is wrong."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.vec_quad = -1

        with self.assertRaisesRegex(ValueError, "Vector quadrant undefined"):
            self.vdata.calc_ocb_polar_angle()
        return

    def test_bad_quad_polar_angle(self):
        """Test failure when quadrant polar angle is bad."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_naz = numpy.nan

        with self.assertRaisesRegex(ValueError,
                                    "AACGM polar angle undefined"):
            self.vdata.calc_ocb_polar_angle()
        return

    def test_bad_quad_vector_angle(self):
        """Test failure when quandrant vector angle is bad."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = numpy.nan

        with self.assertRaisesRegex(ValueError, "Vector angle undefined"):
            self.vdata.calc_ocb_polar_angle()
        return

    def test_bad_calc_vec_sign_direction(self):
        """Test calc_vec_sign failure when no direction is provided."""
        self.vdata.set_ocb(self.ocb, None)

        with self.assertRaisesRegex(ValueError,
                                    "must set at least one direction"):
            self.vdata.calc_ocb_vec_sign()
        return

    def test_bad_calc_sign_ocb_quad(self):
        """Test calc_vec_sign failure with bad OCB quadrant."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_quad = -1

        with self.assertRaisesRegex(ValueError, "OCB quadrant undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)
        return

    def test_bad_calc_sign_vec_quad(self):
        """Test calc_vec_sign failure with bad vector quadrant."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.vec_quad = -1

        with self.assertRaisesRegex(ValueError, "Vector quadrant undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)
        return

    def test_bad_calc_sign_polar_angle(self):
        """Test calc_vec_sign failure with bad polar angle."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_naz = numpy.nan

        with self.assertRaisesRegex(ValueError,
                                    "AACGM polar angle undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)
        return

    def test_bad_calc_sign_pole_angle(self):
        """Test calc_vec_sign failure with bad pole angle."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = numpy.nan

        with self.assertRaisesRegex(ValueError, "Vector angle undefined"):
            self.vdata.calc_ocb_vec_sign(north=True)
        return

    def test_bad_define_quandrants_pole_mlt(self):
        """Test define_quadrants failure with bad pole MLT."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.ocb_aacgm_mlt = numpy.nan

        with self.assertRaisesRegex(ValueError, "OCB pole location required"):
            self.vdata.define_quadrants()
        return

    def test_bad_define_quandrants_vec_mlt(self):
        """Test define_quadrants failure with bad vector MLT."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.aacgm_mlt = numpy.nan

        with self.assertRaisesRegex(ValueError,
                                    "Vector AACGM location required"):
            self.vdata.define_quadrants()
        return

    def test_bad_define_quandrants_pole_angle(self):
        """Test define_quadrants failure with bad pole angle."""
        self.vdata.set_ocb(self.ocb, None)
        self.vdata.pole_angle = numpy.nan

        with self.assertRaisesRegex(
                ValueError, "vector angle in poles-vector triangle required"):
            self.vdata.define_quadrants()
        return


class TestHaversine(unittest.TestCase):
    """Unit tests for the haversine functions."""

    def setUp(self):
        """Initialize the testing set up."""
        self.input_angles = numpy.linspace(-2.0 * numpy.pi, 2.0 * numpy.pi, 9)
        self.hav_out = numpy.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5,
                                    0.0])

        # archaversine is confinded to 0-pi
        self.ahav_out = abs(numpy.array([aa - numpy.sign(aa) * 2.0 * numpy.pi
                                         if abs(aa) > numpy.pi
                                         else aa for aa in self.input_angles]))
        self.out = None
        return

    def tearDown(self):
        """Clean up the testing set up."""
        del self.input_angles, self.hav_out, self.out, self.ahav_out
        return

    def test_haversine(self):
        """Test implimentation of the haversine."""
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
                    self.assertTrue(numpy.all(abs(self.out - tset[1]) < 1.0e-7))

        return

    def test_inverse_haversine(self):
        """Test the implemenation of the inverse haversine."""
        # Cycle through all the possible input options
        for i, tset in enumerate([(self.hav_out[1], self.ahav_out[1]),
                                  (list(self.hav_out), self.ahav_out),
                                  (self.hav_out, self.ahav_out)]):
            with self.subTest(tset=tset):
                self.out = ocbpy.ocb_scaling.archav(tset[0])

                # Assess the output
                if i == 0:
                    self.assertAlmostEqual(self.out, abs(tset[1]))
                else:
                    self.assertTrue(numpy.all(self.out - tset[1] < 1.0e-7))

        del tset

    def test_inverse_haversine_small_float(self):
        """Test the inverse haversine with very small numbers."""
        self.assertEqual(ocbpy.ocb_scaling.archav(1.0e-17), 0.0)
        self.assertEqual(ocbpy.ocb_scaling.archav(-1.0e-17), 0.0)
        return

    def test_inverse_haversine_nan_float(self):
        """Test implimentation of the inverse haversine with NaN."""
        self.assertTrue(numpy.isnan(ocbpy.ocb_scaling.archav(numpy.nan)))
        return

    def test_inverse_haversine_negative_float(self):
        """Test implimentation of the inverse haversine with negative input."""
        self.assertTrue(numpy.isnan(ocbpy.ocb_scaling.archav(-1.0)))
        return

    def test_inverse_haversine_mixed(self):
        """Test the inverse haversine with array input of good/bad values."""
        # Update the test input and output
        self.hav_out[0] = 1.0e-17
        self.ahav_out[0] = 0.0
        self.hav_out[1] = numpy.nan
        self.ahav_out[1] = numpy.nan
        self.hav_out[2] = -1.0
        self.ahav_out[2] = numpy.nan

        self.out = ocbpy.ocb_scaling.archav(self.hav_out)

        for i, hout in enumerate(self.out):
            if numpy.isnan(hout):
                self.assertTrue(numpy.isnan(self.ahav_out[i]))
            else:
                self.assertAlmostEqual(hout, self.ahav_out[i])
        return


class TestOCBScalingArrayMethods(unittest.TestCase):
    """Unit tests for the ocb_scaling methods with array inputs."""

    def setUp(self):
        """Set up the test environment."""
        test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                              "test_data", "test_north_circle")
        self.ocb = ocbpy.OCBoundary(filename=test_file, instrument='image')

        # Construct a set of test vectors that have all the different OCB
        # and vector combinations and one with no magnitude. The test OCB pole
        # is at 87.24 deg, 5.832 h
        lats = numpy.full(shape=(17,), fill_value=75.0)
        lats[8:] = 89.0
        mlts = numpy.zeros(shape=(17,))
        mlts[4:8] = 15.0
        mlts[8:12] = 7.0
        mlts[12:] = 5.0
        north = [10.0, 10.0, -10.0, -10.0, 10.0, 10.0, -10.0, -10.0,
                 10.0, 10.0, -10.0, -10.0, 10.0, 10.0, -10.0, -10.0, 0.0]
        east = [3.0, -3.0, -3.0, 3.0, 3.0, -3.0, -3.0, 3.0, 3.0, -3.0, -3.0,
                3.0, 3.0, -3.0, -3.0, 3.0, 0.0]
        vert = numpy.zeros(shape=(17,))
        vert[0] = 5.0
        self.ref_quads = {'ocb': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
                                  4, 4],
                          'vec': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3,
                                  4, 1]}

        self.vargs = [numpy.arange(0, 17, 1), 27, lats, mlts]
        self.vkwargs = {'aacgm_n': numpy.array(north),
                        'aacgm_e': numpy.array(east),
                        'aacgm_z': numpy.array(vert), 'dat_name': 'Test',
                        'dat_units': 'm/s'}
        self.vdata = None
        self.out = None

        self.aacgm_mag = numpy.full(shape=(17,), fill_value=10.44030650891055)
        self.aacgm_mag[0] = 11.575836902790225
        self.aacgm_mag[-1] = 0.0
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.ocb, self.vargs, self.vkwargs, self.out, self.vdata
        del self.aacgm_mag, self.ref_quads
        return

    def set_vector_ocb_ind(self):
        """Update the input vector to have vector OCB index input."""
        self.vargs[1] = numpy.full(shape=self.vargs[-1].shape, fill_value=27)
        self.vargs[1][8:] = 31
        return

    def test_array_vector_str_not_calc(self):
        """Test VectorData print statement with uncalculated array input."""
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.out = str(self.vdata)
        self.assertRegex(self.out, "Index")
        self.assertRegex(self.out, "nan, nan, {:d}".format(self.vargs[1]))
        return

    def test_array_vector_str_calc(self):
        """Test the VectorData print statement with calculated array input."""
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)
        self.out = str(self.vdata)
        self.assertRegex(self.out, "Index")
        self.assertNotRegex(self.out, "nan, nan")
        return

    def test_array_vector_str_calc_ocb_vec_array(self):
        """Test VectorData print statement with calculated ocb/vec arrays."""
        self.set_vector_ocb_ind()
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)
        self.out = str(self.vdata)
        self.assertRegex(self.out, "Index")
        self.assertNotRegex(self.out, "nan, nan")
        return

    def test_array_vector_str_calc_ocb_array(self):
        """Test the VectorData print statement with calculated ocb arrays."""
        self.vargs[0] = self.vargs[0][0]
        self.set_vector_ocb_ind()
        self.vargs[2] = self.vargs[2][0]
        self.vargs[3] = self.vargs[3][0]
        self.vkwargs['aacgm_n'] = self.vkwargs['aacgm_n'][0]
        self.vkwargs['aacgm_e'] = self.vkwargs['aacgm_e'][0]
        self.vkwargs['aacgm_z'] = self.vkwargs['aacgm_z'][0]
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)
        self.out = str(self.vdata)
        self.assertRegex(self.out, "Index")
        self.assertNotRegex(self.out, "nan, nan")
        return

    def test_init_nez_vec_array(self):
        """Test VectorData initialisation  with vector array components."""
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.assertEqual(len(self.vdata.aacgm_mag), len(self.vargs[0]))
        self.assertEqual(len(self.vdata.ocb_mag), len(self.vargs[0]))
        for i, self.out in enumerate(self.vdata.aacgm_mag):
            self.assertAlmostEqual(self.out, self.aacgm_mag[i])
        return

    def test_init_nez_ocb_vec_array(self):
        """Test VectorData set up with ocb and vector array components."""
        self.set_vector_ocb_ind()
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.assertEqual(len(self.vdata.aacgm_mag), len(self.vargs[0]))
        self.assertEqual(len(self.vdata.ocb_mag), len(self.vargs[0]))
        for i, self.out in enumerate(self.vdata.aacgm_mag):
            self.assertAlmostEqual(self.out, self.aacgm_mag[i])
        return

    def test_init_nez_ocb_array(self):
        """Test VectorData initialisation with ocb array components."""
        self.set_vector_ocb_ind()
        self.vargs[2] = self.vargs[2][0]
        self.vargs[3] = self.vargs[3][0]
        self.vkwargs['aacgm_n'] = self.vkwargs['aacgm_n'][0]
        self.vkwargs['aacgm_e'] = self.vkwargs['aacgm_e'][0]
        self.vkwargs['aacgm_z'] = self.vkwargs['aacgm_z'][0]

        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.assertEqual(len(self.vdata.ocb_mag), len(self.vargs[1]))
        self.assertTrue(abs(self.vdata.aacgm_mag - self.aacgm_mag[0]).all()
                        < 1.0e-7, msg="unexpected AACGM vector magnitude")
        return

    def test_init_mag(self):
        """Test the set up of the VectorData array input with magnitude."""
        self.vkwargs['aacgm_mag'] = self.aacgm_mag
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.assertEqual(len(self.vdata.aacgm_mag), len(self.vargs[0]))
        for i, self.out in enumerate(self.vdata.aacgm_mag):
            self.assertAlmostEqual(self.out, self.aacgm_mag[i])
        return

    def test_vector_all_bad_lat(self):
        """Test VectorData output with all data from the wrong hemisphere."""
        self.vargs[2] *= -1.0
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertTrue(len(self.vdata.ocb_lat), len(self.vargs[2]))
        self.assertTrue(numpy.all(numpy.isnan(self.vdata.ocb_lat)))
        self.assertTrue(numpy.all(numpy.isnan(self.vdata.ocb_mlt)))
        self.assertTrue(numpy.all(numpy.isnan(self.vdata.ocb_n)))
        self.assertTrue(numpy.all(numpy.isnan(self.vdata.ocb_e)))
        self.assertTrue(numpy.all(numpy.isnan(self.vdata.ocb_z)))

        # Ensure that input is not overwritten
        for vkey in self.vkwargs.keys():
            self.out = getattr(self.vdata, vkey)
            if vkey.find('aacgm_') == 0:
                for i, val in enumerate(self.vkwargs[vkey]):
                    self.assertEqual(self.out[i], val)
            else:
                self.assertRegex(self.out, self.vkwargs[vkey])
        return

    def test_vector_some_bad_lat(self):
        """Test the VectorData output with mixed hemisphere input."""
        self.vargs[2][0] *= -1.0
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertTrue(len(self.vdata.ocb_lat), len(self.vargs[2]))

        # Ensure the wrong hemisphere is NaN
        self.assertTrue(numpy.isnan(self.vdata.ocb_lat[0]))
        self.assertTrue(numpy.isnan(self.vdata.ocb_mlt[0]))
        self.assertTrue(numpy.isnan(self.vdata.ocb_n[0]))
        self.assertTrue(numpy.isnan(self.vdata.ocb_e[0]))
        self.assertTrue(numpy.isnan(self.vdata.ocb_z[0]))

        # Ensure that input is not overwritten
        for vkey in self.vkwargs.keys():
            self.out = getattr(self.vdata, vkey)
            if vkey.find('aacgm_') == 0:
                for i, val in enumerate(self.vkwargs[vkey]):
                    self.assertEqual(self.out[i], val)
            else:
                self.assertRegex(self.out, self.vkwargs[vkey])

        # Ensure the right hemisphere is good
        self.assertAlmostEqual(self.vdata.aacgm_mag[1], self.aacgm_mag[1])
        self.assertAlmostEqual(self.vdata.ocb_mag[1], self.aacgm_mag[1])
        return

    def test_calc_vec_pole_angle_flat(self):
        """Test the polar angle calculation with angles of 0 and 180 deg."""
        self.vargs[3] = numpy.full(shape=self.vargs[2].shape,
                                   fill_value=ocbpy.ocb_time.deg2hr(
                                       self.ocb.phi_cent[self.vargs[1]]))
        self.vargs[3][numpy.array(self.ref_quads['ocb']) > 2] += 12.0
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertTrue(numpy.all([quad in [2, 4]
                                   for quad in self.vdata.ocb_quad]))
        self.assertEqual(list(self.vdata.vec_quad), self.ref_quads['vec'])
        self.assertTrue(numpy.all([self.vdata.pole_angle[i] == 0.0
                                   for i, quad in enumerate(self.vdata.ocb_quad)
                                   if quad == 2]))
        self.assertTrue(numpy.all([self.vdata.pole_angle[i] == 180.0
                                   for i, quad in enumerate(self.vdata.ocb_quad)
                                   if quad == 4]))
        return

    def test_array_vec_quad(self):
        """Test the assignment of vector quadrants with array input."""
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertEqual(len(self.vargs[0]), len(self.vdata.vec_quad))
        self.assertEqual(list(self.vdata.vec_quad), self.ref_quads['vec'])
        return

    def test_array_ocb_quad(self):
        """Test the assignment of OCB quadrants with array input."""
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertEqual(len(self.vargs[0]), len(self.vdata.ocb_quad))
        self.assertEqual(list(self.vdata.ocb_quad), self.ref_quads['ocb'])
        return

    def test_one_undefinable_ocb_quadrant(self):
        """Test VectorData array set up for a undefinable OCB quadrant."""
        self.vargs[2][1] = 0.0
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertEqual(self.vdata.ocb_quad[0], 1)
        self.assertEqual(self.vdata.ocb_quad[1], 0)
        return

    def test_one_undefinable_vec_quadrant(self):
        """Test VectorData array set up for a undefinable vec quadrant."""
        self.vkwargs['aacgm_n'][1] = numpy.nan
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        self.assertEqual(self.vdata.vec_quad[0], 1)
        self.assertEqual(self.vdata.vec_quad[1], 0)
        return

    def test_define_quadrants_neg_adj_mlt(self):
        """Test the quadrant assignment with a negative AACGM MLT."""
        self.vargs[3] -= 24.0
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)

        self.assertGreater(self.vdata.ocb_aacgm_mlt - self.vargs[3][0], 24)

        self.assertEqual(list(self.vdata.ocb_quad), self.ref_quads['ocb'])
        self.assertEqual(list(self.vdata.vec_quad), self.ref_quads['vec'])
        return

    def test_scale_vec_pole_angle_zero(self):
        """Test the calculation of the OCB vector sign with no pole angle."""
        self.vdata = ocbpy.ocb_scaling.VectorData(*self.vargs, **self.vkwargs)
        self.vdata.set_ocb(self.ocb)

        # If the OCB pole is closer to the AACGM pole than the vector, set
        # the pole angle to zero deg. Otherwise, set it to 180.0 deg
        self.vdata.pole_angle = numpy.zeros(shape=self.vargs[2].shape)
        self.vdata.pole_angle[self.vdata.ocb_quad > 2] = 180.0

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
                      escale)]:
            with self.subTest(tset=tset):
                setattr(self.vdata, tset[0], tset[1])

                # Run the scale_vector routine with the new attributes
                self.vdata.scale_vector()

                # Assess the ocb north and east components
                self.assertTrue(numpy.all(self.vdata.ocb_n == tset[2]))
                self.assertTrue(numpy.all(self.vdata.ocb_e == tset[3]))

        return
