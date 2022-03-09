#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Test the cycle_boundary sub-module functions."""

import datetime as dt
from io import StringIO
import logging
import numpy as np
from os import path
import unittest

import ocbpy


class TestCycleMatchData(unittest.TestCase):
    """Unit tests for the `match_data_ocb` function."""

    def setUp(self):
        """Initialize the test environment."""
        set_north = {"filename": path.join(path.dirname(ocbpy.__file__),
                                           "tests", "test_data",
                                           "test_north_circle"),
                     "instrument": "image"}
        self.ocb = ocbpy.OCBoundary(**set_north)
        self.ocb.rec_ind = -1
        self.idat = 0
        self.test_func = ocbpy.cycle_boundary.match_data_ocb
        self.rec_ind = 27
        self.rec_ind2 = 31
        self.del_time = 60
        self.bad_time = self.ocb.dtime[37] - dt.timedelta(
            seconds=self.del_time + 1)

        # Initialize logging
        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)
        del set_north
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.ocb, self.lwarn, self.lout, self.log_capture, self.idat
        del self.test_func, self.rec_ind, self.rec_ind2, self.del_time
        del self.bad_time
        return

    def test_deprecated_kwargs(self):
        """Test warnings for deprecated kwargs in `get_next_good_ocb_ind`."""
        # Set the deprecated keyword arguments with standard values
        dep_inputs = {"min_sectors": 7, "rcent_dev": 8.0, "max_r": 23.0,
                      "min_r": 10.0}
        del_ind = self.rec_ind2 - self.rec_ind
        test_times = np.arange(
            self.ocb.dtime[self.ocb.rec_ind],
            self.ocb.dtime[self.ocb.rec_ind + del_ind],
            dt.timedelta(seconds=self.del_time)).astype(dt.datetime)

        # Cycle through the keyword arguments that should raise a warning
        for dkey in dep_inputs.keys():
            kwargs = {dkey: dep_inputs[dkey]}
            with self.subTest(kwargs=kwargs):
                with self.assertWarnsRegex(DeprecationWarning,
                                           "Deprecated kwarg will be removed"):
                    self.test_func(self.ocb, test_times, idat=1, **kwargs)
        return

    def test_bad_class_cycling_method(self):
        """Test raises ValueError when Boundary class missing cycling method."""

        # Define and set a bad boundary class object
        class test_ocb(object):
            def __init__(self):
                self.rec_ind = 0
                self.records = 10
                return

        data_times = self.ocb.dtime
        self.ocb = test_ocb()

        with self.assertRaisesRegex(ValueError, "missing index cycling method"):
            self.test_func(self.ocb, data_times, idat=self.idat)
        return

    def test_match(self):
        """Test to see that the data matching works properly."""
        # Build a array of times for a test dataset
        self.ocb.rec_ind = self.rec_ind
        del_t = (self.ocb.dtime[self.rec_ind2]
                 - self.ocb.dtime[self.rec_ind]).total_seconds()
        if del_t < self.del_time:
            self.del_time = del_t
        test_times = np.arange(
            self.ocb.dtime[self.rec_ind], self.ocb.dtime[self.rec_ind2]
            + dt.timedelta(seconds=1),
            dt.timedelta(seconds=self.del_time)).astype(dt.datetime)

        # Because the array starts at the first good OCB, will return zero
        self.idat = self.test_func(self.ocb, test_times, idat=self.idat)
        self.assertEqual(self.idat, 0)
        self.assertEqual(self.ocb.rec_ind, self.rec_ind)

        # The next test time will cause the OCB to cycle forward to a new
        # record
        self.ocb.rec_ind = self.rec_ind2
        idat = self.test_func(self.ocb, test_times, idat=1)
        self.assertEqual(idat, len(test_times) - 1)
        self.assertEqual(self.ocb.rec_ind, self.rec_ind2)
        self.assertLess(
            abs((test_times[idat]
                 - self.ocb.dtime[self.ocb.rec_ind]).total_seconds()),
            self.del_time)
        del test_times
        return

    def test_match_two_boundaries(self):
        """Test to see that the data chooses the closest good boundary."""
        # Set the input times and time tolerance so that the first and second
        # good boundaries will match, but the second one is a better match
        self.ocb.rec_ind = self.rec_ind
        max_tol = (self.ocb.dtime[self.rec_ind2]
                   - self.ocb.dtime[self.rec_ind]).total_seconds()
        test_times = [self.ocb.dtime[self.rec_ind2] - dt.timedelta(seconds=1)]

        # Match the data and boundaries
        self.idat = self.test_func(self.ocb, test_times, idat=self.idat,
                                   max_tol=max_tol)
        self.assertEqual(self.idat, 0)
        self.assertEqual(self.ocb.rec_ind, self.rec_ind2)

        return

    def test_good_first_match(self):
        """Test ability to find the first good OCB."""
        # Change the logging level
        ocbpy.logger.setLevel(logging.INFO)

        # Because the array starts at the first good OCB, will return zero
        self.idat = self.test_func(self.ocb, [self.ocb.dtime[self.rec_ind]],
                                   idat=self.idat)
        self.assertEqual(self.idat, 0)
        self.assertEqual(self.ocb.rec_ind, self.rec_ind)

        # The first match will be announced in the log
        self.lwarn = "found first good OCB record at"
        self.lout = self.log_capture.getvalue()
        self.assertRegex(self.lout, self.lwarn)
        return

    def test_bad_first_match(self):
        """Test ability to not find a good OCB."""

        # Set requirements for good OCB so high that none will pass
        self.idat = self.test_func(self.ocb, [self.ocb.dtime[self.rec_ind]],
                                   idat=self.idat, max_merit=0.0)
        self.assertEqual(self.idat, 0)
        self.assertGreaterEqual(self.ocb.rec_ind, self.ocb.records)

        # The first match will be announced in the log
        self.lwarn = "unable to find a good OCB record"
        self.lout = self.log_capture.getvalue()
        self.assertRegex(self.lout, self.lwarn)
        return

    def test_bad_ocb_ind(self):
        """Test ability to exit if ocb record counter is too high."""

        # Set the OCB record index to the end
        self.ocb.rec_ind = self.ocb.records
        self.idat = self.test_func(self.ocb, [self.ocb.dtime[self.rec_ind]],
                                   idat=-1)
        self.assertEqual(self.idat, -1)
        self.assertGreaterEqual(self.ocb.rec_ind, self.ocb.records)
        return

    def test_bad_dat_ind(self):
        """Test ability to exit if data record counter is too high."""
        # Set the OCB record index to the end
        self.idat = self.test_func(self.ocb, [self.ocb.dtime[self.rec_ind]],
                                   idat=2)
        self.assertEqual(self.idat, 2)
        self.assertGreaterEqual(self.ocb.rec_ind, -1)
        return

    def test_bad_first_data_time(self):
        """Test ability to cycle past data times not close enough to match."""
        # Set the OCB record index to the beginning and match
        self.idat = self.test_func(
            self.ocb, [self.ocb.dtime[self.rec_ind] - dt.timedelta(days=1),
                       self.ocb.dtime[self.rec_ind]], idat=self.idat)
        self.assertEqual(self.idat, 1)
        self.assertEqual(self.ocb.rec_ind, self.rec_ind)
        return

    def test_data_all_before_first_ocb_record(self):
        """Test failure when data occurs before boundaries."""
        # Change the logging level
        ocbpy.logger.setLevel(logging.ERROR)

        # Set the OCB record index to the beginning and match
        self.idat = self.test_func(
            self.ocb, [self.ocb.dtime[self.rec_ind]
                       - dt.timedelta(days=self.del_time)], idat=self.idat)
        self.assertIsNone(self.idat)
        self.assertGreaterEqual(self.ocb.rec_ind, self.rec_ind)

        # Check the log output
        self.lwarn = "no input data close enough to the first record"
        self.lout = self.log_capture.getvalue()
        self.assertRegex(self.lout, self.lwarn)
        return

    def test_late_data_time_alignment(self):
        """Test failure when data occurs after boundaries."""
        # Change the logging level
        ocbpy.logger.setLevel(logging.INFO)

        # Match OCB with data that occurs after the boundaries end
        self.idat = self.test_func(self.ocb,
                                   [self.ocb.dtime[self.ocb.records - 1]
                                    + dt.timedelta(days=2)], idat=self.idat)
        self.assertIsNone(self.idat, None)
        self.assertGreaterEqual(self.ocb.rec_ind, self.ocb.records)

        # Check the log output
        self.lwarn = "no OCB data available within"
        self.lout = self.log_capture.getvalue()
        self.assertRegex(self.lout, self.lwarn)
        self.lwarn = "of first measurement"
        self.assertRegex(self.lout, self.lwarn)
        return

    def test_no_data_time_alignment(self):
        """ Test failure when data occurs between boundaries """
        # Change the logging level
        ocbpy.logger.setLevel(logging.INFO)

        # Match OCBs with misaligned input data
        self.idat = self.test_func(self.ocb, [self.bad_time], idat=self.idat)
        self.assertEqual(self.idat, 1)
        self.assertGreaterEqual(self.ocb.rec_ind, 0)
        self.assertLessEqual(self.ocb.rec_ind, self.ocb.records)

        # Check the log output
        self.lwarn = "no OCB data available within"
        self.lout = self.log_capture.getvalue()
        self.assertRegex(self.lout, self.lwarn)
        self.lwarn = "of input measurement"
        self.assertRegex(self.lout, self.lwarn)
        return


class TestCycleMatchDualData(TestCycleMatchData):
    """Unit tests for the `match_data_ocb` function."""

    def setUp(self):
        """Initialize the test environment."""
        set_dual = {"ocb_filename": path.join(path.dirname(ocbpy.__file__),
                                              "tests", "test_data",
                                              "test_north_circle"),
                    "ocb_instrument": "image", 'eab_instrument': 'image',
                    'eab_filename': path.join(path.dirname(ocbpy.__file__),
                                              "tests", "test_data",
                                              "test_north_eab")}
        self.ocb = ocbpy.DualBoundary(**set_dual)
        self.ocb.rec_ind = -1
        self.idat = 0
        self.test_func = ocbpy.cycle_boundary.match_data_ocb
        self.rec_ind = 0
        self.rec_ind2 = 1
        self.del_time = 60
        self.bad_time = self.ocb.ocb.dtime[37] - dt.timedelta(
            seconds=self.del_time + 1)

        # Initialize logging
        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)
        del set_dual

        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.ocb, self.lwarn, self.lout, self.log_capture, self.idat
        del self.test_func, self.rec_ind, self.rec_ind2, self.del_time
        del self.bad_time
        return


class TestCycleGoodIndices(unittest.TestCase):
    """Unit tests for the `retrieve_all_good_indices` function."""

    def setUp(self):
        """Initialize the test environment."""
        set_north = {"filename": path.join(path.dirname(ocbpy.__file__),
                                           "tests", "test_data",
                                           "test_north_circle"),
                     "instrument": "image"}
        self.ocb = ocbpy.OCBoundary(**set_north)
        self.ocb.rec_ind = -1
        self.test_func = ocbpy.cycle_boundary.retrieve_all_good_indices
        self.rec_ind = 27
        self.rec_ind2 = 31

        del set_north
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.ocb, self.test_func, self.rec_ind, self.rec_ind2
        return

    def test_retrieve_all_good_ind(self):
        """Test all good indices are retrieved with different rec_ind values."""

        for rind in [-1, 0, 65, self.ocb.records - 1]:
            with self.subTest(rind=rind):
                # Initalize the OCBoundary record index
                self.ocb.rec_ind = rind

                # Retrive all good record indices
                self.out = self.test_func(self.ocb)

                # Test that all records were retrieved
                self.assertEqual(self.out[0], self.rec_ind)
                self.assertEqual(self.out[1], self.rec_ind2)
                self.assertEqual(len(self.out), 36)

                # Test that the OCBoundary record index has not changed
                self.assertEqual(self.ocb.rec_ind, rind)
        return

    def test_retrieve_all_good_ind_empty(self):
        """Test routine that retrieves all good indices, no data loaded."""
        self.ocb = ocbpy.OCBoundary(filename=None)
        self.out = self.test_func(self.ocb)

        self.assertEqual(len(self.out), 0)
        return


class TestGeneralSatelliteFunctions(unittest.TestCase):
    """Unit tests for the general satellite functions."""

    def setUp(self):
        """Set up the test environment."""
        self.test_dir = path.join(path.dirname(ocbpy.__file__), "tests",
                                  "test_data")
        self.ocb = ocbpy.OCBoundary(instrument="dmsp-ssj", hemisphere=1,
                                    filename=path.join(
                                        self.test_dir,
                                        "dmsp-ssj_north_out.ocb"))

        self.mlt = np.arange(0, 24, 0.5)
        self.lat = np.full(shape=self.mlt.shape, fill_value=75.0)
        self.good = [False, False, False, False, False, False, False, False,
                     False, False, False, False, False, False, False, False,
                     False, False, False, True, False, False, False, False,
                     False, False, False, False, False, False, False, True,
                     False, False, False, False, False, False, False, False,
                     False, False, False, False, False, False, False, False]
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.test_dir, self.ocb, self.mlt, self.lat, self.good
        return

    def test_satellite_track_defaults(self):
        """Test the satellite track ID with default boundaries."""

        sat_good = ocbpy.cycle_boundary.satellite_track(
            self.lat, self.mlt, self.ocb.x_1[0], self.ocb.y_1[0],
            self.ocb.x_2[0], self.ocb.y_2[0], 1)
        self.assertListEqual(self.good, list(sat_good))
        return

    def test_satellite_track_delta_xy(self):
        """Test the satellite track ID with wider X/Y limits."""

        # Set the expected output
        self.good = [gval if i < 17 or i > 33 else True
                     for i, gval in enumerate(self.good)]

        # Run and evaluate the output
        sat_good = ocbpy.cycle_boundary.satellite_track(
            self.lat, self.mlt, self.ocb.x_1[0], self.ocb.y_1[0],
            self.ocb.x_2[0], self.ocb.y_2[0], 1, del_x=5.0, del_y=5.0)
        self.assertListEqual(self.good, list(sat_good))
        return

    def test_satellite_track_eq_bound(self):
        """Test the satellite track ID with a wide equatorward boundary."""

        # Update the input and output
        self.lat -= 15.0
        self.good = [False, False, False, False, False, False, False, False,
                     False, False, False, False, False, False, False, False,
                     False, False, False, False, False, False, False, False,
                     False, False, False, False, False, False, False, False,
                     False, False, True, False, False, False, False, False,
                     False, False, False, False, False, False, False, False]

        # Ensure this produces no good results initially
        sat_good = ocbpy.cycle_boundary.satellite_track(
            self.lat, self.mlt, self.ocb.x_1[0], self.ocb.y_1[0],
            self.ocb.x_2[0], self.ocb.y_2[0], 1)
        self.assertFalse(sat_good.any())

        # Test with expanded equatorward boundaries
        sat_good = ocbpy.cycle_boundary.satellite_track(
            self.lat, self.mlt, self.ocb.x_1[0], self.ocb.y_1[0],
            self.ocb.x_2[0], self.ocb.y_2[0], 1, past_bound=20)
        self.assertListEqual(self.good, list(sat_good))
        return

    def test_satellite_track_errors(self):
        """Test that bad input raises appropriate ValueErrors."""

        bad_val = -10.0
        msg_dict = {"del_x": "x- and y-axis allowable difference must be ",
                    "del_y": "x- and y-axis allowable difference must be ",
                    "past_bound": "equatorward buffer for track must be ",
                    "hemisphere": "hemisphere expecting"}

        for key in msg_dict.keys():
            msg = msg_dict[key]
            hemi = 1 if key != "hemisphere" else bad_val
            kwargs = {key: bad_val} if key != "hemisphere" else {}

            with self.subTest(kwargs=kwargs, msg=msg):
                with self.assertRaisesRegex(ValueError, msg):
                    ocbpy.cycle_boundary.satellite_track(
                        self.lat, self.mlt, self.ocb.x_1[0], self.ocb.y_1[0],
                        self.ocb.x_2[0], self.ocb.y_2[0], hemi, **kwargs)
        return
