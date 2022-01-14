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
        del self.test_func
        return

    def test_match(self):
        """Test to see that the data matching works properly."""
        # Build a array of times for a test dataset
        self.ocb.rec_ind = 27
        test_times = np.arange(self.ocb.dtime[self.ocb.rec_ind],
                               self.ocb.dtime[self.ocb.rec_ind + 5],
                               dt.timedelta(seconds=600)).astype(dt.datetime)

        # Because the array starts at the first good OCB, will return zero
        self.idat = self.test_func(self.ocb, test_times, idat=self.idat)
        self.assertEqual(self.idat, 0)
        self.assertEqual(self.ocb.rec_ind, 27)

        # The next test time will cause the OCB to cycle forward to a new
        # record
        idat = ocbpy.ocboundary.match_data_ocb(self.ocb, test_times, idat=1)
        self.assertEqual(idat, 1)
        self.assertEqual(self.ocb.rec_ind, 31)
        self.assertLess(
            abs((test_times[idat]
                 - self.ocb.dtime[self.ocb.rec_ind]).total_seconds()), 600.0)
        del test_times
        return

    def test_good_first_match(self):
        """Test ability to find the first good OCB."""
        # Change the logging level
        ocbpy.logger.setLevel(logging.INFO)

        # Because the array starts at the first good OCB, will return zero
        self.idat = self.test_func(self.ocb, [self.ocb.dtime[27]],
                                   idat=self.idat)
        self.assertEqual(self.idat, 0)
        self.assertEqual(self.ocb.rec_ind, 27)

        # The first match will be announced in the log
        self.lwarn = "found first good OCB record at"
        self.lout = self.log_capture.getvalue()
        self.assertRegex(self.lout, self.lwarn)
        return

    def test_bad_first_match(self):
        """Test ability to not find a good OCB."""
        # Set requirements for good OCB so high that none will pass
        self.idat = self.test_func(self.ocb, [self.ocb.dtime[27]],
                                   idat=self.idat, min_sectors=24)
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
        self.idat = self.test_func(self.ocb, [self.ocb.dtime[27]], idat=-1)
        self.assertEqual(self.idat, -1)
        self.assertGreaterEqual(self.ocb.rec_ind, self.ocb.records)
        return

    def test_bad_dat_ind(self):
        """Test ability to exit if data record counter is too high."""
        # Set the OCB record index to the end
        self.idat = self.test_func(self.ocb, [self.ocb.dtime[27]], idat=2)
        self.assertEqual(self.idat, 2)
        self.assertGreaterEqual(self.ocb.rec_ind, -1)
        return

    def test_bad_first_data_time(self):
        """Test ability to cycle past data times not close enough to match."""
        # Set the OCB record index to the beginning and match
        self.idat = self.test_func(self.ocb,
                                   [self.ocb.dtime[27] - dt.timedelta(days=1),
                                    self.ocb.dtime[27]], idat=self.idat)
        self.assertEqual(self.idat, 1)
        self.assertEqual(self.ocb.rec_ind, 27)
        return

    def test_data_all_before_first_ocb_record(self):
        """Test failure when data occurs before boundaries."""
        # Change the logging level
        ocbpy.logger.setLevel(logging.ERROR)

        # Set the OCB record index to the beginning and match
        self.idat = self.test_func(self.ocb,
                                   [self.ocb.dtime[27] - dt.timedelta(days=1)],
                                   idat=self.idat)
        self.assertIsNone(self.idat)
        self.assertGreaterEqual(self.ocb.rec_ind, 27)

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
        self.idat = self.test_func(
            self.ocb, [self.ocb.dtime[37] - dt.timedelta(seconds=601)],
            idat=self.idat)
        self.assertEqual(self.idat, 1)
        self.assertGreaterEqual(self.ocb.rec_ind, 37)

        # Check the log output
        self.lwarn = "no OCB data available within"
        self.lout = self.log_capture.getvalue()
        self.assertRegex(self.lout, self.lwarn)
        self.lwarn = "of input measurement"
        self.assertRegex(self.lout, self.lwarn)
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

        del set_north
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.ocb, self.test_func
        return

    def eval_retrieved_indices(self):
        """Evaluate the retrieved indices."""

        self.assertEqual(self.out[0], 27)
        self.assertEqual(self.out[1], 31)
        self.assertEqual(len(self.out), 36)
        return

    def test_retrieve_all_good_ind(self):
        """Test that all good indices are retrieved with index at start."""
        self.ocb.rec_ind = -1
        self.out = self.test_func(self.ocb)

        self.eval_retrieved_indices()
        self.assertEqual(self.ocb.rec_ind, -1)
        return
    
    def test_retrieve_all_good_ind_init_middle(self):
        """Test that all good indices are retrieved with index in the middle."""
        self.ocb.rec_ind = 65
        self.out = self.test_func(self.ocb)

        self.eval_retrieved_indices()
        self.assertEqual(self.ocb.rec_ind, 65)
        return

    def test_retrieve_all_good_ind_empty(self):
        """Test routine that retrieves all good indices, no data loaded."""
        self.ocb = ocbpy.OCBoundary(filename=None)
        self.out = self.test_func(self.ocb)

        self.assertEqual(len(self.out), 0)
        return
