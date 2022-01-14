#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the deprecated ocboundary sub-module."""

import datetime as dt
from io import StringIO
import logging
from os import path
import unittest
import warnings

import ocbpy
from test_cycle_boundary import TestCycleMatchData, TestCycleGoodIndices


class TestOCBoundaryDeprecation(unittest.TestCase):
    """Unit tests for the deprecation of the ocboundary sub-module."""

    def setUp(self):
        """Initialize the test class."""

        warnings.simplefilter("always", DeprecationWarning)
        self.msg = "sub-module. It will be removed in version 0.3.1+."
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.msg
        return

    def test_ocboundary_deprecation(self):
        """Test that the OCBoundary class has a deprecation warning."""

        with self.assertWarnsRegex(DeprecationWarning,
                                   "".join(["Class moved to `ocbpy._boundary` ",
                                            self.msg])):
            ocbpy.ocboundary.OCBoundary()

        return

    def test_retrieve_all_good_indices(self):
        """Test that the retrieve_all_good_indices function is deprecated."""

        ocb = ocbpy.OCBoundary(filename=path.join(path.dirname(ocbpy.__file__),
                                                  "tests", "test_data",
                                                  "test_north_circle"),
                               instrument="image")

        with self.assertWarnsRegex(DeprecationWarning, "".join(
                ["Function moved to `ocbpy.cycle_boundary` ", self.msg])):
            ocbpy.ocboundary.retrieve_all_good_indices(ocb)

        del ocb
        return

    def test_match_data_ocb(self):
        """Test that the retrieve_all_good_indices function is deprecated."""

        ocb = ocbpy.OCBoundary(filename=path.join(path.dirname(ocbpy.__file__),
                                                  "tests", "test_data",
                                                  "test_north_circle"),
                               instrument="image")

        with self.assertWarnsRegex(DeprecationWarning, "".join(
                ["Function moved to `ocbpy.cycle_boundary` ", self.msg])):
            ocbpy.ocboundary.match_data_ocb(ocb, ocb.dtime)

        del ocb
        return


class TestOCBMatchData(TestCycleMatchData):
    """Unit tests for the deprecated `match_data_ocb` function."""

    def setUp(self):
        """Initialize the test environment."""
        warnings.simplefilter("ignore", DeprecationWarning)
        set_north = {"filename": path.join(path.dirname(ocbpy.__file__),
                                           "tests", "test_data",
                                           "test_north_circle"),
                     "instrument": "image"}
        self.ocb = ocbpy.OCBoundary(**set_north)
        self.ocb.rec_ind = -1
        self.idat = 0
        self.test_func = ocbpy.ocboundary.match_data_ocb

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


class TestOCBGoodIndices(TestCycleGoodIndices):
    """Unit tests for the deprecated `retrieve_all_good_indices` function."""

    def setUp(self):
        """Initialize the test environment."""
        warnings.simplefilter("ignore", DeprecationWarning)
        set_north = {"filename": path.join(path.dirname(ocbpy.__file__),
                                           "tests", "test_data",
                                           "test_north_circle"),
                     "instrument": "image"}
        self.ocb = ocbpy.OCBoundary(**set_north)
        self.ocb.rec_ind = -1
        self.test_func = ocbpy.ocboundary.retrieve_all_good_indices

        del set_north
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.ocb, self.test_func
        return
