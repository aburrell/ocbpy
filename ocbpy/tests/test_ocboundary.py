#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the deprecated ocboundary sub-module."""

import datetime as dt
from os import path
import unittest
import warnings

import ocbpy


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
