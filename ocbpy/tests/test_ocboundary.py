#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the deprecated ocboundary sub-module."""

import datetime as dt
from io import StringIO
import logging
import numpy
from os import path
import unittest
import warnings

import ocbpy
from .test_cycle_boundary import TestCycleMatchData, TestCycleGoodIndices
from . import test_boundary_ocb as test_ocb


class TestInternalOCBoundaryDeprecations(test_ocb.TestOCBoundaryDeprecations):
    """Test the deprecation warnings within the OCBoundary class."""

    def setUp(self):
        """Initialize the test environment."""
        self.test_class = ocbpy.ocboundary.OCBoundary
        test_dir = path.join(path.dirname(ocbpy.__file__), "tests",
                             "test_data")
        self.inst_init = {"instrument": "image", "hemisphere": 1,
                          "filename": path.join(test_dir,
                                                "test_north_circle")}

    def tearDown(self):
        """Clean up the test environment."""
        del self.test_class, self.inst_init


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
        self.rec_ind = 27
        self.rec_ind2 = 31
        self.del_time = 600
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
        self.rec_ind = 27
        self.rec_ind2 = 31
        self.test_func = ocbpy.ocboundary.retrieve_all_good_indices

        del set_north
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.ocb, self.test_func, self.rec_ind, self.rec_ind2
        return


class TestOldOCBoundaryLogFailure(test_ocb.TestOCBoundaryLogFailure):
    """Test the logging messages raised by the deprecated OCBoundary class."""

    def setUp(self):
        """Initialize the test class."""
        warnings.simplefilter("ignore", DeprecationWarning)
        self.test_class = ocbpy.ocboundary.OCBoundary
        test_dir = path.join(path.dirname(ocbpy.__file__), "tests",
                             "test_data")
        self.inst_init = {"instrument": "image", "hemisphere": 1,
                          "filename": path.join(test_dir,
                                                "test_north_circle")}

        self.lwarn = ""
        self.lout = ""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)
        return

    def tearDown(self):
        """Tear down the test case."""
        del self.lwarn, self.lout, self.log_capture, self.test_class
        return


class TestOldOCBoundaryInstruments(test_ocb.TestOCBoundaryInstruments):
    def setUp(self):
        """Initialize the instrument information."""
        warnings.simplefilter("ignore", DeprecationWarning)
        self.test_class = ocbpy.ocboundary.OCBoundary
        self.test_dir = path.join(path.dirname(ocbpy.__file__), "tests",
                                  "test_data")
        self.inst_attrs = {"image": ["year", "soy", "num_sectors", "a", "fom",
                                     "r_err"],
                           "ampere": ["date", "time", "x", "y", "fom"],
                           "dmsp-ssj": ["date", "time", "sc", "x", "y", "fom",
                                        "x_1", "x_2", "y_1", "y_2"]}
        self.not_attrs = {"image": ["date", "time", "x", "y", "x_1",
                                    "x_2", "y_1", "y_2", "sc"],
                          "ampere": ["year", "soy", "x_1", "y_1", "x_2",
                                     "y_2", "sc", "num_sectors", "a",
                                     "r_err"],
                          "dmsp-ssj": ["year", "soy", "num_sectors", "a",
                                       "r_err"]}
        self.inst_init = [{"instrument": "image", "hemisphere": 1,
                           "filename": path.join(self.test_dir,
                                                 "test_north_circle")},
                          {"instrument": "dmsp-ssj", "hemisphere": 1,
                           "filename": path.join(self.test_dir,
                                                 "dmsp-ssj_north_out.ocb")},
                          {"instrument": "dmsp-ssj", "hemisphere": -1,
                           "filename": path.join(self.test_dir,
                                                 "dmsp-ssj_south_out.ocb")},
                          {"instrument": "ampere", "hemisphere": -1,
                           "filename": path.join(self.test_dir,
                                                 "test_south_circle")}]
        self.ocb = None
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.test_dir, self.inst_attrs, self.inst_init, self.ocb
        del self.test_class
        return


class TestOldOCBoundaryMethodsGeneral(test_ocb.TestOCBoundaryMethodsGeneral):
    def setUp(self):
        """Initialize the test environment."""
        warnings.simplefilter("ignore", DeprecationWarning)
        self.test_class = ocbpy.ocboundary.OCBoundary
        ocb_dir = path.dirname(ocbpy.__file__)
        self.set_empty = {"filename": path.join(ocb_dir, "tests", "test_data",
                                                "test_empty"),
                          "instrument": "image"}
        self.set_default = {"filename": path.join(ocb_dir, "tests",
                                                  "test_data",
                                                  "test_north_circle"),
                            "instrument": "image"}
        self.assertTrue(path.isfile(self.set_empty['filename']))
        self.assertTrue(path.isfile(self.set_default['filename']))
        self.ocb = None
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.set_empty, self.set_default, self.ocb, self.test_class
        return


class TestOldOCBoundaryMethodsNorth(test_ocb.TestOCBoundaryMethodsNorth):
    """Unit tests for the OCBoundary class in the northern hemisphere."""

    def setUp(self):
        """Initialize the test environment."""
        warnings.simplefilter("ignore", DeprecationWarning)
        self.test_class = ocbpy.ocboundary.OCBoundary
        self.set_north = {'filename': path.join(path.dirname(ocbpy.__file__),
                                                "tests", "test_data",
                                                "test_north_circle"),
                          'instrument': 'image'}
        self.assertTrue(path.isfile(self.set_north['filename']))
        self.ocb = self.test_class(**self.set_north)
        self.ocb.rec_ind = 27
        self.ref_boundary = 74.0

        self.mlt = numpy.linspace(0.0, 24.0, num=6)
        self.lat = numpy.linspace(0.0, 90.0, num=len(self.mlt))
        self.ocb_lat = [numpy.nan, 11.25588586, 30.35153908, 47.0979063,
                        66.59889231, 86.86586231]
        self.ocb_mlt = [numpy.nan, 4.75942194, 9.76745427, 14.61843964,
                        19.02060793, 17.832]
        self.r_corr = 0.0
        self.out = None
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.ocb, self.set_north, self.mlt, self.lat, self.ocb_lat
        del self.ocb_mlt, self.r_corr, self.out, self.test_class
        del self.ref_boundary
        return


class TestOldOCBoundaryMethodsSouth(test_ocb.TestOCBoundaryMethodsSouth):
    """Unit tests for the OCBoundary methods in the southern hemisphere."""

    def setUp(self):
        """Initialize the test environment."""
        warnings.simplefilter("ignore", DeprecationWarning)
        self.test_class = ocbpy.ocboundary.OCBoundary

        self.set_south = {"filename": path.join(path.dirname(ocbpy.__file__),
                                                "tests", "test_data",
                                                "test_south_circle"),
                          "instrument": "ampere",
                          "hemisphere": -1,
                          "rfunc": ocbpy.ocb_correction.circular}
        self.ocb = self.test_class(**self.set_south)
        self.ocb.rec_ind = 8
        self.ref_boundary = -74.0

        self.mlt = numpy.linspace(0.0, 24.0, num=6)
        self.lat = numpy.linspace(-90.0, 0.0, num=len(self.mlt))
        self.ocb_lat = [-86.8, -58.14126906, -30.46277504, -5.44127327,
                        22.16097829, numpy.nan]
        self.ocb_mlt = [6.0, 4.91857824, 9.43385497, 14.28303702, 19.23367655,
                        numpy.nan]
        self.r_corr = 0.0
        self.out = None
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.ocb, self.set_south, self.mlt, self.lat, self.ocb_lat
        del self.ocb_mlt, self.r_corr, self.out, self.test_class
        del self.ref_boundary
        return


class TestOldOCBoundaryFailure(test_ocb.TestOCBoundaryFailure):
    """Test the deprecated OCBoundary class failures raise appropriate errors.

    """

    def setUp(self):
        """Set up the test environment."""
        warnings.simplefilter("ignore", DeprecationWarning)
        self.test_class = ocbpy.ocboundary.OCBoundary
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.test_class
        return
