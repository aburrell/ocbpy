#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the boundary EABoundary class."""

import datetime
from io import StringIO
import logging
import numpy
from os import path
import platform
import unittest

import ocbpy
from . import test_boundary_ocboundary as test_ocb


class TestDualBoundaryLogFailure(unittest.TestCase):
    """Test the logging messages raised by the DualBoundary class."""

    def setUp(self):
        """Initialize the test class."""
        self.lwarn = ""
        self.lout = ""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)
        return

    def tearDown(self):
        """Tear down the test case."""
        del self.lwarn, self.lout, self.log_capture
        return

    def test_bad_instrument_name(self):
        """Test OCB initialization with bad instrument name."""
        self.lwarn = "OCB instrument must be a string"

        # Initialize the DualBoundary class with bad instrument names
        for bad_inst in [1, None, True]:
            for btype in ["eab", "ocb"]:
                # Define the kwarg input
                val = {"_".join([btype, "instrument"]): bad_inst}
                with self.subTest(val=val):
                    # Initalize the DualBoundary class
                    bound = ocbpy.DualBoundary(**val)
                    subclass = getattr(bound, btype)

                    # Test the values for the sub-class
                    self.assertIsNone(subclass.filename)
                    self.assertIsNone(subclass.instrument)

                    self.lout = self.log_capture.getvalue()

                    # Test logging error message for each bad initialization
                    self.assertRegex(self.lout, self.lwarn)
        return

    def test_bad_filename(self):
        """Test initialization with a bad default file/instrument pairing."""
        self.lwarn = "name provided is not a file\ncannot open OCB file [hi]"

        # Try to load data with a non-existant file name
        for btype in ["eab", "ocb"]:
            # Define the kwarg input
            val = {"_".join([btype, "filename"]): "hi"}
            with self.subTest(val=val):
                # Initalize the DualBoundary class
                bound = ocbpy.DualBoundary(**val)
                subclass = getattr(bound, btype)

                # Test the values for the sub-class
                self.assertIsNone(subclass.filename)

                self.lout = self.log_capture.getvalue()

                # Test logging error message for each bad initialization
                self.assertTrue(
                    self.lout.find(self.lwarn) >= 0,
                    msg="logging output {:} != expected output {:}".format(
                        repr(self.lout), repr(self.lwarn)))

        return


class TestDualBoundaryInstruments(test_ocb.TestOCBoundaryInstruments):
    """Test the DualBoundary handling of different instruments."""

    def setUp(self):
        """Initialize the instrument information."""
        self.test_class = ocbpy.DualBoundary
        self.test_dir = path.join(path.dirname(ocbpy.__file__), "tests",
                                  "test_data")
        self.inst_attrs = {"image": ["year", "soy", "num_sectors", "a",
                                     "r_err", "fom"],
                           "ampere": ["date", "time", "x", "y", "fom"],
                           "dmsp-ssj": ["date", "time", "sc", "x", "y", "fom",
                                        "x_1", "x_2", "y_1", "y_2"]}
        self.not_attrs = {"image": ["date", "time", "x", "y", "x_1", "x_2",
                                    "y_1", "y_2", "sc"],
                          "ampere": ["year", "soy", "x_1", "y_1", "x_2",
                                     "y_2", "sc", "num_sectors", "a",
                                     "r_err"],
                          "dmsp-ssj": ["year", "soy", "num_sectors", "a",
                                       "r_err"]}
        self.inst_init = [{"eab_instrument": "dmsp-ssj", "hemisphere": 1,
                           "eab_filename": path.join(self.test_dir,
                                                     "dmsp-ssj_north_out.eab"),
                           "ocb_instrument": "image",
                           "ocb_filename": path.join(self.test_dir,
                                                     "test_north_circle")},
                          {"eab_instrument": "dmsp-ssj", "hemisphere": 1,
                           "eab_filename": path.join(self.test_dir,
                                                     "dmsp-ssj_north_out.eab"),
                           "ocb_instrument": "dmsp-ssj",
                           "ocb_filename": path.join(self.test_dir,
                                                     "dmsp-ssj_north_out.ocb")},
                          {"eab_instrument": "dmsp-ssj", "hemisphere": -1,
                           "eab_filename": path.join(self.test_dir,
                                                     "dmsp-ssj_south_out.eab"),
                           "ocb_instrument": "ampere",
                           "ocb_filename": path.join(self.test_dir,
                                                     "test_south_circle")}]
        self.ocb = None
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.test_dir, self.inst_attrs, self.inst_init, self.ocb
        del self.test_class
        return


class TestDualBoundaryMethodsGeneral(test_ocb.TestOCBoundaryMethodsGeneral):
    """Test the DualBoundary general methods."""

    def setUp(self):
        """Initialize the test environment."""
        self.test_class = ocbpy.DualBoundary
        test_dir = path.join(path.dirname(ocbpy.__file__), "tests", "test_data")
        self.set_empty = {"ocb_filename": path.join(test_dir, "test_empty"),
                          "eab_filename": path.join(test_dir, "test_empty"),
                          "ocb_instrument": "image", "eab_instrument": "image",
                          "hemisphere": 1}
        self.set_default = {"ocb_filename":
                            path.join(test_dir, "dmsp-ssj_north_out.ocb"),
                            "eab_filename":
                            path.join(test_dir, "dmsp-ssj_north_out.eab"),
                            "ocb_instrument": "dmsp-ssj",
                            "eab_instrument": "dmsp-ssj", "hemisphere": 1,
                            "max_delta": 600}
        self.ocb = None
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.set_empty, self.set_default, self.ocb
        return

    def test_repr_string(self):
        """Test __repr__ method string."""
        # Initalize the class object
        self.ocb = self.test_class(**self.set_default)

        # Get the representation of the class object and split by subclasses
        rocb = repr(self.ocb).split("ocb=")

        # Test the name of the repr object
        self.assertRegex(rocb[0], self.test_class.__name__)

        # Test each set kwarg has the expected value
        for val in self.set_default.keys():
            with self.subTest(val=val):
                i = 0 if val.find("eab") == 0 else 1

                # Construct the expected string
                if i == 0 or val.find("ocb") == 0:
                    test_str = "=".join([val.split("_")[-1],
                                         repr(self.set_default[val])])
                else:
                    test_str = "=".join([val, repr(self.set_default[val])])

                # Windows has trouble recognizing the filename in Regex
                if(test_str.find("filename")
                   and platform.system().lower() == "windows"):
                    test_str = "filename="

                # Test the correct part of the repr output.
                self.assertRegex(rocb[i], test_str)
        return

    def test_short_str(self):
        """Test the default class print output."""
        self.ocb = self.test_class(**self.set_default)
        self.ocb.records = 1

        self.assertRegex(self.ocb.__str__(), "1 good boundary pairs from")
        return

    def test_bad_rfunc_inst(self):
        """Test failure setting default rfunc for unknown instrument."""

        for bound in ['eab', 'ocb']:
            with self.subTest(bound=bound):
                self.set_empty["_".join([bound, "instrument"])] = "bad"

                with self.assertRaisesRegex(ValueError, "unknown instrument"):
                    self.ocb = self.test_class(**self.set_empty)
        return

    def test_no_file_str(self):
        """Test the unset class print output."""

        for bound in ['eab', 'ocb', 'both']:
            out_str = []
            # Update the kwarg input
            if bound in ['eab', 'both']:
                self.set_default['eab_filename'] = None
                out_str.append("No {:s} file specified".format(
                    ocbpy.EABoundary.__name__))
            if bound in ['ocb', 'both']:
                self.set_default['ocb_filename'] = None
                out_str.append("No {:s} file specified".format(
                    ocbpy.OCBoundary.__name__))

            # Initalise the object
            self.ocb = self.test_class(**self.set_default)

            for val in out_str:
                with self.subTest(val=val):
                    # Test the output string
                    self.assertRegex(self.ocb.__str__(), val)
        return

    def test_nofile_init(self):
        """Ensure that the class can be initialised without loading a file."""
        self.ocb = self.test_class(eab_filename=None, ocb_filename=None)

        self.assertIsNone(self.ocb.eab.filename)
        self.assertIsNone(self.ocb.ocb.filename)
        self.assertIsNone(self.ocb.dtime)
        self.assertEqual(self.ocb.records, 0)
        return
