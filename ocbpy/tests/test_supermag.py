#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the SuperMAG instrument functions."""
import datetime as dt
import filecmp
import numpy as np
import os
import unittest

import ocbpy
import ocbpy.instruments.supermag as ocb_ismag
from ocbpy.instruments import general


class TestSuperMAG2AsciiMethods(unittest.TestCase):
    """Unit tests for the SuperMAG ASCII functions."""

    def setUp(self):
        """Initialize the setup for SuperMAG processing unit tests."""

        self.ocb_dir = os.path.dirname(ocbpy.__file__)
        self.test_ocb = os.path.join(self.ocb_dir, "tests", "test_data",
                                     "test_north_circle")
        self.test_eab = os.path.join(self.ocb_dir, "tests", "test_data",
                                     "test_north_eab")
        self.test_file = os.path.join(self.ocb_dir, "tests", "test_data",
                                      "test_hemi_smag")
        self.test_eq_file = os.path.join(self.ocb_dir, "tests", "test_data",
                                         "test_eq_smag")
        self.test_output_north = os.path.join(self.ocb_dir, "tests",
                                              "test_data", "out_smag")
        self.test_unscaled_north = os.path.join(self.ocb_dir, "tests",
                                                "test_data",
                                                "out_smag_unscaled")
        self.test_output_dual = os.path.join(self.ocb_dir, "tests",
                                             "test_data", "out_dual_smag")
        self.test_output_south = os.path.join(self.ocb_dir, "tests",
                                              "test_data", "out_south_smag")
        self.temp_output = os.path.join(self.ocb_dir, "tests", "test_data",
                                        "temp_smag")
        return

    def tearDown(self):
        """Clean up the test environment."""
        if os.path.isfile(self.temp_output):
            os.remove(self.temp_output)

        del self.test_file, self.test_output_north, self.test_ocb
        del self.test_output_south, self.temp_output, self.test_eq_file
        del self.test_eab, self.test_output_dual, self.test_unscaled_north
        return

    def test_deprecation_kwargs(self):
        """Test the SuperMAG deprecated kwargs."""
        # Set the deprecated keyword arguments with standard values
        dep_inputs = {"min_sectors": 7, "rcent_dev": 8.0, "max_r": 23.0,
                      "min_r": 10.0}

        # Cycle through the keyword arguments that should raise a warning
        for dkey in dep_inputs.keys():
            kwargs = {dkey: dep_inputs[dkey]}
            with self.subTest(kwargs=kwargs):
                with self.assertWarnsRegex(DeprecationWarning,
                                           "Deprecated kwarg will be removed"):
                    ocb_ismag.supermag2ascii_ocb(
                        self.test_file, self.temp_output,
                        ocbfile=self.test_ocb, instrument='image',
                        hemisphere=1, **kwargs)
        return

    def test_supermag2ascii_ocb_choose_north(self):
        """Test SuperMAG data processing for a mixed file choosing north."""

        ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output,
                                     ocbfile=self.test_ocb, instrument='image',
                                     hemisphere=1)

        # Compare created file to stored test file
        self.assertTrue(filecmp.cmp(self.test_output_north,
                                    self.temp_output, shallow=False))
        return

    def test_supermag2ascii_north_from_ocb_w_wo_scaling(self):
        """Test SuperMAG North from OCBoundary with and w/o scaling."""

        subtests = [(self.test_output_north, {}),
                    (self.test_unscaled_north, {'scale_func': None})]

        ocb = ocbpy.OCBoundary(filename=self.test_ocb, instrument='image',
                               hemisphere=1)

        for val in subtests:
            with self.subTest(val=val):
                ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output,
                                             ocb=ocb, hemisphere=0, **val[1])

                # Compare created file to stored test file
                self.assertTrue(filecmp.cmp(val[0], self.temp_output,
                                            shallow=False))
        return

    def test_supermag2ascii_ocb_choose_south(self):
        """Test SuperMAG data processing for a mixed file choosing south."""

        ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output,
                                     ocbfile=self.test_ocb, instrument='image',
                                     hemisphere=-1)

        # Compare created file to stored test file
        self.assertTrue(filecmp.cmp(self.test_output_south,
                                    self.temp_output, shallow=False))
        return

    def test_supermag2ascii_ocb_eq(self):
        """Test hemisphere choice with southern and equatorial data."""

        ocb_ismag.supermag2ascii_ocb(self.test_eq_file, self.temp_output,
                                     ocbfile=self.test_ocb, instrument='image',
                                     hemisphere=0)

        # Compare created file to stored test file
        self.assertTrue(filecmp.cmp(self.test_output_south,
                                    self.temp_output, shallow=False))
        return

    def test_supermag2ascii_hemi_options(self):
        """Test SuperMAG conversion with different hemisphere options."""
        # Initialize the subTest input
        subtests = [(self.test_eq_file, self.test_output_south,
                     {"ocbfile": self.test_ocb, "instrument": 'image'}),
                    (self.test_file, self.test_output_south,
                     {'ocbfile': self.test_ocb, 'instrument': 'image',
                      'hemisphere': -1}),
                    (self.test_file, self.test_output_north,
                     {'ocb': ocbpy.OCBoundary(
                         filename=self.test_ocb, instrument='image',
                         hemisphere=1)}),
                    (self.test_file, self.test_output_north,
                    {'ocbfile': self.test_ocb, 'instrument': 'image',
                     'hemisphere': 1}),
                    (self.test_file, self.test_output_north,
                     {'ocb': ocbpy.EABoundary(
                         filename=self.test_ocb, instrument='image',
                         hemisphere=1, boundary_lat=74.0)}),
                    (self.test_file, self.test_output_dual,
                     {"ocb": ocbpy.DualBoundary(
                         ocb_filename=self.test_ocb, ocb_instrument='image',
                         eab_filename=self.test_eab, eab_instrument='image',
                         hemisphere=1)})]

        for val in subtests:
            with self.subTest(val=val):
                ocb_ismag.supermag2ascii_ocb(val[0], self.temp_output,
                                             **val[2])

                # Compare created file to stored test file
                self.assertTrue(filecmp.cmp(val[1], self.temp_output,
                                            shallow=False))
        return

    def test_supermag2ascii_ocb_bad_hemi(self):
        """Test the failure caused by not choosing a hemisphere at all."""

        with self.assertRaisesRegex(ValueError, "from both hemispheres"):
            ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output,
                                         ocbfile=self.test_ocb,
                                         instrument='image')
        return

    def test_supermag2ascii_ocb_bad_output(self):
        """Test the failure caused by bad output name (directory)."""
        # Run command that will fail to output a file.  Error message changes
        # based on the operating system
        with self.assertRaises(IOError):
            ocb_ismag.supermag2ascii_ocb(self.test_file, "/",
                                         instrument='image', hemisphere=1,
                                         ocbfile=self.test_ocb)
        return

    def test_supermag2ascii_ioerr_messages(self):
        """Test the failures that produce reliable IOError messages."""
        for val in [("SuperMAG file cannot be opened",
                     ["fake_file", "fake_out"]),
                    ("output filename is not a string", [self.test_file, 1])]:
            with self.subTest(val=val):
                with self.assertRaisesRegex(IOError, val[0]):
                    ocb_ismag.supermag2ascii_ocb(*val[1],
                                                 ocbfile=self.test_ocb)
        return

    def test_supermag2ascii_ocb_bad_ocb(self):
        """Test the SuperMAG conversion with a bad ocb file."""
        ocb_ismag.supermag2ascii_ocb(self.test_file, "fake_out",
                                     ocbfile="fake_ocb", hemisphere=1)

        # Compare created file to stored test file
        self.assertFalse(general.test_file("fake_out"))
        return


class TestSuperMAGLoadMethods(unittest.TestCase):
    """Test the SuperMAG loading functions."""

    def setUp(self):
        """Initialize the test set up."""

        self.ocb_dir = os.path.dirname(ocbpy.__file__)
        self.test_file = os.path.join(self.ocb_dir, "tests", "test_data",
                                      "test_smag")
        self.test_vals = {'BE': -6.0, 'BN': -23.6, 'BZ': -25.2, 'DAY': 5,
                          'DEC': 17.13, 'HOUR': 13, 'MIN': 40, 'MLAT': 77.22,
                          'DATETIME': dt.datetime(2000, 5, 5, 13, 40, 30),
                          'MLT': 15.86, 'MONTH': 5, 'NST': 2, 'SEC': 30,
                          'SML': -195, 'SMU': 124, 'STID': "THL", 'SZA': 76.97,
                          'YEAR': 2000}
        self.out = list()
        self.assertTrue(os.path.isfile(self.test_file))
        return

    def tearDown(self):
        """Tear down the test environment."""
        del self.test_file, self.out, self.test_vals
        return

    def test_load_supermag_ascii_data(self):
        """Test the routine to load the SuperMAG data."""
        self.out = ocb_ismag.load_supermag_ascii_data(self.test_file)

        # Test to see that the data keys are all in the header
        self.assertListEqual(sorted([kk for kk in self.test_vals.keys()]),
                             sorted(list(self.out[1].keys())))

        # Test the length of the data file
        self.assertEqual(self.out[1]['MLT'].shape[0], 4)

        # Test the values of the last data line
        for kk in self.test_vals.keys():
            self.assertEqual(self.out[1][kk][1], self.test_vals[kk])
        return

    def test_load_failures(self):
        """Test graceful failures with different bad file inputs."""

        for val in ['fake_file', os.path.join(self.ocb_dir, "test",
                                              "test_data", "test_vort")]:
            with self.subTest(val=val):
                self.out = ocb_ismag.load_supermag_ascii_data(val)

                self.assertListEqual(self.out[0], list())
                self.assertDictEqual(self.out[1], dict())
        return
