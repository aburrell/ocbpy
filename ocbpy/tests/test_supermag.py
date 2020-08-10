#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
""" Tests the ocb_scaling class and functions
"""
import datetime as dt
import numpy as np
import os
from sys import version_info
import platform
import unittest

if platform.system().lower() != "windows":
    import filecmp

import ocbpy
import ocbpy.instruments.supermag as ocb_ismag
from ocbpy.instruments import general


class TestSuperMAG2AsciiMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the setup for SuperMAG processing unit tests
        """

        self.ocb_dir = os.path.dirname(ocbpy.__file__)
        self.test_ocb = os.path.join(self.ocb_dir, "tests", "test_data",
                                     "test_north_circle")
        self.test_file = os.path.join(self.ocb_dir, "tests", "test_data",
                                      "test_hemi_smag")
        self.test_eq_file = os.path.join(self.ocb_dir, "tests", "test_data",
                                         "test_eq_smag")
        self.test_output_north = os.path.join(self.ocb_dir, "tests",
                                              "test_data", "out_smag")
        self.test_output_south = os.path.join(self.ocb_dir, "tests",
                                              "test_data", "out_south_smag")
        self.temp_output = os.path.join(self.ocb_dir, "tests", "test_data",
                                        "temp_smag")

        # Remove this in 2020
        if version_info.major == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp

    def tearDown(self):
        if os.path.isfile(self.temp_output):
            os.remove(self.temp_output)

        del self.test_file, self.test_output_north, self.test_ocb
        del self.test_output_south, self.temp_output, self.test_eq_file

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_supermag2ascii_ocb_choose_north(self):
        """ Test the SuperMAG data processing for a mixed file choosing north
        """

        ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output,
                                     ocbfile=self.test_ocb, instrument='image',
                                     hemisphere=1)

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows

            ldtype = ['|U50' if i < 2 or i == 3 else float for i in range(19)]
            test_out = np.genfromtxt(self.test_output_north, skip_header=1,
                                     dtype=ldtype)
            temp_out = np.genfromtxt(self.temp_output, skip_header=1,
                                     dtype=ldtype)

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the data in each row
            for i, test_row in enumerate(test_out):
                self.assertListEqual(list(test_row), list(temp_out[i]))

            del ldtype, test_out, temp_out
        else:
            # Compare created file to stored test file
            self.assertTrue(filecmp.cmp(self.test_output_north,
                                        self.temp_output, shallow=False))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_supermag2ascii_north_from_ocb(self):
        """ Test the SuperMAG data processing choosing north from OCBoundary
        """

        ocb = ocbpy.ocboundary.OCBoundary(filename=self.test_ocb,
                                          instrument='image', hemisphere=1)

        ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output, ocb=ocb,
                                     hemisphere=0)

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows

            ldtype = ['|U50' if i < 2 or i == 3 else float for i in range(19)]
            test_out = np.genfromtxt(self.test_output_north, skip_header=1,
                                     dtype=ldtype)
            temp_out = np.genfromtxt(self.temp_output, skip_header=1,
                                     dtype=ldtype)

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the data in each row
            for i, test_row in enumerate(test_out):
                self.assertListEqual(list(test_row), list(temp_out[i]))

            del ldtype, test_out, temp_out
        else:
            # Compare created file to stored test file
            self.assertTrue(filecmp.cmp(self.test_output_north,
                                        self.temp_output, shallow=False))

        del ocb

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_supermag2ascii_ocb_choose_south(self):
        """ Test the SuperMAG data processing for a mixed file choosing south
        """

        ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output,
                                     ocbfile=self.test_ocb, instrument='image',
                                     hemisphere=-1)

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows

            ldtype = ['|U50' if i < 2 or i == 3 else float for i in range(19)]
            test_out = np.genfromtxt(self.test_output_south, skip_header=1,
                                     dtype=ldtype)
            temp_out = np.genfromtxt(self.temp_output, skip_header=1,
                                     dtype=ldtype)

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the data in each row
            for i, test_row in enumerate(test_out):
                self.assertListEqual(list(test_row), list(temp_out[i]))

            del ldtype, test_out, temp_out
        else:
            # Compare created file to stored test file
            self.assertTrue(filecmp.cmp(self.test_output_south,
                                        self.temp_output, shallow=False))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_supermag2ascii_ocb_eq(self):
        """ Test hemisphere choice with southern and equatorial data
        """

        ocb_ismag.supermag2ascii_ocb(self.test_eq_file, self.temp_output,
                                     ocbfile=self.test_ocb, instrument='image',
                                     hemisphere=0)

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows

            ldtype = ['|U50' if i < 2 or i == 3 else float for i in range(19)]
            test_out = np.genfromtxt(self.test_output_south, skip_header=1,
                                     dtype=ldtype)
            temp_out = np.genfromtxt(self.temp_output, skip_header=1,
                                     dtype=ldtype)

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the data in each row
            for i, test_row in enumerate(test_out):
                self.assertListEqual(list(test_row), list(temp_out[i]))

            del ldtype, test_out, temp_out
        else:
            # Compare created file to stored test file
            self.assertTrue(filecmp.cmp(self.test_output_south,
                                        self.temp_output, shallow=False))

    @unittest.skipIf(version_info.major < 3,
                     'Already tested, remove in 2020')
    def test_supermag2ascii_hemi_options(self):
        """ Test SuperMAG conversion with different hemisphere options
        """
        # Initialize the subTest input
        subtests = [(self.test_eq_file, self.test_output_south,
                     {"ocbfile": self.test_ocb, "instrument": 'image'}),
                    (self.test_file, self.test_output_south,
                     {'ocbfile': self.test_ocb, 'instrument': 'image',
                      'hemisphere': -1}),
                    (self.test_file, self.test_output_north,
                     {'ocb': ocbpy.ocboundary.OCBoundary(
                         filename=self.test_ocb, instrument='image',
                         hemisphere=1)}),
                    (self.test_file, self.test_output_north,
                    {'ocbfile': self.test_ocb, 'instrument': 'image',
                     'hemisphere': 1})]

        for val in subtests:
            with self.subTest(val=val):
                ocb_ismag.supermag2ascii_ocb(val[0], self.temp_output,
                                             **val[2])

                if platform.system().lower() == "windows":
                    # filecmp doesn't work on windows

                    ldtype = ['|U50' if i < 2 or i == 3 else float
                              for i in range(19)]
                    test_out = np.genfromtxt(val[1], skip_header=1,
                                             dtype=ldtype)
                    temp_out = np.genfromtxt(self.temp_output, skip_header=1,
                                             dtype=ldtype)

                    # Test the number of rows and columns
                    self.assertTupleEqual(test_out.shape, temp_out.shape)

                    # Test the data in each row
                    for i, test_row in enumerate(test_out):
                        self.assertListEqual(list(test_row), list(temp_out[i]))

                    del ldtype, test_out, temp_out
                else:
                    # Compare created file to stored test file
                    self.assertTrue(filecmp.cmp(val[1], self.temp_output,
                                                shallow=False))

    def test_supermag2ascii_ocb_bad_hemi(self):
        """ Test the failure caused by not choosing a hemisphere at all """

        with self.assertRaisesRegex(ValueError, "from both hemispheres"):
            ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output,
                                         ocbfile=self.test_ocb,
                                         instrument='image')

    def test_supermag2ascii_ocb_bad_output(self):
        """ Test the failure caused by bad output name (directory)
        """
        # Run command that will fail to output a file.  Error message changes
        # based on the operating system
        with self.assertRaises(IOError):
            ocb_ismag.supermag2ascii_ocb(self.test_file, "/",
                                         instrument='image', hemisphere=1,
                                         ocbfile=self.test_ocb)

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_supermag2ascii_ocb_bad_output_str(self):
        """ Test failure caused by an non-string output name
        """
        with self.assertRaisesRegex(IOError,
                                    "output filename is not a string"):
            ocb_ismag.supermag2ascii_ocb(self.test_file, 1,
                                         ocbfile=self.test_ocb)

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_supermag2ascii_ocb_bad_input(self):
        """ Test the failure when a bad SuperMAG file is input
        """
        with self.assertRaisesRegex(IOError, "SuperMAG file cannot be opened"):
            ocb_ismag.supermag2ascii_ocb("fake_file", "fake_out",
                                         ocbfile=self.test_ocb)

    @unittest.skipIf(version_info.major < 3,
                     'Already tested, remove in 2020')
    def test_supermag2ascii_ioerr_messages(self):
        """ Test the failures that produce reliable IOError messages
        """
        for val in [("SuperMAG file cannot be opened",
                     ["fake_file", "fake_out"]),
                    ("output filename is not a string", [self.test_file, 1])]:
            with self.subTest(val=val):
                with self.assertRaisesRegex(IOError, val[0]):
                    ocb_ismag.supermag2ascii_ocb(*val[1],
                                                 ocbfile=self.test_ocb)

    def test_supermag2ascii_ocb_bad_ocb(self):
        """ Test the SuperMAG conversion with a bad ocb file """
        ocb_ismag.supermag2ascii_ocb(self.test_file, "fake_out",
                                     ocbfile="fake_ocb", hemisphere=1)

        # Compare created file to stored test file
        self.assertFalse(general.test_file("fake_out"))


class TestSuperMAGLoadMethods(unittest.TestCase):
    def setUp(self):
        """ Initialize the filenames and data needed to test SuperMAG loading
        """

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

        # Remove this in 2020
        if version_info.major == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp

    def tearDown(self):
        del self.test_file, self.out, self.test_vals

    def test_load_supermag_ascii_data(self):
        """ Test the routine to load the SuperMAG data """
        self.out = ocb_ismag.load_supermag_ascii_data(self.test_file)

        # Test to see that the data keys are all in the header
        self.assertListEqual(sorted([kk for kk in self.test_vals.keys()]),
                             sorted(list(self.out[1].keys())))

        # Test the length of the data file
        self.assertEqual(self.out[1]['MLT'].shape[0], 2)

        # Test the values of the last data line
        for kk in self.test_vals.keys():
            self.assertEqual(self.out[1][kk][-1], self.test_vals[kk])

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_load_failure(self):
        """ Test the routine to load the SuperMAG data for bad filename """
        self.out = ocb_ismag.load_supermag_ascii_data("fake_file")

        # Test to see that the data keys are all in the header
        self.assertListEqual(self.out[0], [])
        self.assertListEqual(list(self.out[1].keys()), [])

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_wrong_load(self):
        """ Test the routine to load the SuperMAG data """
        bad_file = os.path.join(self.ocb_dir, "test", "test_data", "test_vort")
        self.out = ocb_ismag.load_supermag_ascii_data(bad_file)

        self.assertListEqual(self.out[0], list())
        self.assertDictEqual(self.out[1], dict())
        del bad_file

    @unittest.skipIf(version_info.major < 3,
                     'Already tested, remove in 2020')
    def test_load_failures(self):
        """ Test graceful failures with different bad file inputs"""

        for val in ['fake_file', os.path.join(self.ocb_dir, "test",
                                              "test_data", "test_vort")]:
            with self.subTest(val=val):
                self.out = ocb_ismag.load_supermag_ascii_data(val)

                self.assertListEqual(self.out[0], list())
                self.assertDictEqual(self.out[1], dict())
