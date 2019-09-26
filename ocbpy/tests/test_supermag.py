#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
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
from ocbpy.instruments.general import test_file

class TestSuperMAGHemiMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        
        self.ocb_dir = os.path.dirname(ocbpy.__file__)
        self.test_ocb = os.path.join(self.ocb_dir, "tests", "test_data",
                                     "test_north_circle")
        self.test_file = os.path.join(self.ocb_dir, "tests", "test_data",
                                      "test_hemi_smag")
        self.test_output_north = os.path.join(self.ocb_dir, "tests",
                                              "test_data", "out_smag")
        self.test_output_south = os.path.join(self.ocb_dir, "tests",
                                              "test_data", "out_south_smag")
        self.temp_output = os.path.join(self.ocb_dir, "tests", "test_data",
                                        "temp_smag")
        self.assertTrue(os.path.isfile(self.test_file))

        # Remove this in 2020
        if version_info.major == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp

    def tearDown(self):
        if os.path.isfile(self.temp_output):
            os.remove(self.temp_output)

        del self.test_file, self.test_output_north, self.test_ocb
        del self.test_output_south, self.temp_output

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
            temp_out = np.genfromtxt(self.temp_output_north, skip_header=1,
                                     dtype=ldtype)

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the data in each row
            for i,test_row in enumerate(test_out):
                self.assertListEqual(list(test_row), list(temp_out[i]))

            del ldtype, test_out, temp_out
        else:
            # Compare created file to stored test file
            self.assertTrue(filecmp.cmp(self.test_output_north,
                                        self.temp_output, shallow=False))

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
            temp_out = np.genfromtxt(self.temp_output_north, skip_header=1,
                                     dtype=ldtype)

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the data in each row
            for i,test_row in enumerate(test_out):
                self.assertListEqual(list(test_row), list(temp_out[i]))

            del ldtype, test_out, temp_out
        else:
            # Compare created file to stored test file
            self.assertTrue(filecmp.cmp(self.test_output_north,
                                        self.temp_output, shallow=False))

        del ocb

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
            temp_out = np.genfromtxt(self.temp_output_south, skip_header=1,
                                     dtype=ldtype)

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the data in each row
            for i,test_row in enumerate(test_out):
                self.assertListEqual(list(test_row), list(temp_out[i]))

            del ldtype, test_out, temp_out
        else:
            # Compare created file to stored test file
            self.assertTrue(filecmp.cmp(self.test_output_south,
                                        self.temp_output, shallow=False))

    def test_supermag2ascii_ocb_bad_hemi(self):
        """ Test the failure caused by not choosing a hemisphere at all
        """
        # Run command that will fail to output a file.  Error message changes
        # based on the operating system
        with self.assertRaisesRegex(ValueError, "from both hemispheres"):
            ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output,
                                         ocbfile=self.test_ocb,
                                         instrument='image')

                                         
class TestSuperMAGMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        
        self.ocb_dir = os.path.dirname(ocbpy.__file__)
        self.test_ocb = os.path.join(self.ocb_dir, "tests", "test_data",
                                     "test_north_circle")
        self.test_file = os.path.join(self.ocb_dir, "tests", "test_data",
                                      "test_smag")
        self.test_output = os.path.join(self.ocb_dir, "tests", "test_data",
                                        "out_smag")
        self.temp_output = os.path.join(self.ocb_dir, "tests", "test_data",
                                        "temp_smag")
        self.assertTrue(os.path.isfile(self.test_file))

        # Remove this in 2020
        if version_info.major == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp

    def tearDown(self):
        if os.path.isfile(self.temp_output):
            os.remove(self.temp_output)

        del self.test_file, self.test_output, self.test_ocb, self.temp_output

    def test_load_supermag_ascii_data(self):
        """ Test the routine to load the SuperMAG data
        """

        header, data = ocb_ismag.load_supermag_ascii_data(self.test_file)

        # Test to see that the data keys are all in the header
        ktest = ['BE', 'BN', 'BZ', 'DATETIME', 'DAY', 'DEC', 'HOUR', 'MIN',
                 'MLAT', 'MLT', 'MONTH', 'NST', 'SEC', 'SML', 'SMU', 'STID',
                 'SZA', 'YEAR']
        self.assertListEqual(ktest, sorted(list(data.keys())))

        # Test the length of the data file
        self.assertEqual(data['MLT'].shape[0], 2)

        # Test the values of the last data line
        test_vals = {'BE':-6.0, 'BN':-23.6, 'BZ':-25.2, 'DAY':5, 'DEC':17.13, 
                     'DATETIME':dt.datetime(2000,5,5,13,40,30), 'HOUR':13,
                     'MIN':40, 'MLAT':77.22, 'MLT':15.86, 'MONTH':5, 'NST':2,
                     'SEC':30, 'SML':-195, 'SMU':124, 'STID':"THL", 'SZA':76.97,
                     'YEAR':2000}
        for kk in test_vals.keys():
            self.assertEqual(data[kk][-1], test_vals[kk])

        del header, data, ktest, test_vals

    def test_load_failure(self):
        """ Test the routine to load the SuperMAG data for bad filename
        """

        header, data = ocb_ismag.load_supermag_ascii_data("fake_file")

        # Test to see that the data keys are all in the header
        self.assertListEqual(header, [])
        self.assertListEqual(list(data.keys()), [])

        del header, data

    def test_wrong_load(self):
        """ Test the routine to load the SuperMAG data
        """
        bad_file = os.path.join(self.ocb_dir, "test", "test_data", "test_vort")
        header, data = ocb_ismag.load_supermag_ascii_data(bad_file)

        self.assertListEqual(header, list())
        self.assertDictEqual(data, dict())
        del bad_file, data, header

    def test_supermag2ascii_ocb(self):
        """ Test the conversion of SuperMAG data from AACGM coordinates into
        OCB coordinates
        """

        ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output,
                                     ocbfile=self.test_ocb, instrument='image')

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows

            ldtype = ['|U50' if i < 2 or i == 3 else float for i in range(19)]
            test_out = np.genfromtxt(self.test_output, skip_header=1,
                                     dtype=ldtype)
            temp_out = np.genfromtxt(self.temp_output, skip_header=1,
                                     dtype=ldtype)

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the data in each row
            for i,test_row in enumerate(test_out):
                self.assertListEqual(list(test_row), list(temp_out[i]))

            del ldtype, test_out, temp_out
        else:
            # Compare created file to stored test file
            self.assertTrue(filecmp.cmp(self.test_output, self.temp_output,
                                        shallow=False))

    def test_supermag2ascii_ocb_bad_output(self):
        """ Test the failure caused by bad output name
        """
        # Run command that will fail to output a file.  Error message changes
        # based on the operating system
        with self.assertRaises(IOError):
            ocb_ismag.supermag2ascii_ocb(self.test_file, "/",
                                         instrument='image',
                                         ocbfile=self.test_ocb)

    def test_supermag2ascii_ocb_bad_output_str(self):
        """ Test failure caused by an non-string output name
        """
        # Run command that will fail to output a file

        with self.assertRaisesRegex(IOError, "output filename is not a string"):
            ocb_ismag.supermag2ascii_ocb(self.test_file, 1,
                                         ocbfile=self.test_ocb)

    def test_supermag2ascii_ocb_bad_input(self):
        """ Test the conversion of SuperMAG data from AACGM coordinates into
        OCB coordinates
        """

        with self.assertRaisesRegexp(IOError, "SuperMAG file cannot be opened"):
            ocb_ismag.supermag2ascii_ocb("fake_file", "fake_out",
                                         ocbfile=self.test_ocb)

    def test_supermag2ascii_ocb_bad_ocb(self):
        """ Test the conversion of SuperMAG data from AACGM coordinates into
        OCB coordinates
        """

        ocb_ismag.supermag2ascii_ocb(self.test_file, "fake_out",
                                     ocbfile="fake_ocb")

        # Compare created file to stored test file
        self.assertFalse(test_file("fake_out"))
        

if __name__ == '__main__':
    unittest.main()
