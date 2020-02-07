#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocb_scaling class and functions
"""
import datetime as dt
from io import StringIO
import logging
import numpy as np
import os
from sys import version_info
import platform
import unittest

if platform.system().lower() != "windows":
    import filecmp

import ocbpy
import ocbpy.instruments.vort as ocb_ivort

class TestVortLogWarnings(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        
        self.ocb_dir = os.path.dirname(ocbpy.__file__)
        self.test_file = os.path.join(self.ocb_dir, "tests", "test_data",
                                      "test_vort")
        self.test_ocb = os.path.join(self.ocb_dir, "tests", "test_data",
                                     "test_north_circle")
        self.temp_output = os.path.join(self.ocb_dir, "tests", "test_data",
                                        "temp_vort")
        self.assertTrue(os.path.isfile(self.test_file))

        self.lwarn = u''
        self.lout = u''
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

    def tearDown(self):
        """ Tear down the test case"""
        if os.path.isfile(self.temp_output):
            os.remove(self.temp_output)

        del self.test_file, self.temp_output, self.test_ocb, self.ocb_dir
        del self.lwarn, self.lout, self.log_capture

    def test_vort2ascii_ocb_wrong_hemi(self):
        """ Test the vorticity failure of choosing the wrong hemisphere """

        self.lwarn = u'No southern hemisphere data in file'
        ocb_ivort.vort2ascii_ocb(self.test_file, self.temp_output,
                                 ocbfile=self.test_ocb, instrument='image',
                                 hemisphere=-1)
        self.lout = self.log_capture.getvalue()

        # Test logging error message
        self.assertTrue(self.lout.find(self.lwarn) >= 0)


class TestVortHemiMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        
        self.ocb_dir = os.path.dirname(ocbpy.__file__)
        self.test_ocb = os.path.join(self.ocb_dir, "tests", "test_data",
                                     "test_north_circle")
        self.test_file = os.path.join(self.ocb_dir, "tests", "test_data",
                                      "test_hemi_vort")
        self.test_eq_file = os.path.join(self.ocb_dir, "tests", "test_data",
                                      "test_eq_hemi_vort")
        self.test_output_north = os.path.join(self.ocb_dir, "tests",
                                              "test_data",  "out_vort")
        self.test_output_south = os.path.join(self.ocb_dir, "tests",
                                              "test_data",  "out_south_vort")
        self.temp_output = os.path.join(self.ocb_dir, "tests", "test_data",
                                        "temp_vort")
        self.test_vals = {'CENTRE_MLAT':67.27, 'DAY':5, 'MLT':3.127,
                          'UTH':13.65, 'VORTICITY':0.0020967, 'YEAR':2000,
                          'DATETIME':dt.datetime(2000,5,5,13,39,00), 'MONTH':5}
        self.assertTrue(os.path.isfile(self.test_file))

        # Remove in 2020
        if version_info.major == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp

    def tearDown(self):

        if os.path.isfile(self.temp_output):
            os.remove(self.temp_output)

        del self.test_file, self.temp_output, self.test_ocb, self.ocb_dir
        del self.test_output_north, self.test_output_south, self.test_eq_file

    def test_vort2ascii_ocb_north(self):
        """ Test vorticity data conversion selecting just the north
        """

        ocb_ivort.vort2ascii_ocb(self.test_file, self.temp_output,
                                 ocbfile=self.test_ocb, instrument='image',
                                 hemisphere=1)

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows

            ldtype = ['|U50' if i < 2 else float for i in range(5)]
            test_out = np.genfromtxt(self.test_output_north, skip_header=1,
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
            self.assertTrue(filecmp.cmp(self.test_output_north,
                                        self.temp_output, shallow=False))

    def test_vort2ascii_north_from_ocb(self):
        """ Test vorticity data conversion selecting the north from OCBoundary
        """

        ocb = ocbpy.ocboundary.OCBoundary(filename=self.test_ocb,
                                          instrument='image', hemisphere=1)
        ocb_ivort.vort2ascii_ocb(self.test_file, self.temp_output, ocb=ocb,
                                 hemisphere=0)

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows

            ldtype = ['|U50' if i < 2 else float for i in range(5)]
            test_out = np.genfromtxt(self.test_output_north, skip_header=1,
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
            self.assertTrue(filecmp.cmp(self.test_output_north,
                                        self.temp_output, shallow=False))

    def test_vort2ascii_south_from_vortfile(self):
        """ Test vorticity data conversion selecting the north from OCBoundary
        """

        ocb_ivort.vort2ascii_ocb(self.test_eq_file, self.temp_output,
                                 ocbfile=self.test_ocb, instrument='image')

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows

            ldtype = ['|U50' if i < 2 else float for i in range(5)]
            test_out = np.genfromtxt(self.test_output_south, skip_header=1,
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
            self.assertTrue(filecmp.cmp(self.test_output_south,
                                        self.temp_output, shallow=False))
    def test_vort2ascii_ocb_south(self):
        """ Test vorticity data conversion selecting just the south
        """

        ocb_ivort.vort2ascii_ocb(self.test_file, self.temp_output,
                                 ocbfile=self.test_ocb, instrument='image',
                                 hemisphere=-1)

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows

            ldtype = ['|U50' if i < 2 else float for i in range(5)]
            test_out = np.genfromtxt(self.test_output_south, skip_header=1,
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
            self.assertTrue(filecmp.cmp(self.test_output_south,
                                        self.temp_output, shallow=False))

    def test_vort2ascii_ocb_bad_hemi(self):
        """ Test the failure by not choosing a hemisphere at all """

        with self.assertRaisesRegex(ValueError, "from both hemispheres"):
            ocb_ivort.vort2ascii_ocb(self.test_file, self.temp_output,
                                     ocbfile=self.test_ocb, instrument='image')

class TestVortMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        
        self.ocb_dir = os.path.dirname(ocbpy.__file__)
        self.test_empty = os.path.join(self.ocb_dir, "tests", "test_data",
                                       "test_empty")
        self.test_ocb = os.path.join(self.ocb_dir, "tests", "test_data",
                                     "test_north_circle")
        self.test_file = os.path.join(self.ocb_dir, "tests", "test_data",
                                      "test_vort")
        self.test_output = os.path.join(self.ocb_dir, "tests", "test_data",
                                        "out_vort")
        self.temp_output = os.path.join(self.ocb_dir, "tests", "test_data",
                                        "temp_vort")
        self.test_vals = {'CENTRE_MLAT':67.27, 'DAY':5, 'MLT':3.127,
                          'UTH':13.65, 'VORTICITY':0.0020967, 'YEAR':2000,
                          'DATETIME':dt.datetime(2000,5,5,13,39,00), 'MONTH':5}
        self.assertTrue(os.path.isfile(self.test_file))

        # Remove in 2020
        if version_info.major == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp

    def tearDown(self):

        if os.path.isfile(self.temp_output):
            os.remove(self.temp_output)

        del self.test_file, self.test_output, self.test_ocb, self.temp_output
        del self.test_empty

    def test_load_vort_data(self):
        """ Test the routine to load the SuperDARN vorticity data
        """
        data = ocb_ivort.load_vorticity_ascii_data(self.test_file)

        # Test to see that the data keys are all in the header
        ktest = ['CENTRE_MLAT', 'DATETIME', 'DAY', 'MLT', 'MONTH', 'UTH',
                 'VORTICITY', 'YEAR']
        self.assertListEqual(ktest, sorted(list(data.keys())))

        # Test the length of the data file
        self.assertEqual(data['UTH'].shape[0], 5)

        # Test the values of the last data line
        for kk in self.test_vals.keys():
            self.assertEqual(data[kk][-1], self.test_vals[kk])

        del data, ktest

    def test_load_failure(self):
        """ Test the routine to load the SuperDARN vorticity data
        """
        data = ocb_ivort.load_vorticity_ascii_data("fake_file")

        self.assertIsNone(data)
        del data

    def test_wrong_load(self):
        """ Test the routine to load the SuperDARN vorticity data
        """
        bad_file = os.path.join(self.ocb_dir, "test", "test_data", "test_smag")
        data = ocb_ivort.load_vorticity_ascii_data(bad_file)

        self.assertIsNone(data)
        del bad_file, data

    def test_load_all_vort_data(self):
        """ Test the routine to load the SuperDARN vorticity data, loading
        all of the possible data values
        """
        data = ocb_ivort.load_vorticity_ascii_data(self.test_file,
                                                   save_all=True)

        # Test to see that the right number of keys were retrieved
        self.assertEqual(len(data.keys()), 32)

        # Test the values of the last data line, using only the data keys
        # needed for the OCB calculation
        for kk in self.test_vals.keys():
            self.assertEqual(data[kk][-1], self.test_vals[kk])

        del data

    def test_vort2ascii_ocb(self):
        """ Test the conversion of vorticity data from AACGM coordinates into
        OCB coordinates
        """

        ocb_ivort.vort2ascii_ocb(self.test_file, self.temp_output,
                                 ocbfile=self.test_ocb, instrument='image',
                                 hemisphere=1)

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows

            ldtype = ['|U50' if i < 2 else float for i in range(5)]
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

    def test_vort2ascii_ocb_write_failure(self):
        """ Test the conversion of vorticity data with a bad output filename
        """

        with self.assertRaises(IOError):
            ocb_ivort.vort2ascii_ocb(self.test_file, "/fake_dir/fake_out",
                                     ocbfile=self.test_ocb, instrument='image',
                                     hemisphere=1)

    def test_vort2ascii_ocb_load_failure(self):
        """ Test the conversion of vorticity data with a bad vorticity filename
        """

        with self.assertRaisesRegex(IOError, "vorticity file cannot be opened"):
            ocb_ivort.vort2ascii_ocb("fake_file", "fake_out",
                                     ocbfile=self.test_ocb)

    def test_vort2ascii_vort_load_failure(self):
        """Test conversion of vorticity data with an empty vorticity file"""

        with self.assertRaisesRegex(ValueError,
                                    "unable to load necessary data"):
            ocb_ivort.vort2ascii_ocb(self.test_empty, "fake_out",
                                     ocbfile=self.test_ocb)

            
    def test_vort2ascii_ocb_no_ocb(self):
        """ Test the conversion of vorticity data from AACGM coordinates into
        OCB coordinates
        """
        ocb_ivort.vort2ascii_ocb(self.test_file, "fake_out", ocbfile="fake_ocb")

        # Compare created file to stored test file
        self.assertFalse(ocbpy.instruments.general.test_file("fake_out"))

    def test_vort2ascii_ocb_output_failure(self):
        """ Test failure when bad filename is provided
        """
        # Error message changes based on operating system
        with self.assertRaises(IOError):
            ocb_ivort.vort2ascii_ocb(self.test_file, "/", ocbfile=self.test_ocb,
                                     instrument='image', hemisphere=1)

    def test_vort2ascii_ocb_output_failure_str(self):
        """ Test failure when a filename that is not a string is provided
        """
        with self.assertRaisesRegex(IOError, "output filename is not a string"):
            ocb_ivort.vort2ascii_ocb(self.test_file, 1, ocbfile=self.test_ocb)


if __name__ == '__main__':
    unittest.main()
