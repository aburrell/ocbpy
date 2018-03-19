#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocb_scaling class and functions
"""
import ocbpy.instruments.vort as ocb_ivort
import unittest
import numpy as np
import datetime as dt

class TestVortMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        from os import path
        import ocbpy
        
        ocb_dir = path.split(ocbpy.__file__)
        self.ocb_dir = ocb_dir[0] 
        self.test_ocb = path.join(self.ocb_dir, "tests", "test_data",
                                  "test_north_circle")
        self.test_file = path.join(self.ocb_dir, "tests", "test_data",
                                   "test_vort")
        self.test_output = path.join(self.ocb_dir, "tests", "test_data",
                                     "out_vort")
        self.temp_output = path.join(self.ocb_dir, "tests", "test_data",
                                     "temp_vort")
        self.test_vals = {'CENTRE_MLAT':67.27, 'DAY':5, 'MLT':3.127,
                          'UTH':13.65, 'VORTICITY':0.0020967, 'YEAR':2000,
                          'DATETIME':dt.datetime(2000,5,5,13,39,00), 'MONTH':5}
        self.assertTrue(path.isfile(self.test_file))

    def tearDown(self):
        import os

        if os.path.isfile(self.temp_output):
            os.remove(self.temp_output)

        del self.test_file, self.test_output, self.test_ocb, self.temp_output

    def test_load_vort_data(self):
        """ Test the routine to load the SuperDARN vorticity data
        """
        data = ocb_ivort.load_vorticity_ascii_data(self.test_file)

        # Test to see that the data keys are all in the header
        ktest = ['CENTRE_MLAT', 'DATETIME', 'DAY', 'MLT', 'MONTH', 'UTH',
                 'VORTICITY', 'YEAR']
        self.assertListEqual(ktest, sorted(data.keys()))

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
        from os.path import join
        bad_file = join(self.ocb_dir, "test", "test_data", "test_smag")
        data = ocb_ivort.load_vorticity_ascii_data("test_data/test_smag")

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
        import platform

        ocb_ivort.vort2ascii_ocb(self.test_file, self.temp_output,
                                 ocbfile=self.test_ocb)

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows
            from ocbpy.instruments import general

            kwout = {"datetime_cols":[0, 1], "datetime_fmt":"%Y-%m-%d %H:%M:%S"}
            test_out = general.load_ascii_data(self.test_output, 1, **kwout)
            temp_out = general.load_ascii_data(self.temp_output, 1, **kwout)

            # Test the headers
            self.assertListEqual(test_out[0], temp_out[0])

            # Test the data
            self.assertDictEqual(test_out[1], temp_out[1])

            del kwout, test_out, temp_out
        else:
            import filecmp
            # Compare created file to stored test file
            self.assertTrue(filecmp.cmp(self.test_output, self.temp_output,
                                        shallow=False))

    def test_vort2ascii_ocb_load_failure(self):
        """ Test the conversion of vorticity data from AACGM coordinates into
        OCB coordinates with a bad vorticity file
        """
        import filecmp
        from ocbpy.instruments.general import test_file

        try:
            ocb_ivort.vort2ascii_ocb("fake_file", "fake_out",
                                     ocbfile=self.test_ocb)

            # Compare created file to stored test file
            self.assertFalse(test_file("fake_out"))
        except:
            pass

    def test_vort2ascii_ocb_no_ocb(self):
        """ Test the conversion of vorticity data from AACGM coordinates into
        OCB coordinates
        """
        import filecmp
        from ocbpy.instruments.general import test_file

        ocb_ivort.vort2ascii_ocb(self.test_file, "fake_out", ocbfile="fake_ocb")

        # Compare created file to stored test file
        self.assertFalse(test_file("fake_out"))

    def test_vort2ascii_ocb_output_failure(self):
        """ Test the conversion of vorticity data from AACGM coordinates into
        OCB coordinates
        """
        import filecmp
        from ocbpy.instruments.general import test_file

        ocb_ivort.vort2ascii_ocb(self.test_file, "/", ocbfile=self.test_ocb)

        # Compare created file to stored test file
        self.assertFalse(test_file("/"))

if __name__ == '__main__':
    unittest.main()
