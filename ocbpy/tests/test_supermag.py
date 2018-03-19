#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocb_scaling class and functions
"""
import ocbpy.instruments.supermag as ocb_ismag
import unittest
import numpy as np

class TestSuperMAGMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        from os import path
        import ocbpy
        
        ocb_dir = path.split(ocbpy.__file__)
        self.test_ocb = path.join(ocb_dir[0], "tests", "test_data",
                                  "test_north_circle")
        self.test_file = path.join(ocb_dir[0], "tests", "test_data",
                                   "test_smag")
        self.test_output = path.join(ocb_dir[0], "tests", "test_data",
                                     "out_smag")
        self.temp_output = path.join(ocb_dir[0], "tests", "test_data",
                                     "temp_smag")
        self.assertTrue(path.isfile(self.test_file))

    def tearDown(self):
        import os

        if os.path.isfile(self.temp_output):
            os.remove(self.temp_output)

        del self.test_file, self.test_output, self.test_ocb, self.temp_output

    def test_load_supermag_ascii_data(self):
        """ Test the routine to load the SuperMAG data
        """
        import datetime as dt

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

    def test_load_failure(self):
        """ Test the routine to load the SuperMAG data
        """
        import datetime as dt

        header, data = ocb_ismag.load_supermag_ascii_data("fake_file")

        # Test to see that the data keys are all in the header
        self.assertListEqual(header, [])
        self.assertListEqual(list(data.keys()), [])

    def test_supermag2ascii_ocb(self):
        """ Test the conversion of SuperMAG data from AACGM coordinates into
        OCB coordinates
        """
        import platform

        ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output,
                                     ocbfile=self.test_ocb)

        if platform.system().lower() == "windows":
            # filecmp doesn't work on windows
            from ocbpy.instruments import general
            kwout = {"datetime_cols":[0, 1], "datetime_fmt":"%Y-%m-%d %H:%M:%S",
                     "str_cols":[3]}
            test_out = general.load_ascii_data(self.test_output, 1, **kwout)
            temp_out = general.load_ascii_data(self.temp_output, 1, **kwout)

            # Test the headers
            self.assertListEqual(test_out[0], temp_out[0])

            # Test the data
            self.assertDictEqual(test_out[1], temp_out[1])
        else:
            import filecmp
            # Compare created file to stored test file
            self.assertTrue(filecmp.cmp(self.test_output, self.temp_output,
                                        shallow=False))

    def test_supermag2ascii_ocb_bad_output(self):
        """ Test the conversion of SuperMAG data from AACGM coordinates into
        OCB coordinates
        """
        from ocbpy.instruments.general import test_file

        try:
            ocb_ismag.supermag2ascii_ocb(self.test_file, "/",
                                         ocbfile=self.test_ocb)
        except:
            pass

    def test_supermag2ascii_ocb_bad_ocb(self):
        """ Test the conversion of SuperMAG data from AACGM coordinates into
        OCB coordinates
        """
        from ocbpy.instruments.general import test_file

        ocb_ismag.supermag2ascii_ocb(self.test_file, "fake_out",
                                     ocbfile="fake_ocb")

        # Compare created file to stored test file
        self.assertFalse(test_file("fake_out"))
        

if __name__ == '__main__':
    unittest.main()
