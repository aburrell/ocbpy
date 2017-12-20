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

class TestVortMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        from os.path import isfile
        import ocbpy
        
        ocb_dir = ocbpy.__file__.split("/")
        self.test_ocb = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                            "tests/test_data/test_north_circle")
        self.test_file = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                            "tests/test_data/test_smag")
        self.test_output = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                              "tests/test_data/out_smag")
        self.temp_output = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                              "tests/test_data/temp_smag")
        self.assertTrue(isfile(self.test_file))

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
        self.assertListEqual(ktest, sorted(data.keys()))

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
        self.assertListEqual(data.keys(), [])

    def test_supermag2ascii_ocb(self):
        """ Test the conversion of SuperMAG data from AACGM coordinates into
        OCB coordinates
        """
        import filecmp

        ocb_ismag.supermag2ascii_ocb(self.test_file, self.temp_output,
                                     ocbfile=self.test_ocb)

        # Compare created file to stored test file
        self.assertTrue(filecmp.cmp(self.test_output, self.temp_output,
                                    shallow=False))
        

if __name__ == '__main__':
    unittest.main()
