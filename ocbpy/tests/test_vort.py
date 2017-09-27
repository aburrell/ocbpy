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
        from os.path import isfile
        import ocbpy
        
        ocb_dir = ocbpy.__file__.split("/")
        self.test_ocb = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                            "tests/test_data/test_north_circle")
        self.test_file = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                            "tests/test_data/test_vort")
        self.test_output = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                              "tests/test_data/out_vort")
        self.temp_output = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                              "tests/test_data/temp_vort")
        self.test_vals = {'CENTRE_MLAT':67.27, 'DAY':5, 'MLT':3.127,
                          'UTH':13.65, 'VORTICITY':0.0020967, 'YEAR':2000,
                          'DATETIME':dt.datetime(2000,5,5,13,39,00), 'MONTH':5}
        self.assertTrue(isfile(self.test_file))

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

    def test_vort2ascii_ocb(self):
        """ Test the conversion of vorticity data from AACGM coordinates into
        OCB coordinates
        """
        import filecmp

        ocb_ivort.vort2ascii_ocb(self.test_file, self.temp_output,
                                 ocbfile=self.test_ocb)

        # Compare created file to stored test file
        self.assertTrue(filecmp.cmp(self.test_output, self.temp_output,
                                    shallow=False))
        

if __name__ == '__main__':
    unittest.main()
