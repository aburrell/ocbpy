#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocb_scaling class and functions
"""
import ocbpy.instruments.tec as ocb_tec
import unittest
import numpy as np

class TestTECMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file
        """
        from os.path import isfile
        import ocbpy
        
        ocb_dir = ocbpy.__file__.split("/")
        self.test_ocb = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                            "tests/test_data/test_north_circle")
        self.test_file = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                            "tests/test_data/test_tec")
        self.bad_file = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                            "tests/test_data/test_tec_bad")
        self.test_output = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                              "tests/test_data/out_tec")
        self.temp_output = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                              "tests/test_data/temp_tec")
        self.assertTrue(isfile(self.test_file))

    def tearDown(self):
        import os

        if os.path.isfile(self.temp_output):
            os.remove(self.temp_output)

        del self.test_file, self.test_output, self.test_ocb, self.temp_output

    def test_load_madrigal_hdf5_tec(self):
        """ Test the routine to load the Madrigal TEC data
        """
        import datetime as dt

        data = ocb_tec.load_madrigal_hdf5_tec(self.test_file)

        # Test to see that the data keys are all in the header
        ktest = ['datetime', 'day', 'dtec', 'gdlat', 'glon', 'hour', 'min',
                 'month', 'recno', 'sec', 'tec', 'ut1_unix', 'ut2_unix', 'year']
        self.assertListEqual(ktest, sorted(list(data.keys())))

        # Test the length of the data file
        self.assertEqual(data['tec'].shape[0], 2)

        # Test the values of the last data line
        test_vals = {'dtec':1.9, 'ut1_unix':957534000, 'gdlat':89.0, 'hour':13,
                     'min':40, 'recno':2, 'tec':4.5, 'month':5,
                     'ut2_unix':957534300, 'sec':30, 'year':2000, 'glon':-40.0,
                     'datetime':dt.datetime(2000, 5, 5, 13, 40, 30), 'day':5}
        for kk in test_vals.keys():
            self.assertEqual(data[kk][-1], test_vals[kk])

    def test_load_failure(self):
        """ Test the routine to load the Madrigal TEC data
        """

        data = ocb_tec.load_madrigal_hdf5_tec("fake_file")
        self.assertIsNone(data)

    def test_load_bad_file(self):
        """ Test the routine to load a non-Madrigal TEC file
        """

        data = ocb_tec.load_madrigal_hdf5_tec(self.bad_file)
        self.assertIsNone(data)

    def test_tec2ascii_ocb(self):
        """ Test the conversion of Madrigal TEC data from geographic coordinates
        into OCB coordinates
        """
        import filecmp

        try:
            ocb_tec.madrigal_tec2ascii_ocb(self.test_file, self.temp_output,
                                           ocbfile=self.test_ocb)

            # Compare created file to stored test file
            self.assertTrue(filecmp.cmp(self.test_output, self.temp_output,
                                        shallow=False))
        except ImportError:
            pass

    def test_tec2ascii_ocb_bad_infile(self):
        """ Test the conversion of Madrigal TEC data from geographic coordinates
        into OCB coordinates
        """
        import filecmp

        try:
            ocb_tec.madrigal_tec2ascii_ocb(123, self.temp_output,
                                           ocbfile=self.test_ocb)
            self.assertFalse(filecmp.cmp(self.test_output, self.temp_output,
                                         shallow=False))
        except:
            pass

    def test_tec2ascii_ocb_bad_ocb(self):
        """ Test the conversion of Madrigal TEC data from geographic coordinates
        into OCB coordinates
        """
        from ocbpy.instruments.general import test_file

        try:
            ocb_tec.madrigal_tec2ascii_ocb(self.test_file, self.temp_output,
                                           ocbfile="fake_ocb")
            self.assertFalse(test_file(self.temp_output))
        except:
            pass
                
    def test_tec2ascii_ocb_bad_output(self):
        """ Test the conversion of Madrigal TEC data from geographic coordinates
        into OCB coordinates
        """
        from ocbpy.instruments.general import test_file

        try:
            ocb_tec.madrigal_tec2ascii_ocb(self.test_file, "/",
                                           ocbfile=self.test_ocb)
            self.assertFalse(test_file("/"))
        except:
            pass
                
if __name__ == '__main__':
    unittest.main()
