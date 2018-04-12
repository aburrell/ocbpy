#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocb_scaling class and functions
"""

import ocbpy.instruments.general as ocb_igen
import unittest
import numpy as np
import logbook

class TestGeneralMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        from os import path
        import ocbpy
        
        ocb_dir = path.split(ocbpy.__file__)[0]
        self.test_file = path.join(ocb_dir, "tests", "test_data",
                                   "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.temp_output = path.join(ocb_dir, "tests", "test_data",
                                     "temp_gen")
        self.log_handler = logbook.TestHandler()
        self.log_handler.push_thread()

    def tearDown(self):
        import os

        if os.path.isfile(self.temp_output):
            os.remove(self.temp_output)

        self.log_handler.pop_thread()
        del self.test_file, self.log_handler

    def test_file_test_true(self):
        """ Test the general file testing routine with a good file
        """
        self.assertTrue(ocb_igen.test_file(self.test_file))

    def test_file_test_not_file(self):
        """ Test the general file testing routine with a bad filename
        """
        self.assertFalse(ocb_igen.test_file("/"))

        self.assertEqual(len(self.log_handler.formatted_records), 1)
        self.assertTrue(self.log_handler.formatted_records[0].find( \
                                            'name provided is not a file') > 0)

    def test_file_test_empty_file(self):
        """ Test the general file testing routine with a bad filename
        """
        # Create an empty file
        open(self.temp_output, 'a').close()

        self.assertFalse(ocb_igen.test_file(self.temp_output))

        self.assertEqual(len(self.log_handler.formatted_records), 1)
        self.assertTrue(self.log_handler.formatted_records[0].find('empty file')
                        > 0)

    def test_load_ascii_data_badfile(self):
        """ Test the general loading routine for ASCII data with bad input
        """
        header, data = ocb_igen.load_ascii_data("/", 0)
        self.assertIsInstance(header, list)
        self.assertEqual(len(header), 0)
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data.keys()), 0)

        self.assertEqual(len(self.log_handler.formatted_records), 1)
        self.assertTrue(self.log_handler.formatted_records[0].find( \
                                            'name provided is not a file') > 0)

    def test_load_ascii_data_standard(self):
        """ Test the general routine to load ASCII data
        """

        hh = ["YEAR SOY NB PHICENT RCENT R A RERR"]
        header, data = ocb_igen.load_ascii_data(self.test_file, 0, header=hh)

        # Test to ensure the output header equals the input header
        self.assertListEqual(header, hh)

        # Test to see that the data keys are all in the header
        ktest = sorted(hh[0].split())
        self.assertListEqual(ktest, sorted(list(data.keys())))

        # Test the length of the data file
        self.assertEqual(data['A'].shape[0], 75)

        # Test the values of the last data line
        test_vals = {"YEAR":2000.0, "SOY":11187202.0, "NB":9.0, "A":1.302e+07,
                     "PHICENT":315.29, "RCENT":2.67, "R":18.38, "RERR":0.47}
        for kk in test_vals.keys():
            self.assertEqual(data[kk][-1], test_vals[kk])

        del hh, header, data, ktest, test_vals

    def test_load_ascii_data_int_cols(self):
        """ Test the general routine to load ASCII data assigning some
        columns as integers
        """

        hh = ["YEAR SOY NB PHICENT RCENT R A RERR"]
        int_cols = [0, 1, 2]
        int_keys = ["YEAR", "SOY", "NB"]
        header, data = ocb_igen.load_ascii_data(self.test_file, 0, header=hh,
                                                int_cols=int_cols)

        # Test to ensure the output header equals the input header
        self.assertListEqual(header, hh)

        # Test to see that the data keys are all in the header
        ktest = sorted(hh[0].split())
        self.assertListEqual(ktest, sorted(list(data.keys())))

        # Test the length of the data file
        self.assertEqual(data['A'].shape[0], 75)

        # Test the values of the last data line
        test_vals = {"YEAR":2000, "SOY":11187202, "NB":9, "A":1.302e+07,
                     "PHICENT":315.29, "RCENT":2.67, "R":18.38, "RERR":0.47}
        for kk in test_vals.keys():
            self.assertEqual(data[kk][-1], test_vals[kk])

            if kk in int_keys:
                isint = (isinstance(data[kk][-1], np.int64) or
                         isinstance(data[kk][-1], np.int32) or
                         isinstance(data[kk][-1], int))
                self.assertTrue(isint)

                del isint
            else:
                self.assertIsInstance(data[kk][-1], float)

        del hh, int_cols, int_keys, header, data, ktest, test_vals

    def test_load_ascii_data_str_cols(self):
        """ Test the general routine to load ASCII data assigning some
        columns as strings
        """

        hh = ["YEAR SOY NB PHICENT RCENT R A RERR"]
        str_cols = [0, 1]
        str_keys = ["YEAR", "SOY"]
        header, data = ocb_igen.load_ascii_data(self.test_file, 0, header=hh,
                                                str_cols=str_cols)

        # Test to ensure the output header equals the input header
        self.assertListEqual(header, hh)

        # Test to see that the data keys are all in the header
        ktest = sorted(hh[0].split())
        self.assertListEqual(ktest, sorted(list(data.keys())))

        # Test the length of the data file
        self.assertEqual(data['A'].shape[0], 75)

        # Test the values of the last data line
        test_vals = {"YEAR":"2000", "SOY":"11187202", "NB":9, "A":1.302e+07,
                     "PHICENT":315.29, "RCENT":2.67, "R":18.38, "RERR":0.47}
        for kk in test_vals.keys():
            self.assertEqual(data[kk][-1], test_vals[kk])

            if kk in str_keys:
                try:
                    self.assertIsInstance(data[kk][-1], str)
                except:
                    self.assertIsInstance(data[kk][-1], unicode)
            else:
                self.assertIsInstance(data[kk][-1], float)

        del hh, str_cols, str_keys, ktest, test_vals, header, data

    def test_load_ascii_data_w_datetime(self):
        """ Test the general routine to load ASCII data
        """
        import datetime as dt

        hh = ["YEAR SOY NB PHICENT RCENT R A RERR"]
        header, data = ocb_igen.load_ascii_data(self.test_file, 0,
                                                datetime_cols=[0,1],
                                                datetime_fmt="YEAR SOY",
                                                header=hh)

        # Test to ensure the output header equals the input header
        self.assertListEqual(header, hh)

        # Test to see that the data keys are all in the header
        ktest = hh[0].split()
        ktest.append("datetime")
        self.assertListEqual(sorted(ktest), sorted(list(data.keys())))

        # Test the length of the data file
        self.assertEqual(data['A'].shape[0], 75)

        # Test the values of the last data line
        test_vals = {"YEAR":2000, "SOY":11187202, "NB":9.0, "A":1.302e+07,
                     "PHICENT":315.29, "RCENT":2.67, "R":18.38, "RERR":0.47,
                     "datetime":dt.datetime(2000,5,9,11,33,22)}
        for kk in test_vals.keys():
            self.assertEqual(data[kk][-1], test_vals[kk])

        del hh, header, data, ktest, test_vals

if __name__ == '__main__':
    unittest.main()

