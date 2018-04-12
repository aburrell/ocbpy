#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocboundary class and functions
"""

import ocbpy
import unittest
import datetime as dt

class TestOCBTimeMethods(unittest.TestCase):

    def test_year_soy_to_datetime(self):
        """ Test to see that the seconds of year conversion works
        """
        self.assertEqual(ocbpy.ocb_time.year_soy_to_datetime(2001, 0),
                         dt.datetime(2001,1,1))

    def test_convert_time_date_tod(self):
        """ Test to see that the datetime construction works
        """
        # Test the default date implimentation
        self.assertEqual(ocbpy.ocb_time.convert_time(date="2001-01-01",
                                                     tod="00:00:00"),
                         dt.datetime(2001,1,1))

    def test_convert_time_date_tod_fmt(self):
        """ Test to see that the datetime construction works
        """
        # Test the custom date implimentation
        self.assertEqual(ocbpy.ocb_time.convert_time(date="2001-01-01", \
                            tod="00-00-00", datetime_fmt="%Y-%m-%d %H-%M-%S"),
                         dt.datetime(2001,1,1))

    def test_convert_time_year_soy(self):
        """ Test to see that the datetime construction works
        """
        # Test the year-soy implimentation
        self.assertEqual(ocbpy.ocb_time.convert_time(year=2001, soy=0),
                         dt.datetime(2001,1,1))

    def test_convert_time_yyddd_tod(self):
        """ Test to see that the datetime construction works
        """
        # Test the year-soy implimentation
        self.assertEqual(ocbpy.ocb_time.convert_time(yyddd="101001",
                                                     tod="00:00:00"),
                         dt.datetime(2001,1,1))

    def test_convert_time_yyddd_sod(self):
        """ Test to see that the datetime construction works
        """
        # Test the year-soy implimentation
        self.assertEqual(ocbpy.ocb_time.convert_time(yyddd="101001", sod=0),
                         dt.datetime(2001,1,1))

    def test_convert_time_dict_input(self):
        """ Test to see that the datetime construction works
        """
        # Test dictionary input implimentation
        input_dict = {"year":None, "soy":None, "yyddd":None, "sod":None,
                      "date":"2001-01-01", "tod":"000000",
                      "datetime_fmt":"%Y-%m-%d %H%M%S"}
        self.assertEqual(ocbpy.ocb_time.convert_time(**input_dict),
                         dt.datetime(2001,1,1))

        # Test dictionary input implimentation
        input_dict = {"year":None, "soy":None, "yyddd":None, "sod":0.0,
                      "date":"2001-01-01", "tod":None}
        self.assertEqual(ocbpy.ocb_time.convert_time(**input_dict),
                         dt.datetime(2001,1,1))

        del input_dict
        
    def test_yyddd_to_date(self):
        """ Test to see that the datetime construction works
        """
        # Test the year-soy implimentation for 2001 and 1901
        self.assertEqual(ocbpy.ocb_time.yyddd_to_date(yyddd="101001"),
                         dt.datetime(2001,1,1))
        self.assertEqual(ocbpy.ocb_time.yyddd_to_date(yyddd="01001"),
                         dt.datetime(1901,1,1))

if __name__ == '__main__':
    unittest.main()
