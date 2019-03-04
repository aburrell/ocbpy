#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocboundary class and functions
"""

import numpy as np
import unittest
from unittest import TestCase

from ocbpy import ocb_time

class TestOCBTimeMethods(TestCase):
    def setUp(self):
        """ Set up test runs """
        import datetime as dt

        self.dtime = dt.datetime(2001, 1, 1)
        self.dtime2 = dt.datetime(1901, 1, 1)

    def tearDown(self):
        """ Clean up after each test """

        del self.dtime, self.dtime2

    def test_year_soy_to_datetime(self):
        """ Test to see that the seconds of year conversion works
        """
        self.assertEqual(ocb_time.year_soy_to_datetime(2001, 0), self.dtime)

    def test_convert_time_date_tod(self):
        """ Test to see that the default datetime construction works
        """
        # Test the default date implimentation
        self.assertEqual(ocb_time.convert_time(date="2001-01-01",
                                               tod="00:00:00"),
                         self.dtime)

    def test_convert_time_date_tod_uncoverted(self):
        """ Test the datetime construction with unconverted data
        """
        # Test the default date implimentation
        self.assertEqual(ocb_time.convert_time(date="2001-01-01",
                                               tod="00:00:00.000001"),
                         self.dtime)

    def test_convert_time_date_tod_fmt(self):
        """ Test to see that the datetime construction works with custom format
        """
        # Test the custom date implimentation
        self.assertEqual(ocb_time.convert_time(date="2001-01-01", \
                            tod="00-00-00", datetime_fmt="%Y-%m-%d %H-%M-%S"),
                         self.dtime)

    def test_convert_time_year_soy(self):
        """ Test to see that the datetime construction works with year-soy
        """
        # Test the year-soy implimentation
        self.assertEqual(ocb_time.convert_time(year=2001, soy=0), self.dtime)

    def test_convert_time_yyddd_tod(self):
        """ Test to see that the datetime construction works with yyddd and tod
        """
        # Test the year-soy implimentation
        self.assertEqual(ocb_time.convert_time(yyddd="101001", tod="00:00:00"),
                         self.dtime)

    def test_convert_time_yyddd_tod_w_fmt(self):
        """ Test the datetime construction with yyddd, tod, and datetime_fmt
        """
        # Test the year-soy implimentation
        self.assertEqual(ocb_time.convert_time(yyddd="101001", tod="00 00 00",
                                               datetime_fmt="YYDDD %H %M %S"),
                         self.dtime)

    def test_convert_time_yyddd_tod_w_time_fmt(self):
        """ Test the datetime construction with yyddd, tod, and time fmt
        """
        # Test the year-soy implimentation
        self.assertEqual(ocb_time.convert_time(yyddd="101001", tod="00 00 00",
                                               datetime_fmt="%H %M %S"),
                         self.dtime)

    def test_convert_time_yyddd_sod(self):
        """ Test to see that the datetime construction works  with yyddd and sod
        """
        # Test the year-soy implimentation
        self.assertEqual(ocb_time.convert_time(yyddd="101001", sod=0),
                         self.dtime)

    def test_convert_time_yyddd_sod_ms(self):
        """ Test the datetime construction works with yyddd, sod, and ms
        """
        self.dtime = self.dtime.replace(microsecond=1)
        # Test the year-soy implimentation
        self.assertEqual(ocb_time.convert_time(yyddd="101001", sod=1.0e-6),
                         self.dtime)

    def test_convert_time_dict_input(self):
        """ Test to see that the datetime construction works with dict inputs
        """
        # Test dictionary input implimentation
        input_dict = {"year":None, "soy":None, "yyddd":None, "sod":None,
                      "date":"2001-01-01", "tod":"000000",
                      "datetime_fmt":"%Y-%m-%d %H%M%S"}
        self.assertEqual(ocb_time.convert_time(**input_dict), self.dtime)

        # Test dictionary input implimentation
        input_dict = {"year":None, "soy":None, "yyddd":None, "sod":0.0,
                      "date":"2001-01-01", "tod":None}
        self.assertEqual(ocb_time.convert_time(**input_dict), self.dtime)

        del input_dict

    def test_convert_time_failure_yyddd(self):
        """ Test convert_time failure with non-string input for yyddd
        """
        with self.assertRaisesRegexp(ValueError, "YYDDD must be a string"):
            ocb_time.convert_time(yyddd=101001)

    def test_convert_time_failure_soy(self):
        """ Test convert_time failure with bad input for year-soy
        """
        with self.assertRaisesRegexp(ValueError, "does not match format"):
            ocb_time.convert_time(soy=200)

    def test_convert_time_failure_bad_date_fmt(self):
        """ Test convert_time failure with bad input for incorrect date format
        """
        with self.assertRaisesRegexp(ValueError, "does not match format"):
            ocb_time.convert_time(date="2000", tod="00")

    def test_yyddd_to_date(self):
        """ Test to see that the datetime construction works
        """
        # Test the year-soy implimentation for 2001 and 1901
        self.assertEqual(ocb_time.yyddd_to_date(yyddd="101001"), self.dtime)
        self.assertEqual(ocb_time.yyddd_to_date(yyddd="01001"), self.dtime2)

    def test_yyddd_to_date_failure(self):
        """ Test yyddd_to_date failure with non-string input
        """
        with self.assertRaisesRegexp(ValueError, "YYDDD must be a string"):
            ocb_time.yyddd_to_date(yyddd=101001)

    def test_datetime2hr(self):
        """ Test datetime to hour of day conversion"""
        self.assertEqual(ocb_time.datetime2hr(self.dtime), 0.0)

    def test_datetime2hr_all_fracs(self):
        """ Test datetime to hour of day conversion for a time with h,m,s,ms"""
        self.dtime = self.dtime.replace(hour=1, minute=1, second=1,
                                        microsecond=1)
        self.assertAlmostEqual(ocb_time.datetime2hr(self.dtime), 1.01694444472)

    def test_datetime2hr_input_failure(self):
        """ Test datetime to hour of day conversion with bad input"""
        with self.assertRaises(AttributeError):
            ocb_time.datetime2hr(5.0)


class TestOCBTimeUnits(TestCase):
    def setUp(self):
        """ Set up test runs """

        self.lon = np.linspace(0.0, 360.0, 37)
        self.lt = np.linspace(0.0, 24.0, 37)

    def tearDown(self):
        """ Clean up after each test """

        del self.lon, self.lt

    def test_deg2hr_array(self):
        """ Test degree to hour conversion for an array"""

        out = ocb_time.deg2hr(self.lon)

        for i,val in enumerate(self.lt):
            self.assertAlmostEqual(out[i], val)
        del out, i, val

    def test_deg2hr_value(self):
        """ Test degree to hour conversion for a single value"""

        out = ocb_time.deg2hr(self.lon[0])

        self.assertAlmostEqual(out, self.lt[0])
        del out

    def test_hr2deg_array(self):
        """ Test hour to degree conversion for an array"""

        out = ocb_time.hr2deg(self.lt)

        for i,val in enumerate(self.lon):
            self.assertAlmostEqual(out[i], val)
        del out, i, val

    def test_hr2deg_value(self):
        """ Test hour to degree conversion for a single value"""

        out = ocb_time.deg2hr(self.lt[0])

        self.assertAlmostEqual(out, self.lon[0])
        del out

    def test_hr2rad_array(self):
        """ Test hour to radian conversion for an array"""

        out = ocb_time.hr2rad(self.lt)

        for i,val in enumerate(np.radians(self.lon)):
            self.assertAlmostEqual(out[i], val)
        del out, i, val

    def test_hr2rad_value(self):
        """ Test hour to radian conversion for a single value"""

        out = ocb_time.hr2rad(self.lt[0])

        self.assertAlmostEqual(out, np.radians(self.lon[0]))
        del out

    def test_rad2hr_array(self):
        """ Test radian to hour conversion for an array"""

        out = list(ocb_time.rad2hr(np.radians(self.lon)))

        for i,val in enumerate(out):
            self.assertAlmostEqual(val, self.lt[i])
        del out, i, val

    def test_rad2hr_value(self):
        """ Test radian to hour conversion for a single value"""

        out = ocb_time.rad2hr(np.radians(self.lon[0]))

        self.assertAlmostEqual(out, self.lt[0])
        del out


class TestOCBGeographicTime(TestCase):
    def setUp(self):
        """ Set up test runs """
        import datetime as dt

        self.dtime = dt.datetime(2001, 1, 1, 1)
        self.lon = [359.0, 90.0, -15.0]
        self.lt = [0.9333333333333336, 7.0, 0.0]

    def tearDown(self):
        """ Clean up after each test """

        del self.lon, self.lt, self.dtime

    def test_glon2slt(self):
        """ Test longitude to slt conversion for a range of values"""

        for i, lon in enumerate(self.lon):
            self.assertAlmostEqual(ocb_time.glon2slt(lon, self.dtime),
                                   self.lt[i])
        del i, lon
            
    def test_slt2glon(self):
        """ Test slt to longitude conversion for a range of values"""

        for i, lt in enumerate(self.lt):
            lon = self.lon[i] if self.lon[i] < 180.0 else self.lon[i] - 360.0
            self.assertAlmostEqual(ocb_time.slt2glon(lt, self.dtime), lon)
        del i, lt, lon

    def test_slt2glon_list_failure(self):
        """ Test local time to longtiude conversion with list input"""
        with self.assertRaises(TypeError):
            ocb_time.slt2glon(self.lt, self.dtime)

    def test_slt2glon_array_failure(self):
        """ Test local time to longtiude conversion with array input"""
        with self.assertRaises(ValueError):
            ocb_time.slt2glon(np.array(self.lt), self.dtime)

    def test_glon2slt_list_failure(self):
        """ Test longtiude to lt  conversion with list input"""
        with self.assertRaises(TypeError):
            ocb_time.glon2slt(self.lon, self.dtime)

    def test_glon2slt_array_failure(self):
        """ Test longtiude to lt conversion with array input"""
        with self.assertRaises(ValueError):
            ocb_time.glon2slt(np.array(self.lon), self.dtime)


if __name__ == '__main__':
    unittest.main()
