#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
""" Tests the ocboundary class and functions
"""

import datetime as dt
import numpy as np
from sys import version_info
import unittest

from ocbpy import ocb_time


class TestOCBTimeMethods(unittest.TestCase):
    def setUp(self):
        """ Set up test runs """

        self.dtime = dt.datetime(2001, 1, 1)
        self.dtime2 = dt.datetime(1901, 1, 1)

        if version_info.major == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp

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
        self.assertEqual(
            ocb_time.convert_time(date="2001-01-01", tod="00-00-00",
                                  datetime_fmt="%Y-%m-%d %H-%M-%S"),
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
        """Test to see that the datetime construction works  with yyddd and sod
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
        input_dict = {"year": None, "soy": None, "yyddd": None, "sod": None,
                      "date": "2001-01-01", "tod": "000000",
                      "datetime_fmt": "%Y-%m-%d %H%M%S"}
        self.assertEqual(ocb_time.convert_time(**input_dict), self.dtime)

        # Test dictionary input implimentation
        input_dict = {"year": None, "soy": None, "yyddd": None, "sod": 0.0,
                      "date": "2001-01-01", "tod": None}
        self.assertEqual(ocb_time.convert_time(**input_dict), self.dtime)

        del input_dict

    def test_convert_time_failure_yyddd(self):
        """ Test convert_time failure with non-string input for yyddd
        """
        with self.assertRaisesRegex(ValueError, "YYDDD must be a string"):
            ocb_time.convert_time(yyddd=101001)

    def test_convert_time_failure_soy(self):
        """ Test convert_time failure with bad input for year-soy
        """
        with self.assertRaisesRegex(ValueError, "does not match format"):
            ocb_time.convert_time(soy=200)

    def test_convert_time_failure_bad_date_fmt(self):
        """ Test convert_time failure with bad input for incorrect date format
        """
        with self.assertRaisesRegex(ValueError, "does not match format"):
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
        with self.assertRaisesRegex(ValueError, "YYDDD must be a string"):
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


class TestOCBTimeUnits(unittest.TestCase):
    def setUp(self):
        """ Set up test runs """
        self.lon = np.linspace(0.0, 360.0, 37)
        self.lt = np.linspace(0.0, 24.0, 37)
        self.out = None

    def tearDown(self):
        """ Clean up after each test """
        del self.lon, self.lt, self.out

    def test_deg2hr_array(self):
        """ Test degree to hour conversion for an array"""
        self.out = ocb_time.deg2hr(self.lon)

        for i, val in enumerate(self.lt):
            self.assertAlmostEqual(self.out[i], val)

    def test_deg2hr_value(self):
        """ Test degree to hour conversion for a single value"""
        self.out = ocb_time.deg2hr(self.lon[0])

        self.assertAlmostEqual(self.out, self.lt[0])

    def test_hr2deg_array(self):
        """ Test hour to degree conversion for an array"""
        self.out = ocb_time.hr2deg(self.lt)

        for i, val in enumerate(self.lon):
            self.assertAlmostEqual(self.out[i], val)

    def test_hr2deg_value(self):
        """ Test hour to degree conversion for a single value"""
        self.out = ocb_time.deg2hr(self.lt[0])

        self.assertAlmostEqual(self.out, self.lon[0])

    def test_hr2rad_array(self):
        """ Test hour to radian conversion for an array"""
        self.out = ocb_time.hr2rad(self.lt)

        for i, val in enumerate(np.radians(self.lon)):
            self.assertAlmostEqual(self.out[i], val)

    def test_hr2rad_value(self):
        """ Test hour to radian conversion for a single value"""
        self.out = ocb_time.hr2rad(self.lt[0])

        self.assertAlmostEqual(self.out, np.radians(self.lon[0]))

    def test_rad2hr_array(self):
        """ Test radian to hour conversion for an array"""
        self.out = list(ocb_time.rad2hr(np.radians(self.lon)))

        for i, val in enumerate(self.out):
            self.assertAlmostEqual(val, self.lt[i])

    def test_rad2hr_value(self):
        """ Test radian to hour conversion for a single value"""
        self.out = ocb_time.rad2hr(np.radians(self.lon[0]))

        self.assertAlmostEqual(self.out, self.lt[0])


class TestOCBGeographicTime(unittest.TestCase):
    def setUp(self):
        """ Set up test runs """
        self.dtime = dt.datetime(2001, 1, 1, 1)
        self.lon = [390.0, 359.0, 90.0, -15.0, -30.0]
        self.lt = [27.0, 0.9333333333333336, 7.0, 0.0, -1.0]
        self.out = list()

    def tearDown(self):
        """ Clean up after each test """
        del self.lon, self.lt, self.dtime, self.out

    def test_glon2slt(self):
        """ Test longitude to slt conversion for a range of values"""
        # Prepare test output
        self.out = np.array(self.lt)
        self.out[self.out >= 24.0] -= 24.0
        self.out[self.out < 0.0] += 24.0

        # Test each value
        for i, lon in enumerate(self.lon):
            self.assertAlmostEqual(ocb_time.glon2slt(lon, self.dtime),
                                   self.out[i])

    def test_slt2glon(self):
        """ Test slt to longitude conversion for a range of values"""
        # Prepare test output
        self.out = np.array(self.lon)
        self.out[self.out > 180.0] -= 360.0
        self.out[self.out <= -180.0] += 360.0

        for i, lt in enumerate(self.lt):
            self.assertAlmostEqual(ocb_time.slt2glon(lt, self.dtime),
                                   self.out[i])

    def test_slt2glon_list(self):
        """ Test slt to longitude conversion for a list of values"""
        # Prepare test output
        self.lon = np.asarray(self.lon)
        self.lon[self.lon > 180.0] -= 360.0
        self.lon[self.lon <= -180.0] += 360.0

        self.out = ocb_time.slt2glon(self.lt, self.dtime)

        for i, ll in enumerate(self.out):
            self.assertAlmostEqual(ll, self.lon[i])

    def test_slt2glon_array(self):
        """ Test slt to longitude conversion for an array of values"""
        # Prepare test output
        self.lon = np.asarray(self.lon)
        self.lon[self.lon > 180.0] -= 360.0
        self.lon[self.lon <= -180.0] += 360.0

        self.out = ocb_time.slt2glon(np.asarray(self.lt), self.dtime)

        for i, ll in enumerate(self.out):
            self.assertAlmostEqual(ll, self.lon[i])

    def test_glon2slt_list(self):
        """ Test longtiude to lt conversion with list input"""
        # Prepare test output
        self.lt = np.asarray(self.lt)
        self.lt[self.lt >= 24.0] -= 24.0
        self.lt[self.lt < -0.0] += 24.0

        self.out = ocb_time.glon2slt(self.lon, self.dtime)

        for i, ll in enumerate(self.out):
            self.assertAlmostEqual(ll, self.lt[i])

    def test_glon2slt_array(self):
        """ Test longtiude to lt conversion with array input"""
        # Prepare test output
        self.lt = np.asarray(self.lt)
        self.lt[self.lt >= 24.0] -= 24.0
        self.lt[self.lt < -0.0] += 24.0

        self.out = ocb_time.glon2slt(np.asarray(self.lon), self.dtime)

        for i, ll in enumerate(self.out):
            self.assertAlmostEqual(ll, self.lt[i])


class TestTimeFormatMethods(unittest.TestCase):
    def setUp(self):
        """ Set up test runs """
        self.dtime = dt.datetime(2001, 1, 1)
        self.dt_formats = ['No Directives', '%y-%m-%d %H:%M:%S', '%a %b %Y %f',
                           '%A %B %z %Z', '%c', '%j %x', '%X']
        self.out_fmt = u''
        self.out_len = 0

    def tearDown(self):
        """ Clean up after each test """
        del self.dt_formats, self.dtime, self.out_fmt, self.out_len

    @unittest.skipIf(version_info.major < 3,
                     'Already tested, remove in 2020')
    def test_get_datetime_fmt_len(self):
        """ Test the datetime format length determination"""
        # Cycle through the different formatting options
        for val in self.dt_formats:
            with self.subTest(val=val):
                # Get the function format string length maximum
                self.out_len = ocb_time.get_datetime_fmt_len(val)

                # Get the formatted time string
                self.out_fmt = self.dtime.strftime(val)

                # Test to see that the returned length is greater than or
                # equal to the formatted time string
                self.assertGreaterEqual(self.out_len, len(self.out_fmt))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_get_datetime_fmt_len_nodate(self):
        """ Test the datetime format length determination for string w/o date
        """
        # Get the function format string length maximum
        self.out_len = ocb_time.get_datetime_fmt_len(self.dt_formats[0])

        # Get the formatted time string
        self.out_fmt = self.dtime.strftime(self.dt_formats[0])

        # Test to see that the returned length equalsthe formatted time string
        self.assertEqual(self.out_len, len(self.out_fmt))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_get_datetime_fmt_len_samelength(self):
        """ Test the datetime format length cal for a string of the same length
        """
        # Get the function format string length maximum
        self.out_len = ocb_time.get_datetime_fmt_len(self.dt_formats[1])

        # Get the formatted time string
        self.out_fmt = self.dtime.strftime(self.dt_formats[1])

        # Test to see that the returned length equalsthe formatted time string
        self.assertEqual(self.out_len, len(self.out_fmt))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_get_datetime_fmt_len_abYf(self):
        """ Test the datetime format length calc for the %a %b %Y %f directives
        """
        # Get the function format string length maximum
        self.out_len = ocb_time.get_datetime_fmt_len(self.dt_formats[2])

        # Get the formatted time string
        self.out_fmt = self.dtime.strftime(self.dt_formats[2])

        # Test to see that the returned length equalsthe formatted time string
        self.assertGreaterEqual(self.out_len, len(self.out_fmt))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_get_datetime_fmt_len_ABzZ(self):
        """ Test the datetime format length calc for the %A %B %z %Z directives
        """
        # Get the function format string length maximum
        self.out_len = ocb_time.get_datetime_fmt_len(self.dt_formats[3])

        # Get the formatted time string
        self.out_fmt = self.dtime.strftime(self.dt_formats[3])

        # Test to see that the returned length equalsthe formatted time string
        self.assertGreaterEqual(self.out_len, len(self.out_fmt))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_get_datetime_fmt_len_c(self):
        """ Test the datetime format length calc for the %c directive
        """
        # Get the function format string length maximum
        self.out_len = ocb_time.get_datetime_fmt_len(self.dt_formats[4])

        # Get the formatted time string
        self.out_fmt = self.dtime.strftime(self.dt_formats[4])

        # Test to see that the returned length equalsthe formatted time string
        self.assertGreaterEqual(self.out_len, len(self.out_fmt))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_get_datetime_fmt_len_x(self):
        """ Test the datetime format length calc for the %x directive
        """
        # Get the function format string length maximum
        self.out_len = ocb_time.get_datetime_fmt_len(self.dt_formats[5])

        # Get the formatted time string
        self.out_fmt = self.dtime.strftime(self.dt_formats[5])

        # Test to see that the returned length equalsthe formatted time string
        self.assertGreaterEqual(self.out_len, len(self.out_fmt))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_get_datetime_fmt_len_X(self):
        """ Test the datetime format length calc for the %X directive
        """
        # Get the function format string length maximum
        self.out_len = ocb_time.get_datetime_fmt_len(self.dt_formats[6])

        # Get the formatted time string
        self.out_fmt = self.dtime.strftime(self.dt_formats[6])

        # Test to see that the returned length equalsthe formatted time string
        self.assertGreaterEqual(self.out_len, len(self.out_fmt))


class TestFixRange(unittest.TestCase):
    def setUp(self):
        """ Set up test runs """

        self.vals = np.linspace(-190.0, 360.0, 37)
        self.out = None

        if version_info.major == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp

    def tearDown(self):
        """ Clean up after each test """

        del self.vals, self.out

    def test_fix_range_max_min_failure(self):
        """Test fix_range failure with bad max/min input"""
        with self.assertRaisesRegex(ValueError, "Minimum is not less than"):
            ocb_time.fix_range(self.vals, 10.0, -10.0)

    def test_fix_range_range_failure(self):
        """Test fix_range failure with bad value range input"""
        with self.assertRaisesRegex(ValueError, "Value range must be greater"):
            ocb_time.fix_range(self.vals, -10.0, 10.0, 0.0)

    @unittest.skipIf(version_info.major < 3,
                     'Already tested, remove in 2020')
    def test_fix_range(self):
        """ Test fix_range success """
        for tset in [([self.vals, 0.0, 360.0], {}),
                     ([list(self.vals), 0.0, 360.0], {}),
                     ([self.vals, -180.0, 360.0], {'val_range': 360.0}),
                     ([self.vals, 0.0, 24.0], {}),
                     ([self.vals[0], 0.0, 360.0], {})]:
            with self.subTest(tset=tset):
                self.out = ocb_time.fix_range(*tset[0], **tset[1])

                self.assertTrue(np.all(np.greater_equal(self.out, tset[0][1])))
                self.assertTrue(np.all(np.less(self.out, tset[0][2])))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_fix_range_default_array(self):
        """ Test fix_range success for array input and default value range
        """
        self.out = ocb_time.fix_range(self.vals, 0.0, 360.0)
        self.assertTrue(np.all(np.greater_equal(self.out, 0.0)))
        self.assertTrue(np.all(np.less(self.out, 360.0)))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_fix_range_default_list(self):
        """ Test fix_range success for list input and default value range
        """
        self.out = ocb_time.fix_range(list(self.vals), 0.0, 360.0)
        self.assertTrue(np.all(np.greater_equal(self.out, 0.0)))
        self.assertTrue(np.all(np.less(self.out, 360.0)))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_fix_range_set_range_array(self):
        """ Test fix_range success for array input and specified value range
        """
        self.out = ocb_time.fix_range(self.vals, -180.0, 360.0, 360.0)
        self.assertTrue(np.all(np.greater_equal(self.out, -180.0)))
        self.assertTrue(np.all(np.less(self.out, 360.0)))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_fix_range_large_range(self):
        """ Test fix_range success for input that deviates far from max/min
        """
        self.out = ocb_time.fix_range(self.vals, 0.0, 24.0)
        self.assertTrue(np.all(np.greater_equal(self.out, 0.0)))
        self.assertTrue(np.all(np.less(self.out, 24.0)))

    @unittest.skipIf(version_info.major > 2,
                     'Python 2.7 does not support subTest')
    def test_fix_range_default_float(self):
        """ Test fix_range success for a single value
        """
        self.out = ocb_time.fix_range(self.vals[0], 0.0, 360.0)
        self.assertGreaterEqual(self.out, 0.0)
        self.assertLess(self.out, 360.0)
