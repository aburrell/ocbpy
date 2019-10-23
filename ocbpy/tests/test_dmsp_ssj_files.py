#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the boundaries.dmsp_ssj_files functions
"""
from __future__ import absolute_import, unicode_literals

import datetime as dt
import numpy as np
import os
import sys
import unittest

import ocbpy
from ocbpy import boundaries

no_ssj = False if hasattr(boundaries, 'dmsp_ssj_files') else True

@unittest.skipIf(no_ssj,
                 "ssj_auroral_boundary not installed, cannot test routines")
class TestSSJFetch(unittest.TestCase):

    def setUp(self):
        """ Initialize the test class
        """
        self.ocb_dir = os.path.dirname(ocbpy.__file__)
        self.sat_nums = [16, 17, 18]
        self.in_args = [dt.datetime(2010, 1, 1), dt.datetime(2010, 1, 2),
                        os.path.join(self.ocb_dir, "tests", "test_data"),
                        self.sat_nums]
        self.fetch_files = list()

        # Remove in 2020 when dropping support for 2.7
        if sys.version_info.major == 2:
            self.assertRegex = self.assertRegexpMatches

    def tearDown(self):
        if len(self.fetch_files) > 0:
            for ff in self.fetch_files:
                os.remove(ff)

        del self.ocb_dir, self.fetch_files, self.in_args, self.sat_nums

    def test_fetch_ssj_files_default(self):
        """ Test the default download behaviour for fetch_ssj_files"""

        self.fetch_files = boundaries.dmsp_ssj_files.fetch_ssj_files(
            self.in_args[0], self.in_args[1])

        self.assertEqual(len(self.sat_nums), len(self.fetch_files))
        self.in_args[2] = os.path.join(self.ocb_dir, "boundaries")
        for ff in self.fetch_files:
            self.assertRegex(os.path.dirname(ff), self.in_args[2])
            sat_num = int(ff.split('dmsp-f')[1][:2])
            self.assertIn(sat_num, self.sat_nums)

    def test_fetch_ssj_files_single(self):
        """ Test fetch_ssj_file downloading for a single satellite"""

        self.in_args[-1] = [self.sat_nums[0]]
        self.fetch_files = boundaries.dmsp_ssj_files.fetch_ssj_files(
            *self.in_args)
        self.assertEqual(len(self.fetch_files), 1)
        sat_num = int(self.fetch_files[0].split('dmsp-f')[1][:2])
        self.assertEqual(self.sat_nums[0], sat_num)

    def test_fetch_ssj_files_none(self):
        """ Test fetch_ssj_file downloading for no satellites"""

        self.in_args[-1] = []
        self.fetch_files = boundaries.dmsp_ssj_files.fetch_ssj_files(
            *self.in_args)
        self.assertEqual(len(self.fetch_files), 0)

    @unittest.skipIf(sys.version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_fetch_ssj_files_failure(self):
        """ Test fetch_ssj_files raising ValueError """

        # Cycle through the different value error raises
        for ii in [(2, "fake_dir", "can't find the output directory"),
                   (3, [-47], "unknown satellite ID")]:
            with self.subTest(ii=list(ii)):
                temp = self.in_args[ii[0]]
                self.in_args[ii[0]] = ii[1]
                with self.assertRaisesRegex(ValueError, ii[2]):
                    self.out = boundaries.dmsp_ssj_files.fetch_ssj_files(*self.in_args)
                self.in_args[ii[0]] = temp
        del temp

    @unittest.skipIf(sys.version_info.major == 3,
                     'Already tested, remove in 2020')
    def test_fetch_ssj_files_failure_fake_dir(self):
        """ Test fetch_ssj_files raising ValueError for fake directory"""

        self.in_args[2] = "fake_dir"
        with self.assertRaisesRegexp(ValueError,
                                    "can't find the output directory"):
            self.fetch_files = boundaries.dmsp_ssj_files.fetch_ssj_files(
                *self.in_args)

    @unittest.skipIf(sys.version_info.major == 3,
                     'Already tested, remove in 2020')
    def test_fetch_ssj_files_failure_unknown_sat(self):
        """ Test fetch_ssj_files raising ValueError for bad sat ID"""

        self.in_args[-1] = [-47]
        with self.assertRaisesRegexp(ValueError, "unknown satellite ID"):
            self.fetch_files = boundaries.dmsp_ssj_files.fetch_ssj_files(
                *self.in_args)

    def test_fetch_ssj_files_failure_bad_sat(self):
        """ Test fetch_ssj_files raising ValueError for bad sat ID"""

        self.in_args[-1] = -47
        with self.assertRaises(TypeError):
            self.fetch_files = boundaries.dmsp_ssj_files.fetch_ssj_files(
                *self.in_args)

@unittest.skipIf(no_ssj,
                 "ssj_auroral_boundary not installed, cannot test routines")
class TestSSJCreate(unittest.TestCase):

    def setUp(self):
        """ Initialize the test class
        """
        self.ocb_dir = os.path.dirname(ocbpy.__file__)
        self.test_dir = os.path.join(self.ocb_dir, "tests", "test_data")
        self.cdf_file = os.path.join(self.test_dir,
                'dmsp-f16_ssj_precipitating-electrons-ions_20101231_v1.1.2.cdf')
        self.out_cols = ['mlat', 'mlon']
        self.out_files = list()

        # Remove in 2020 when dropping support for 2.7
        if sys.version_info.major == 2:
            self.assertRegex = self.assertRegexpMatches

    def tearDown(self):
        if len(self.out_files) > 0:
            for ff in self.out_files:
                os.remove(ff)

        del self.ocb_dir, self.out_files, self.test_dir, self.cdf_file
        del.out_cols

@unittest.skipIf(not no_ssj,
                 "ssj_auroral_boundary installed, cannot test failure")
class TestSSJFailure(unittest.TestCase):
    def setUp(self):
        """ No initialization needed """
        pass

    def tearDown(self):
        """ No teardown needed"""
        pass

    def test_import_failure(self):
        """ Test ssj_auroral_boundary import failure"""

        with self.assertRaisesRegex(ImportError,
                                    'unable to load the DMSP SSJ module'):
            from ocbpy.boundaries import dmsp_ssj_files

if __name__ == '__main__':
    unittest.main()

