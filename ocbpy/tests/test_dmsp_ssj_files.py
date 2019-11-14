#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the boundaries.dmsp_ssj_files functions
"""
from __future__ import absolute_import, unicode_literals

import datetime as dt
from io import StringIO
import logging
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
        for ii in [[2, "fake_dir", "can't find the output directory"],
                   [3, [-47], "unknown satellite ID"]]:
            with self.subTest(ii=ii):
                temp = self.in_args[ii[0]]
                self.in_args[ii[0]] = ii[1]
                with self.assertRaisesRegex(ValueError, ii[2]):
                    self.fetch_files = boundaries.dmsp_ssj_files.fetch_ssj_files(*self.in_args)
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
        """ Initialize the test class"""
        self.ocb_dir = os.path.dirname(ocbpy.__file__)
        self.test_dir = os.path.join(self.ocb_dir, "tests")
        self.comp_files = [os.path.join(self.test_dir, "test_data",
    "dmsp-f16_ssj_precipitating-electrons-ions_20101231_v1.1.2_boundaries.csv")]
        self.cdf_files = [os.path.join(self.test_dir, "test_data",
            'dmsp-f16_ssj_precipitating-electrons-ions_20101231_v1.1.2.cdf')]
        self.out_cols = ['mlat', 'mlt']
        self.out = list()
        self.lout = ''
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        # Remove in 2020 when dropping support for 2.7
        if sys.version_info.major == 2:
            self.assertRegex = self.assertRegexpMatches
            self.assertRaisesRegex = self.assertRaisesRegexp

    def tearDown(self):
        if len(self.out) > 0:
            for ff in self.out:
                os.remove(ff)

        del self.ocb_dir, self.out, self.test_dir, self.cdf_files, self.out_cols
        del self.comp_files, self.log_capture, self.lout

    @unittest.skipIf(sys.version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_create_ssj_boundary_files_failure(self):
        """ Test create_ssj_boundary_files raising ValueError """

        # Cycle through the different value error raises
        for ii in [({"out_dir": "fake_dir"}, "unknown output directory"),
                   ({"make_plots": True, "plot_dir": "fake_dir"},
                    "unknown plot directory")]:
            with self.subTest(ii=list(ii)):
                with self.assertRaisesRegex(ValueError, ii[1]):
                    self.out = boundaries.dmsp_ssj_files.create_ssj_boundary_files(self.cdf_files, **ii[0])

    @unittest.skipIf(sys.version_info.major == 3,
                     'Already tested, remove in 2020')
    def test_create_ssj_boundary_files_plotdir_failure(self):
        """ Test create_ssj_boundary_files plot directory failure """

        with self.assertRaisesRegex(ValueError, "unknown plot directory"):
            self.out = boundaries.dmsp_ssj_files.create_ssj_boundary_files(
                self.cdf_files, make_plots=True, plot_dir='fake_dir')

    @unittest.skipIf(sys.version_info.major == 3,
                     'Already tested, remove in 2020')
    def test_create_ssj_boundary_files_outdir_failure(self):
        """ Test create_ssj_boundary_files output directory failure """

        with self.assertRaisesRegex(ValueError, "unknown output directory"):
            self.out = boundaries.dmsp_ssj_files.create_ssj_boundary_files(
                self.cdf_files, out_dir='fake_dir')

    @unittest.skipIf(sys.version_info.major == 3,
                     'Already tested, remove in 2020')
    def test_create_ssj_boundary_files_cdfname_failure(self):
        """ Test create_ssj_boundary_files bad cdf filename failure """

        # Try to read in a bad CDF filename
        self.out = boundaries.dmsp_ssj_files.create_ssj_boundary_files(
            self.comp_files)
        self.assertEqual(len(self.out), 0)

        # Test the logging output
        self.lout = self.log_capture.getvalue()
        self.assertTrue(self.lout.find("CDF") >= 0)

    @unittest.skipIf(sys.version_info.major == 3,
                     'Already tested, remove in 2020')
    def test_create_ssj_boundary_files_notafile_failure(self):
        """ Test create_ssj_boundary_files bad filename failure """

        # Try to read in a bad CDF filename
        self.out = boundaries.dmsp_ssj_files.create_ssj_boundary_files(
            [self.test_dir])
        self.assertEqual(len(self.out), 0)

        # Test the logging output
        self.lout = self.log_capture.getvalue()
        self.assertTrue(self.lout.find("bad input file") >= 0)

    def test_create_ssj_boundary_files_outcols_failure(self):
        """ Test create_ssj_boundary_files bad outcols failure """

        with self.assertRaises(TypeError):
            self.out = boundaries.dmsp_ssj_files.create_ssj_boundary_files(
                self.cdf_files, out_dir=self.test_dir, out_cols=['fake'])

    @unittest.skipIf(sys.version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_create_ssj_boundary_files_log_failure(self):
        """ Test create_ssj_boundary_files raising logging errors """

        # Cycle through the different value error raises
        for ii in [(self.comp_files, "CDF"),
                   ([self.test_dir], "bad input file")]:
            with self.subTest(ii=list(ii)):
                # Run with bad input file
                self.out = boundaries.dmsp_ssj_files.create_ssj_boundary_files(
                    ii[0])
                self.assertEqual(len(self.out), 0)

                # Test logging output
                self.lout = self.log_capture.getvalue()
                self.assertTrue(self.lout.find(ii[1]) >= 0)

    def test_create_ssj_boundary_files_default(self):
        """ Test the default implementation of create_ssj_boundary_files"""

        self.out = boundaries.dmsp_ssj_files.create_ssj_boundary_files(
            self.cdf_files, out_dir=self.test_dir)

        self.assertTrue(len(self.out), len(self.comp_files))

        # Compare the non-header data (since header has creation date)
        for i, fout in enumerate(self.out):
            test_out = np.genfromtxt(fout, skip_header=11, delimiter=',')
            temp_out = np.genfromtxt(self.comp_files[i], skip_header=11,
                                     delimiter=',')

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the data in each row
            for j, test_row in enumerate(test_out):
                self.assertListEqual(list(test_row), list(temp_out[j]))

        del test_out, temp_out, i, j, test_row, fout

    def test_create_ssj_boundary_files_outcols(self):
        """ Test the alternative output columns in create_ssj_boundary_files"""

        self.out = boundaries.dmsp_ssj_files.create_ssj_boundary_files(
            self.cdf_files, out_dir=self.test_dir, out_cols=self.out_cols)

        self.assertTrue(len(self.out), len(self.comp_files))

        for i, fout in enumerate(self.out):
            test_out = np.genfromtxt(fout, skip_header=11, delimiter=',')
            temp_out = np.genfromtxt(self.comp_files[i], skip_header=11,
                                     delimiter=',')

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the first eight data columns in each row
            for j, test_row in enumerate(test_out):
                self.assertListEqual(list(test_row[:8]), list(temp_out[j][:8]))

        del test_out, temp_out, i, j, fout, test_row

    def test_create_ssj_boundary_files_plots(self):
        """ Test the plot creation in create_ssj_boundary_files"""

        self.out = boundaries.dmsp_ssj_files.create_ssj_boundary_files(
            self.cdf_files, out_dir=self.test_dir, make_plots=True)

        self.assertTrue(len(self.out), len(self.comp_files))

        plot_files = list()
        for i, fout in enumerate(self.out):
            plot_root = fout.split("_boundaries")[0]
            test_out = np.genfromtxt(fout, skip_header=11, delimiter=',')
            temp_out = np.genfromtxt(self.comp_files[i], skip_header=11,
                                     delimiter=',')

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the first eight data columns in each row
            for j, test_row in enumerate(test_out):
                self.assertListEqual(list(test_row[:8]), list(temp_out[j][:8]))

                # Construct the plot filename
                plot_name = "{:s}_{:s}pass_uts{:05d}_uts{:05d}.png".format(
                    plot_root, "N" if test_row[2] == 1 else "S",
                    int(test_row[0]), int(test_row[1]))

                # Test the filename
                self.assertTrue(os.path.isfile(plot_name))
                plot_files.append(plot_name)

        self.out.extend(plot_files)

        del test_out, temp_out, i, j, fout, test_row, plot_name
        del plot_files, plot_root


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

