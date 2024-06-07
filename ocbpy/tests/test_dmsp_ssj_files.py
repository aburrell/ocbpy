#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DOI: 10.5281/zenodo.1179230
# Full license can be found in License.md
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# -----------------------------------------------------------------------------
"""Tests the boundaries.dmsp_ssj_files functions."""

import datetime as dt
from glob import glob
import numpy as np
import os
import unittest

from requests.exceptions import ProxyError, ConnectionError

from ocbpy import boundaries
import ocbpy.tests.class_common as cc

no_ssj = False if hasattr(boundaries, 'dmsp_ssj_files') else True

if not no_ssj:
    from ocbpy.boundaries import dmsp_ssj_files


@unittest.skipIf(no_ssj,
                 "ssj_auroral_boundary not installed, cannot test routines")
class TestSSJFetchDep(unittest.TestCase):
    """Unit tests for the DMSP-SSJ fetch functions."""

    def setUp(self):
        """Initialize the test class."""
        self.ocb_dir = os.path.split(os.path.split(cc.test_dir)[0])[0]
        self.sat_nums = [16, 17, 18]
        self.in_args = [dt.datetime(2010, 1, 1), dt.datetime(2010, 1, 2),
                        os.path.join(self.ocb_dir, "tests", "test_data"),
                        self.sat_nums]
        self.fetch_files = list()
        return

    def tearDown(self):
        """Clean up the test environment."""
        if len(self.fetch_files) > 0:
            for ff in self.fetch_files:
                os.remove(ff)

        del self.ocb_dir, self.fetch_files, self.in_args, self.sat_nums
        return

    def test_deprecated_fetch_ssj_files(self):
        """Test that the deprecated function raises a warning."""
        dep_str = "ssj_auroral_boundaries package is no longer supported"

        with self.assertWarnsRegex(DeprecationWarning, dep_str):
            try:
                dmsp_ssj_files.fetch_ssj_files(*self.in_args)
            except (ConnectionError, ProxyError, TimeoutError):
                pass

        return

    def test_fetch_ssj_files_local(self):
        """Test download behaviour for fetch_ssj_files with local files."""

        # Update the inputs
        self.in_args[0] = dt.datetime(2010, 12, 31)
        self.in_args[1] = dt.datetime(2011, 1, 1)
        self.sat_nums = [16]
        self.in_args[-1] = self.sat_nums

        # Run the fetch function
        self.fetch_files = dmsp_ssj_files.fetch_ssj_files(*self.in_args)

        # Evaluate the output
        self.assertEqual(len(self.sat_nums), len(self.fetch_files))
        for ffile in self.fetch_files:
            self.assertRegex(os.path.dirname(ffile), self.in_args[2])
            sat_num = int(ffile.split('dmsp-f')[1][:2])
            self.assertIn(sat_num, self.sat_nums)

        # Delete the fetch_files attr to avoid deleting the local test file
        self.fetch_files = list()
        return

    def test_fetch_ssj_files_default(self):
        """Test the default download behaviour for fetch_ssj_files."""

        try:
            self.fetch_files = dmsp_ssj_files.fetch_ssj_files(
                self.in_args[0], self.in_args[1])

            self.assertEqual(len(self.sat_nums), len(self.fetch_files))
            self.in_args[2] = os.path.join(self.ocb_dir, "boundaries")
            for ff in self.fetch_files:
                self.assertRegex(os.path.dirname(ff), self.in_args[2])
                sat_num = int(ff.split('dmsp-f')[1][:2])
                self.assertIn(sat_num, self.sat_nums)
        except (ConnectionError, ProxyError, TimeoutError):
            pass

        return

    def test_fetch_ssj_files_single(self):
        """Test fetch_ssj_file downloading for a single satellite."""

        self.in_args[-1] = [self.sat_nums[0]]

        try:
            self.fetch_files = dmsp_ssj_files.fetch_ssj_files(*self.in_args)
            self.assertEqual(len(self.fetch_files), 1)
            sat_num = int(self.fetch_files[0].split('dmsp-f')[1][:2])
            self.assertEqual(self.sat_nums[0], sat_num)
        except (ConnectionError, ProxyError, TimeoutError):
            pass
        return

    def test_fetch_ssj_files_none(self):
        """Test fetch_ssj_file downloading for no satellites."""

        self.in_args[-1] = []
        self.fetch_files = dmsp_ssj_files.fetch_ssj_files(*self.in_args)
        self.assertEqual(len(self.fetch_files), 0)
        return

    def test_fetch_ssj_files_failure(self):
        """Test fetch_ssj_files raising ValueError."""

        # Cycle through the different value error raises
        for ii in [[2, "fake_dir", "can't find the output directory"],
                   [3, [-47], "unknown satellite ID"]]:
            with self.subTest(ii=ii):
                temp = self.in_args[ii[0]]
                self.in_args[ii[0]] = ii[1]
                with self.assertRaisesRegex(ValueError, ii[2]):
                    self.fetch_files = dmsp_ssj_files.fetch_ssj_files(
                        *self.in_args)
                self.in_args[ii[0]] = temp
        del temp
        return

    def test_fetch_ssj_files_failure_bad_sat(self):
        """Test fetch_ssj_files raising ValueError for bad sat ID."""

        self.in_args[-1] = -47
        with self.assertRaises(TypeError):
            self.fetch_files = dmsp_ssj_files.fetch_ssj_files(
                *self.in_args)
        return


@unittest.skipIf(no_ssj,
                 "ssj_auroral_boundary not installed, cannot test routines")
class TestSSJCreate(cc.TestLogWarnings):
    """Unit tests for `create_ssj_boundary_files`."""

    def setUp(self):
        """Initialize the test class."""
        super().setUp()

        self.test_dir = os.path.split(cc.test_dir)[0]
        self.base_file = "".join(["dmsp-f16_ssj_precipitating-electrons-ions",
                                  "_20101231_v1.1.2"])
        self.comp_files = [os.path.join(cc.test_dir,
                                        "{:s}_boundaries.csv".format(
                                            self.base_file))]
        self.cdf_files = [os.path.join(cc.test_dir,
                                       '{:s}.cdf'.format(self.base_file))]
        self.out_cols = ['mlat', 'mlt']
        self.out = list()
        self.eval_ref = self.comp_files[0]
        self.eval_out = None
        return

    def tearDown(self):
        """Clean up the test environment."""
        if len(self.out) > 0:
            for ff in self.out:
                os.remove(ff)

        del self.out, self.cdf_files, self.out_cols, self.eval_out
        del self.comp_files, self.base_file, self.eval_ref, self.test_dir

        super().tearDown()
        return

    def eval_file_data(self, geo=False, mag=False, test_plot=False):
        """Evaluate the output of the created file.

        Parameters
        ----------
        geo : bool
            Flag to evaluate the geocentric locations (default=False)
        mag : bool
            Flag to evaluate the magnetic locations (default=False)
        test_plot : bool
            Flag to evaluate the creation of plots (default=False)

        """
        # Load the CSV data
        test_out = np.genfromtxt(self.eval_out, skip_header=11, delimiter=',')
        test_ref = np.genfromtxt(self.eval_ref, skip_header=11, delimiter=',')

        if test_plot:
            plot_root = self.eval_out.split("_boundaries")[0]

        # Test the number of rows and columns. Reference contains geocentric
        # and magnetic coordinates, so may have more data columns
        self.assertEqual(test_out.shape[0], test_ref.shape[0])
        self.assertLessEqual(test_out.shape[1], test_ref.shape[1])

        # Test the data in each row
        for j, test_row in enumerate(test_out):
            cmax = 8
            self.assertListEqual(list(test_row[:cmax]),
                                 list(test_ref[j][:cmax]))

            if geo:
                cmin = cmax
                cmax += 8
                self.assertListEqual(list(test_row[cmin:cmax]),
                                     list(test_ref[j][cmin:cmax]),
                                     msg="Geocentric evaluation failed")
            if mag:
                cmin = cmax
                cmax += 8
                self.assertListEqual(list(test_row[cmin:cmax]),
                                     list(test_ref[j][-8:]),
                                     msg="Magnetic evaluation failed")

            if test_plot:
                # Construct the plot filename
                plot_name = "{:s}_{:s}pass_uts{:05d}_uts{:05d}.png".format(
                    plot_root, "N" if test_row[2] == 1 else "S",
                    int(test_row[0]), int(test_row[1]))

                # Test the filename and ready it for removal on clean-up
                self.assertTrue(os.path.isfile(plot_name))
                self.out.append(plot_name)
        return

    def test_deprecated_function(self):
        """Test that the deprecated function raises a warning."""
        dep_str = "ssj_auroral_boundaries package is no longer supported"

        with self.assertWarnsRegex(DeprecationWarning, dep_str):
            self.out = dmsp_ssj_files.create_ssj_boundary_files(
                self.cdf_files, out_dir=self.test_dir)
        return

    def test_create_ssj_boundary_files_failure(self):
        """Test create_ssj_boundary_files raising ValueError."""

        # Cycle through the different value error raises
        for ii in [({"out_dir": "fake_dir"}, "unknown output directory"),
                   ({"make_plots": True, "plot_dir": "fake_dir"},
                    "unknown plot directory")]:
            with self.subTest(ii=list(ii)):
                with self.assertRaisesRegex(ValueError, ii[1]):
                    dmsp_ssj_files.create_ssj_boundary_files(
                        self.cdf_files, **ii[0])
        return

    def test_create_ssj_boundary_files_outcols_failure(self):
        """Test create_ssj_boundary_files bad outcols failure."""

        with self.assertRaises(TypeError):
            dmsp_ssj_files.create_ssj_boundary_files(
                self.cdf_files, out_dir=self.test_dir, out_cols=['fake'])
        return

    def test_create_ssj_boundary_files_log_failure(self):
        """Test create_ssj_boundary_files raising logging warning."""

        # Run with bad input files
        for in_files, self.lwarn in ([self.comp_files, "CDF"],
                                     [[cc.test_dir], "bad input file"]):
            with self.subTest(in_files=in_files):
                self.out = dmsp_ssj_files.create_ssj_boundary_files(in_files)
                self.assertEqual(len(self.out), 0,
                                 msg="unexpected output: {:}".format(self.out))

                # Test logging output
                self.eval_logging_message()
        return

    def test_create_ssj_boundary_files_default(self):
        """Test the default implementation of create_ssj_boundary_files."""

        self.out = dmsp_ssj_files.create_ssj_boundary_files(
            self.cdf_files, out_dir=self.test_dir)

        self.assertEqual(len(self.out), len(self.comp_files))

        # Compare the non-header data (since header has creation date)
        for i, fout in enumerate(self.out):
            self.eval_out = fout
            self.eval_ref = self.comp_files[i]
            self.eval_file_data(geo=True)
        return

    def test_create_ssj_boundary_files_str_input(self):
        """Test the single file input with create_ssj_boundary_files."""

        i = 0
        self.out = dmsp_ssj_files.create_ssj_boundary_files(
            self.cdf_files[i], out_dir=self.test_dir)

        self.assertEqual(len(self.out), 1)

        # Compare the non-header data (since header has creation date)
        self.eval_out = self.out[i]
        self.eval_ref = self.comp_files[i]
        self.eval_file_data(geo=True)
        return

    def test_create_ssj_boundary_files_outcols(self):
        """Test the alternative output columns in create_ssj_boundary_files."""

        self.out = dmsp_ssj_files.create_ssj_boundary_files(
            self.cdf_files, out_dir=self.test_dir, out_cols=self.out_cols)

        self.assertEqual(len(self.out), len(self.comp_files))

        for i, fout in enumerate(self.out):
            self.eval_out = fout
            self.eval_ref = self.comp_files[i]
            self.eval_file_data(mag=True)
        return

    def test_create_ssj_boundary_files_plots(self):
        """Test the plot creation in create_ssj_boundary_files."""

        try:
            self.out = dmsp_ssj_files.create_ssj_boundary_files(
                self.cdf_files, out_dir=self.test_dir, make_plots=True)

            self.assertEqual(len(self.out), len(self.comp_files))

            for i, fout in enumerate(self.out):
                self.eval_out = fout
                self.eval_ref = self.comp_files[i]
                self.eval_file_data(geo=True, test_plot=True)

        except IndexError as ierr:
            print("Allowed failure: {:}".format(ierr))
        return


@unittest.skipIf(no_ssj,
                 "ssj_auroral_boundary not installed, cannot test routines")
class TestSSJFetch(unittest.TestCase):
    """Unit tests for the DMSP-SSJ fetch functions."""

    def setUp(self):
        """Initialize the test class."""
        self.ocb_dir = os.path.split(os.path.split(cc.test_dir)[0])[0]
        self.sat_nums = [16, 17, 18]
        self.in_args = [dt.datetime(2010, 1, 1), dt.datetime(2010, 1, 2),
                        cc.test_dir, self.sat_nums]
        self.fetch_files = list()
        return

    def tearDown(self):
        """Clean up the test environment."""
        if len(self.fetch_files) > 0:
            for ff in self.fetch_files:
                os.remove(ff)

        del self.ocb_dir, self.fetch_files, self.in_args, self.sat_nums
        return

    def test_fetch_ssj_files_local(self):
        """Test download behaviour for fetch_ssj_files with local files."""

        # Update the inputs
        self.in_args[0] = dt.datetime(2010, 12, 31)
        self.in_args[1] = dt.datetime(2011, 1, 1)
        self.sat_nums = [16]
        self.in_args[-1] = self.sat_nums

        # Run the fetch function
        self.fetch_files = dmsp_ssj_files.fetch_ssj_boundary_files(
            *self.in_args)

        # Evaluate the output
        self.assertEqual(len(self.sat_nums), len(self.fetch_files))
        for ffile in self.fetch_files:
            self.assertRegex(os.path.dirname(ffile), self.in_args[2])
            sat_num = int(ffile.split('dmsp-f')[1][:2])
            self.assertIn(sat_num, self.sat_nums)

        # Delete the fetch_files attr to avoid deleting the local test file
        self.fetch_files = list()
        return

    def test_fetch_ssj_files_basic(self):
        """Test the standard download behaviour for fetch_ssj_boundary_files."""

        self.fetch_files = dmsp_ssj_files.fetch_ssj_boundary_files(
            self.in_args[0], self.in_args[1])

        self.assertEqual(len(self.sat_nums), len(self.fetch_files))
        self.in_args[2] = os.path.join(self.ocb_dir, "boundaries")
        for ff in self.fetch_files:
            self.assertRegex(os.path.dirname(ff), self.in_args[2])
            sat_num = int(ff.split('dmsp-f')[1][:2])
            self.assertIn(sat_num, self.sat_nums)
        return

    def test_fetch_ssj_files_all(self):
        """Test the download behaviour for getting all remote boundary files."""

        self.fetch_files = dmsp_ssj_files.fetch_ssj_boundary_files()

        self.assertEqual(4594, len(self.fetch_files))
        self.in_args[2] = os.path.join(self.ocb_dir, "boundaries")
        return

    def test_fetch_ssj_files_single(self):
        """Test fetch_ssj_boundary_file downloading for a single satellite."""

        self.in_args[-1] = [self.sat_nums[0]]
        self.fetch_files = dmsp_ssj_files.fetch_ssj_boundary_files(
            *self.in_args)
        self.assertEqual(len(self.fetch_files), 1)
        sat_num = int(self.fetch_files[0].split('dmsp-f')[1][:2])
        self.assertEqual(self.sat_nums[0], sat_num)
        return

    def test_fetch_ssj_files_none(self):
        """Test fetch_ssj_file downloading for no satellites."""

        self.in_args[-1] = []
        self.fetch_files = dmsp_ssj_files.fetch_ssj_boundary_files(
            *self.in_args)
        self.assertEqual(len(self.fetch_files), 0)
        return

    def test_fetch_ssj_files_value_failure(self):
        """Test fetch_ssj_files raising ValueError."""
        self.in_args.append('10.5281/zenodo.3373811')

        # Cycle through the different value error raises
        for ii in [[2, "fake_dir", "can't find the output directory"],
                   [3, [-47], "unknown satellite ID"], [4, 'baddoi', "DOI"]]:
            with self.subTest(ii=ii):
                temp = self.in_args[ii[0]]
                self.in_args[ii[0]] = ii[1]
                with self.assertRaisesRegex(ValueError, ii[2]):
                    self.fetch_files = dmsp_ssj_files.fetch_ssj_boundary_files(
                        *self.in_args)
                self.in_args[ii[0]] = temp
        del temp
        return

    def test_fetch_ssj_files_failure_bad_sat(self):
        """Test fetch_ssj_files raising TypeError for bad sat ID."""

        self.in_args[-1] = -47
        with self.assertRaises(TypeError):
            self.fetch_files = dmsp_ssj_files.fetch_ssj_boundary_files(
                *self.in_args)
        return


@unittest.skipIf(no_ssj,
                 "ssj_auroral_boundary not installed, cannot test routines")
class TestSSJEvalFilename(unittest.TestCase):
    """Unit tests for `evaluate_dmsp_boundary_file`."""

    def setUp(self):
        """Initialize the test class."""
        self.csv_file = "".join(["dmsp-f16_ssj_precipitating-electrons-ions_",
                                 "20101231_v1.1.2_boundaries.csv"])
        self.ftime = dt.datetime(2010, 12, 31)
        self.sat_id = 16
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.csv_file, self.ftime, self.sat_id

    def test_eval_dmsp_good_file(self):
        """Test the filename evaluation for good conditions."""

        stimes = [self.ftime, None]
        etimes = [self.ftime + dt.timedelta(days=1), None]
        sat_nums = [self.sat_id]

        for stime in stimes:
            for etime in etimes:
                with self.subTest(stime=stime, etime=etime):
                    self.assertTrue(
                        dmsp_ssj_files.evaluate_dmsp_boundary_file(
                            self.csv_file, stime, etime, sat_nums))
        return

    def test_eval_dmsp_bad_sat_id_file(self):
        """Test the filename evaluation for bad sat ID."""

        self.assertFalse(dmsp_ssj_files.evaluate_dmsp_boundary_file(
            self.csv_file, self.ftime, self.ftime, [self.sat_id + 1]))
        return

    def test_eval_dmsp_bad_time_file(self):
        """Test the filename evaluation for bad time range."""

        self.assertFalse(dmsp_ssj_files.evaluate_dmsp_boundary_file(
            self.csv_file, None, self.ftime - dt.timedelta(days=2),
            [self.sat_id]))
        return


@unittest.skipIf(no_ssj,
                 "ssj_auroral_boundary not installed, cannot test routines")
class TestSSJFormat(cc.TestLogWarnings):
    """Unit tests for `format_ssj_boundary_files`."""

    def setUp(self):
        """Initialize the test class."""
        super().setUp()

        self.test_dir = cc.test_dir
        self.comp_files = {"dmsp-ssj_north_20101231_20101231_v1.1.2.eab":
                           os.path.join(cc.test_dir,
                                        "dmsp-ssj_north_out.eab"),
                           "dmsp-ssj_south_20101231_20101231_v1.1.2.eab":
                           os.path.join(cc.test_dir,
                                        "dmsp-ssj_south_out.eab"),
                           "dmsp-ssj_north_20101231_20101231_v1.1.2.ocb":
                           os.path.join(cc.test_dir,
                                        "dmsp-ssj_north_out.ocb"),
                           "dmsp-ssj_south_20101231_20101231_v1.1.2.ocb":
                           os.path.join(cc.test_dir,
                                        "dmsp-ssj_south_out.ocb")}
        self.csv_files = [os.path.join(cc.test_dir,
                                       "".join(["dmsp-f16_ssj_precipitating",
                                                "-electrons-ions_20101231_",
                                                "v1.1.2_boundaries.csv"]))]
        self.out = list()
        self.ldtype = [int, '|U50', '|U50', float, float, float,
                       float, float, float, float, float]
        return

    def tearDown(self):
        """Clean up the test environment."""
        if len(self.out) > 0:
            for fout in self.out:
                os.remove(fout)

        del self.out, self.test_dir, self.csv_files, self.comp_files
        del self.ldtype
        super().tearDown()
        return

    def eval_formatted_output(self, limit_line_comp=False):
        """Evaluate the formatted output.

        Parameters
        ----------
        limit_line_comp : bool
            Limit the line comparison to the first three elements if True,
            compare all columns if False (default=False)

        """

        # Ensure the expected number of files were created
        self.assertEqual(len(self.out), len(self.comp_files.values()),
                         msg="unexpected number of boundary files")

        # Compare the non-header data (since header has creation date)
        for fout in self.out:
            # Get the comparison filename
            fname = os.path.split(fout)[-1]
            self.assertIn(fname, self.comp_files.keys())

            # Load the data
            test_out = np.genfromtxt(fout, skip_header=1, dtype=self.ldtype)
            temp_out = np.genfromtxt(self.comp_files[fname], skip_header=1,
                                     dtype=self.ldtype)

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, temp_out.shape)

            # Test the data in each row
            for j, test_row in enumerate(test_out):
                if limit_line_comp:
                    for i in range(3):
                        self.assertEqual(test_row[i], temp_out[j][i])
                else:
                    self.assertListEqual(list(test_row), list(temp_out[j]))
        return

    def test_format_ssj_boundary_files_default(self):
        """Test the default implementation of format_ssj_boundary_files."""

        self.out = dmsp_ssj_files.format_ssj_boundary_files(
            self.csv_files)
        self.eval_formatted_output()
        return

    def test_format_ssj_boundary_files_single_file_input(self):
        """Test `format_ssj_boundary_files` with a single file input."""

        self.out = dmsp_ssj_files.format_ssj_boundary_files(self.csv_files[0])
        self.eval_formatted_output()
        return

    def test_format_ssj_boundary_files_mixed_input(self):
        """Test `format_ssj_boundary_files` with mixed good/bad file input."""

        # Create an empty, badly formatted input file
        self.csv_files.append(os.path.join(cc.test_dir, "".join([
            "dmsp-f47_ssj_precipitating-electrons-ion_20101231_v1.1.2_",
            "boundaries.csv"])))
        with open(self.csv_files[-1], "w") as fp:
            fp.write("#sc date time\n47 2010-12-31 00:00:00")

        # Initialize the logger warnings
        self.lwarn = "unable to format"

        # Format the files
        self.out = dmsp_ssj_files.format_ssj_boundary_files(
            self.csv_files)

        # Evaluate the good output
        self.eval_formatted_output()

        # Evaluate the logging messages
        self.eval_logging_message()

        # Prepare the empty file for clean up
        self.out.append(self.csv_files[-1])
        return

    def test_format_ssj_boundary_files_diff_alt(self):
        """Test format_ssj_boundary_files with a different altitude."""

        self.out = dmsp_ssj_files.format_ssj_boundary_files(
            self.csv_files, ref_alt=300.0)
        self.eval_formatted_output(limit_line_comp=True)

        return

    def test_format_ssj_boundary_files_diff_method(self):
        """Test format_ssj_boundary_files with a different method."""

        self.out = dmsp_ssj_files.format_ssj_boundary_files(
            self.csv_files, method='GEOCENTRIC|TRACE')
        self.eval_formatted_output(limit_line_comp=True)

        return

    def test_format_ssj_boundary_files_notafile_failure(self):
        """Test create_ssj_boundary_files bad filename failure."""

        with self.assertRaisesRegex(ValueError, "empty list of input CSV"):
            # Try to read in a bad CDF filename
            self.out = dmsp_ssj_files.format_ssj_boundary_files([cc.test_dir])

        # Test the logging output
        self.lwarn = "bad input file"
        self.eval_logging_message()
        return

    def test_format_ssj_boundary_files_failure(self):
        """Test format_ssj_boundary_files no input failure."""

        with self.assertRaisesRegex(ValueError, "empty list of input CSV"):
            self.out = dmsp_ssj_files.format_ssj_boundary_files([])
        return


@unittest.skipIf(no_ssj,
                 "ssj_auroral_boundary not installed, cannot test routines")
class TestSSJFetchFormat(unittest.TestCase):
    """Unit tests for the combined fetch/format function."""

    def setUp(self):
        """Initialize the test class."""
        self.test_dir = os.path.split(cc.test_dir)[0]
        self.comp_files = {"dmsp-ssj_north_20101231_20101231_v1.1.2.eab":
                           os.path.join(cc.test_dir, "dmsp-ssj_north_out.eab"),
                           "dmsp-ssj_south_20101231_20101231_v1.1.2.eab":
                           os.path.join(cc.test_dir, "dmsp-ssj_south_out.eab"),
                           "dmsp-ssj_north_20101231_20101231_v1.1.2.ocb":
                           os.path.join(cc.test_dir, "dmsp-ssj_north_out.ocb"),
                           "dmsp-ssj_south_20101231_20101231_v1.1.2.ocb":
                           os.path.join(cc.test_dir, "dmsp-ssj_south_out.ocb")}
        self.in_args = [dt.datetime(2010, 12, 31), dt.datetime(2011, 1, 1),
                        self.test_dir]
        self.out = list()
        self.ldtype = [int, '|U50', '|U50', float, float, float,
                       float, float, float, float, float]
        return

    def tearDown(self):
        """Clean up the test environment."""
        if len(self.out) > 0:
            for ff in self.out:
                os.remove(ff)

        del self.out, self.in_args, self.comp_files, self.ldtype, self.test_dir
        return

    def test_use_deprecated(self):
        """Test that using the deprecated functions raises a warning."""
        dep_str = "kwarg to access deprecated routines will be removed"

        with self.assertWarnsRegex(DeprecationWarning, dep_str):
            try:
                dmsp_ssj_files.fetch_format_ssj_boundary_files(
                    *self.in_args, use_dep=True)
            except (ConnectionError, ProxyError, TimeoutError):
                pass
        return

    def test_fetch_format_ssj_boundary_files_default(self):
        """Test success with `fetch_format_ssj_boundary_files` defaults."""
        self.out = dmsp_ssj_files.fetch_format_ssj_boundary_files(
            *self.in_args)

        comp_keys = list(self.comp_files.keys())
        self.assertEqual(len(self.out), len(comp_keys))

        # Compare the non-header data (since header has creation date)
        for fout in self.out:
            # Get the comparison filename
            fname = os.path.split(fout)[-1]
            self.assertIn(fname, comp_keys)

            # Load the data
            test_out = np.genfromtxt(fout, skip_header=1, dtype=self.ldtype)
            temp_out = np.genfromtxt(self.comp_files[fname], skip_header=1,
                                     dtype=self.ldtype)

            # The comparison data will only have one spacecraft, while
            # the routine includes data from three spacecraft
            self.assertGreater(test_out.shape, temp_out.shape)

            # Test the data in each row
            for comp_row in temp_out:
                self.assertIn(comp_row, test_out)

        return

    def test_fetch_format_ssj_boundary_files_no_rm_temp(self):
        """Test `fetch_format_ssj_boundary_files` with old temp files."""

        try:
            self.out = dmsp_ssj_files.fetch_format_ssj_boundary_files(
                *self.in_args, rm_temp=False, use_dep=True)

            comp_keys = list(self.comp_files.keys())
            self.assertEqual(len(self.out), len(comp_keys))

            # Compare the non-header data (since header has creation date)
            nsat = dict()
            for fout in self.out:
                # Get the comparison filename
                fname = os.path.split(fout)[-1]
                self.assertIn(fname, comp_keys)

                # Load the data
                test_out = np.genfromtxt(fout, skip_header=1, dtype=self.ldtype)
                temp_out = np.genfromtxt(self.comp_files[fname], skip_header=1,
                                         dtype=self.ldtype)

                # The comparison data will only have one spacecraft, while
                # the routine includes data from three spacecraft
                self.assertGreater(test_out.shape, temp_out.shape)

                # Test the data in each row
                for comp_row in temp_out:
                    self.assertIn(comp_row, test_out)

                # Get the number of spacecraft
                for comp_row in test_out:
                    nsat[comp_row[0]] = 1

            fname = sum([comp_row for comp_row in nsat.values()])

            # See how many files of each temporary type were downloaded
            temp_out = glob(os.path.join(self.test_dir, "dmsp-f*.cdf"))
            test_out = glob(os.path.join(self.test_dir, "dmsp-f*.csv"))

            self.assertEqual(len(temp_out), len(test_out),
                             msg="number of CDF and CSV files unequal")

            # Test that the files downloaded match the requested satellites
            self.assertEqual(fname, len(temp_out))
            for fout in nsat.keys():
                fname = "f{:02d}".format(fout)
                self.assertEqual(sum([1 if fname in comp_row else 0
                                      for comp_row in temp_out]), 1)
                self.assertEqual(sum([1 if fname in comp_row else 0
                                      for comp_row in test_out]), 1)

            # Append the temporary files to the output for removal on teardown
            self.out.extend(temp_out)
            self.out.extend(test_out)
        except (ConnectionError, ProxyError, TimeoutError):
            pass

        return

    def test_fetch_format_ssj_boundary_files_time_failure(self):
        """Test `fetch_format_ssj_boundary_files` time failure."""

        self.in_args[0] = dt.datetime(1000, 1, 1)
        self.in_args[1] = dt.datetime(1000, 1, 2)

        with self.assertRaisesRegex(ValueError, "unable to download"):
            dmsp_ssj_files.fetch_format_ssj_boundary_files(*self.in_args)
        return

    def test_fetch_format_ssj_boundary_files_dir_failure(self):
        """Test `fetch_format_ssj_boundary_files` output directory failure."""

        self.in_args[2] = "/fake_dir/"

        with self.assertRaisesRegex(ValueError,
                                    "can't find the output directory"):
            dmsp_ssj_files.fetch_format_ssj_boundary_files(*self.in_args)
        return


@unittest.skipIf(not no_ssj,
                 "ssj_auroral_boundary installed, cannot test failure")
class TestSSJFailure(unittest.TestCase):
    """Test for informative failure when ssj_auroral_boundaries is missing."""

    def test_import_failure(self):
        """ Test ssj_auroral_boundary import failure"""

        with self.assertRaisesRegex(ImportError,
                                    'unable to load the DMSP SSJ module'):
            from ocbpy.boundaries import dmsp_ssj_files  # NOQA F401
        return
