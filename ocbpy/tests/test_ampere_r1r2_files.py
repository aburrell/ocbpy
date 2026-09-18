#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DOI: 10.5281/zenodo.1179230
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the boundaries.ampere_r1r2_files functions."""

from glob import glob
import numpy as np
import os
import unittest

from requests.exceptions import HTTPError

from ocbpy import boundaries
import ocbpy.tests.class_common as cc

no_amp = False if hasattr(boundaries, 'ampere_r1r2_files') else True
total_files = 30  # There are 30 files available as of 18/9/2026

if not no_amp:
    from ocbpy.boundaries import ampere_r1r2_files


@unittest.skipIf(no_amp, "pyfigshare not installed, cannot test routines")
class TestAMPEREFetch(unittest.TestCase):
    """Unit tests for the AMPERE fetch functions."""

    def setUp(self):
        """Initialize the test class."""
        self.ocb_dir = os.path.split(os.path.split(cc.test_dir)[0])[0]
        self.fetch_files = list()
        self.file_id = 39529696
        self.file_name = "AMPERE_R1R2_radii_v2_2018_south.txt"
        return

    def tearDown(self):
        """Clean up the test environment."""
        if len(self.fetch_files) > 0:
            for ff in self.fetch_files:
                os.remove(ff)

        del self.ocb_dir, self.fetch_files, self.file_id, self.file_name
        return

    def test_fetch_ampere_boundary_files_basic(self):
        """Test the standard download behaviour for AMPERE files."""

        self.fetch_files = ampere_r1r2_files.fetch_ampere_boundary_files()

        self.assertEqual(total_files, len(self.fetch_files))
        file_path = os.path.join(self.ocb_dir, "boundaries")
        old_year = 0
        for ff in self.fetch_files:
            self.assertRegex(os.path.dirname(ff), file_path)
            new_year = int(ff.split("_")[-2])
            self.assertGreaterEqual(new_year, old_year)
            old_year = int(new_year)
        return

    def test_fetch_ampere_files_single(self):
        """Test fetch_ampere_boundary_file downloading for a single file."""

        self.fetch_files = ampere_r1r2_files.fetch_ampere_boundary_files(
            out_dir=cc.test_dir, file_id=self.file_id)
        self.assertEqual(len(self.fetch_files), 1)
        self.assertRegex(self.fetch_files[0], self.file_name)
        return

    def test_fetch_ampere_files_no_duplicate(self):
        """Test fetch_ampere_boundary_file not re-downloading a single file."""

        self.fetch_files = ampere_r1r2_files.fetch_ampere_boundary_files(
            out_dir=cc.test_dir, file_id=[self.file_id, self.file_id])
        self.assertEqual(len(self.fetch_files), 1)
        self.assertRegex(self.fetch_files[0], self.file_name)
        return

    def test_fetch_ampere_files_bad_outdir(self):
        """Test raises ValueError for a bad output directory."""
        with self.assertRaisesRegex(ValueError, "can't find the output dir"):
            ampere_r1r2_files.fetch_ampere_boundary_files(out_dir='not/dir')

        return

    def test_fetch_ampere_files_bad_figshare_id(self):
        """Test raises ValueError for a bad figshare ID."""
        with self.assertRaisesRegex(HTTPError, "Client Error"):
            ampere_r1r2_files.fetch_ampere_boundary_files(figshare_id=-1)

        return


@unittest.skipIf(no_amp, "pyfigshare not installed, cannot test routines")
class TestAMPEREFormat(cc.TestLogWarnings):
    """Unit tests for `format_ssj_boundary_files`."""

    def setUp(self):
        """Initialize the test class."""
        super().setUp()

        self.test_dir = cc.test_dir
        self.comp_files = [os.path.join(cc.test_dir,
                                        "amp_south_radii.eab"),
                           os.path.join(cc.test_dir,
                                        "amp_south_radii.ocb")]
        self.pyfigshare_files = [
            os.path.join(cc.test_dir, "AMPERE_R1R2_radii_v2_2010_south.txt")]
        self.out = list()
        self.ldtype = ['|U50', '|U50', float, float, float, float]
        return

    def tearDown(self):
        """Clean up the test environment."""
        if len(self.out) > 0:
            for fout in self.out:
                os.remove(fout)

        del self.out, self.test_dir, self.pyfigshare_files, self.comp_files
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
        self.assertEqual(len(self.out), len(self.comp_files),
                         msg="".join(["unexpected number of boundary files: ",
                                      repr(self.out), " != ",
                                      repr(self.comp_files)]))

        # Compare the non-header data (since header has creation date)
        for fout in self.out:
            # Get the comparison filename
            fname = os.path.join(cc.test_dir, os.path.split(fout)[-1])
            self.assertIn(fname, self.comp_files)

            # Load the data
            test_out = np.genfromtxt(fout, dtype=self.ldtype)
            comp_out = np.genfromtxt(fname, dtype=self.ldtype)

            # Test the number of rows and columns
            self.assertTupleEqual(test_out.shape, comp_out.shape)

            # Test the data in each row
            for j, test_row in enumerate(test_out):
                if limit_line_comp:
                    # Only compare the time and FOM
                    for i in [0, 1, 5]:
                        self.assertEqual(test_row[i], comp_out[j][i])
                else:
                    self.assertListEqual(list(test_row), list(comp_out[j]))
        return

    def test_format_ampere_boundary_files_default(self):
        """Test the default implementation of format_ampere_boundary_files."""

        self.out = ampere_r1r2_files.format_ampere_boundary_files(
            self.pyfigshare_files)
        self.eval_formatted_output()
        return

    def test_format_ampere_boundary_files_single_file_input(self):
        """Test `format_ampere_boundary_files` with a single file input."""

        self.out = ampere_r1r2_files.format_ampere_boundary_files(
            self.pyfigshare_files[0])
        self.eval_formatted_output()
        return

    def test_format_ampere_boundary_files_mixed_input(self):
        """Test formatting with mixed good/bad file input."""

        # Create an empty, badly formatted input file
        self.pyfigshare_files.append(os.path.join(cc.test_dir, "".join([
            "AMPERE_R1R2_radii_v0_1999_south.txt"])))
        with open(self.pyfigshare_files[-1], "w") as fp:
            fp.write("#sc date time\n47 2010-12-31 00:00:00")

        # Initialize the logger warnings
        self.lwarn = "unable to format"

        # Format the files
        self.out = ampere_r1r2_files.format_ampere_boundary_files(
            self.pyfigshare_files)

        # Evaluate the good output
        self.eval_formatted_output()

        # Evaluate the logging messages
        self.eval_logging_message()

        # Prepare the empty file for clean up
        self.out.append(self.pyfigshare_files[-1])
        return

    def test_format_ampere_boundary_files_diff_bounds(self):
        """Test format_ampere_boundary_files with a different boundaries."""

        self.out = ampere_r1r2_files.format_ampere_boundary_files(
            self.pyfigshare_files, ocb_bnd='r1', eab_bnd='r2')
        self.eval_formatted_output(limit_line_comp=True)

        return

    def test_format_ampere_boundary_files_notafile_failure(self):
        """Test create_ampere_boundary_files bad filename failure."""

        with self.assertRaisesRegex(ValueError, "empty list of input files"):
            # Try to read in a bad filename
            ampere_r1r2_files.format_ampere_boundary_files([cc.test_dir])

        # Test the logging output
        self.lwarn = "bad input file"
        self.eval_logging_message()
        return

    def test_format_ampere_boundary_files_empty_failure(self):
        """Test format_ampere_boundary_files no input failure."""

        with self.assertRaisesRegex(ValueError, "empty list of input files"):
            ampere_r1r2_files.format_ampere_boundary_files([])
        return

    def test_format_ampere_boundary_files_ocb_failure(self):
        """Test format_ampere_boundary_files bad OCB key failure."""

        with self.assertRaisesRegex(ValueError, "unknown OCB proxy"):
            ampere_r1r2_files.format_ampere_boundary_files(
                self.pyfigshare_files, ocb_bnd='ocb')
        return

    def test_format_ampere_boundary_files_eab_failure(self):
        """Test format_ampere_boundary_files bad EAB key failure."""

        with self.assertRaisesRegex(ValueError, "unknown EAB proxy"):
            ampere_r1r2_files.format_ampere_boundary_files(
                self.pyfigshare_files, eab_bnd='eab')
        return

    def test_format_ampere_boundary_files_outdir_failure(self):
        """Test format_ampere_boundary_files bad output dir failure."""

        with self.assertRaisesRegex(ValueError, "can't find the output"):
            ampere_r1r2_files.format_ampere_boundary_files(
                self.pyfigshare_files, out_dir='/not/a/dir')
        return


@unittest.skipIf(no_amp, "pyfigshare not installed, cannot test routines")
class TestAMPEREFetchFormat(unittest.TestCase):
    """Unit tests for the combined fetch/format function."""

    def setUp(self):
        """Initialize the test class."""
        self.test_dir = os.path.split(cc.test_dir)[0]
        self.comp_files = [os.path.join(cc.test_dir,
                                        "amp_south_radii.eab"),
                           os.path.join(cc.test_dir,
                                        "amp_south_radii.ocb"),
                           os.path.join(cc.test_dir,
                                        "amp_north_radii.eab"),
                           os.path.join(cc.test_dir,
                                        "amp_north_radii.ocb")]
        self.out = list()
        self.ldtype = ['|U50', '|U50', float, float, float, float]
        return

    def tearDown(self):
        """Clean up the test environment."""
        if len(self.out) > 0:
            for ff in self.out:
                os.remove(ff)

        del self.out, self.comp_files, self.ldtype, self.test_dir
        return

    def eval_file_output(self):
        """Evaluate the file outputs."""
        # Ensure the number of files is correct
        self.assertEqual(len(self.out), len(self.comp_files))

        # Compare the data
        for fout in self.out:
            # Get the comparison filename
            fname = os.path.join(cc.test_dir, os.path.split(fout)[-1])
            self.assertIn(fname, self.comp_files)

            # Load the data
            test_out = np.genfromtxt(fout, dtype=self.ldtype)
            comp_out = np.genfromtxt(fname, dtype=self.ldtype)

            # The comparison data will only have one spacecraft, while
            # the routine includes data from three spacecraft
            self.assertGreater(test_out.shape, comp_out.shape)

            # Test the data in each row
            for comp_row in comp_out:
                self.assertIn(comp_row, test_out)
            return

    def test_fetch_format_ampere_boundary_files_default(self):
        """Test success with `fetch_format_ampere_boundary_files` defaults."""
        self.out = ampere_r1r2_files.fetch_format_ampere_boundary_files(
            out_dir=self.test_dir)

        self.eval_file_output()
        return

    def test_fetch_format_ampere_boundary_files_no_rm_temp(self):
        """Test `fetch_format_ampere_boundary_files` with old temp files."""

        self.out = ampere_r1r2_files.fetch_format_ampere_boundary_files(
            out_dir=self.test_dir, rm_temp=False)

        self.eval_file_output()

        # See how many files of the temporary type were downloaded
        temp_out = glob(os.path.join(self.test_dir,
                                     "AMPERE_R1R2_radii_v?_????_?????.txt"))

        self.assertEqual(len(temp_out), total_files,
                         msg="{:d}/{:d} AMPERE files downloaded".format(
                             len(temp_out), total_files))

        # Append the temporary files to the output for removal on teardown
        self.out.extend(temp_out)

        return


@unittest.skipIf(not no_amp, "pyfigshare installed, cannot test failure")
class TestAMPEREFailure(unittest.TestCase):
    """Test for informative failure when ampere_r1r2_boundaries is missing."""

    def test_import_failure(self):
        """ Test ssj_auroral_boundary import failure"""

        with self.assertRaisesRegex(ImportError,
                                    'unable to load `pyfigshare` module'):
            from ocbpy.boundaries import ampere_r1r2_files  # NOQA F401
        return
