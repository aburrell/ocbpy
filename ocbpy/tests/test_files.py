#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the boundaries.files functions."""

import datetime as dt
from io import StringIO
import logging
import os
import unittest

import ocbpy
from ocbpy.boundaries import files


class TestDMSPFileMethods(unittest.TestCase):
    """"Unit tests for the DMSP SSJ file routines."""

    def setUp(self):
        """Initialize the test case by copying over necessary files."""
        self.test_dmsp = os.path.join(os.path.dirname(ocbpy.__file__), "tests",
                                      "test_data", "dmsp-ssj_north_out.ocb")
        self.temp_files = [os.path.join(files.get_boundary_directory(),
                                        "dmsp-ssj_north_out1.ocb"),
                           os.path.join(files.get_boundary_directory(),
                                        "dmsp-ssj_north_out2.ocb"),
                           os.path.join(files.get_boundary_directory(),
                                        "unknown_north_20090101_20100101.ocb"),
                           os.path.join(files.get_boundary_directory(),
                                        "unknown_north_out.ocb")]
        self.comp_dict = {"instrument": "dmsp-ssj", "hemisphere": 1,
                          "stime": dt.datetime(2010, 12, 31),
                          "etime": dt.datetime(2011, 1, 1)}
        self.out = list()
        self.tfile = u''

        self.lwarn = u''
        self.lout = u''
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)
        return

    def tearDown(self):
        """Clean up the test environment."""
        for self.tfile in self.temp_files:
            if os.path.isfile(self.tfile):
                os.remove(self.tfile)

        del self.tfile, self.temp_files, self.test_dmsp, self.comp_dict
        del self.out, self.lwarn, self.lout, self.log_capture
        return

    def test_no_short_name_one_file(self):
        """Test get_default_file for dmsp-ssj with one boundary file."""
        # Copy over one temporary file to the boundary directory
        os.system("cp {:s} {:s}".format(self.test_dmsp, self.temp_files[0]))

        # Get the default file and instrument
        self.out = files.get_default_file(**self.comp_dict)

        self.assertEqual(self.out[0], self.temp_files[0])
        self.assertEqual(self.out[1], self.comp_dict['instrument'])
        return

    def test_no_short_name_mult_files(self):
        """Test get_default_file for dmsp-ssj with one boundary file."""
        # Copy over one temporary file to the boundary directory
        for self.tfile in self.temp_files[:2]:
            os.system("cp {:s} {:s}".format(self.test_dmsp, self.tfile))

        # Get the default file and instrument
        self.out = files.get_default_file(**self.comp_dict)

        self.assertTrue(self.out[0] in self.temp_files)
        self.assertEqual(self.out[1], self.comp_dict['instrument'])
        return

    def test_good_unknown_inst_file(self):
        """Test get_boundary_file for a good unknown instrument file."""
        # Copy over one temporary file to the boundary directory
        os.system("cp {:s} {:s}".format(self.test_dmsp, self.temp_files[2]))
        self.tfile = os.path.basename(self.temp_files[2])

        # Get the default file and instrument
        self.out = files.get_boundary_files()

        self.assertTrue(self.tfile in self.out.keys())
        self.assertListEqual(
            sorted([kk for kk in self.out[self.tfile].keys()]),
            [u'etime', u'hemisphere', u'instrument', u'stime'])
        return

    def test_bad_unknown_inst_file(self):
        """Test get_boundary_file for a bad unknown instrument file."""
        self.lwarn = u'Unknown boundary file present'

        # Copy over one temporary file to the boundary directory
        os.system("cp {:s} {:s}".format(self.test_dmsp, self.temp_files[3]))
        self.tfile = os.path.basename(self.temp_files[3])

        # Get the default file and instrument
        self.out = files.get_boundary_files()
        self.lout = self.log_capture.getvalue()

        # Test logging error message and data output
        self.assertTrue(self.lout.find(self.lwarn) >= 0)
        self.assertTrue(self.tfile in self.out.keys())
        self.assertListEqual(
            sorted([kk for kk in self.out[self.tfile].keys()]),
            [u'hemisphere', u'instrument'])
        return

    def test_get_default_from_many_options(self):
        """Test get_default_file with good ranges with other files present."""
        # Copy over one temporary file to the boundary directory
        os.system("cp {:s} {:s}".format(self.test_dmsp, self.temp_files[0]))
        self.tfile = os.path.basename(self.temp_files[0])

        # Get the default file and instrument
        self.out = files.get_default_file(self.comp_dict['stime'],
                                          self.comp_dict['etime'],
                                          self.comp_dict['hemisphere'])

        # Evaluate the output
        self.assertRegex(self.out[0], 'amp_north_radii.ocb')
        self.assertRegex(self.out[1], 'ampere')
        return


class TestFilesMethods(unittest.TestCase):
    """Unit tests for functions in the `files` sub-module."""

    def setUp(self):
        """Initialize the test class."""
        self.out = None
        self.orig_file = ocbpy.__file__
        self.comp_dict = {'ocb': {'amp_north_radii.ocb':
                                  {'instrument': 'amp', 'hemisphere': 1,
                                   'stime': dt.datetime(2010, 1, 1, 0, 0),
                                   'etime': dt.datetime(2017, 1, 1, 0, 0)},
                                  'wic_north_circle.ocb':
                                  {'instrument': 'wic', 'hemisphere': 1,
                                   'stime': dt.datetime(2000, 5, 4, 0, 0),
                                   'etime': dt.datetime(2002, 11, 1, 0, 0)},
                                  'si12_north_circle.ocb':
                                  {'instrument': 'si12', 'hemisphere': 1,
                                   'stime': dt.datetime(2000, 5, 4, 0, 0),
                                   'etime': dt.datetime(2002, 11, 1, 0, 0)},
                                  'si13_north_circle.ocb':
                                  {'instrument': 'si13', 'hemisphere': 1,
                                   'stime': dt.datetime(2000, 5, 3, 0, 0),
                                   'etime': dt.datetime(2002, 11, 1, 0, 0)},
                                  'image_north_circle.ocb':
                                  {'instrument': 'image', 'hemisphere': 1,
                                   'stime': dt.datetime(2000, 5, 3, 0, 0),
                                   'etime': dt.datetime(2002, 11, 1, 0, 0)},
                                  'amp_south_radii.ocb':
                                  {'instrument': 'amp', 'hemisphere': -1,
                                   'stime': dt.datetime(2010, 1, 1, 0, 0),
                                   'etime': dt.datetime(2017, 1, 1, 0, 0)}},
                          'eab': {'wic_north_circle.eab':
                                  {'instrument': 'wic', 'hemisphere': 1,
                                   'stime': dt.datetime(2000, 5, 4, 0, 0),
                                   'etime': dt.datetime(2002, 11, 1, 0, 0)},
                                  'image_north_circle.eab':
                                  {'instrument': 'image', 'hemisphere': 1,
                                   'stime': dt.datetime(2000, 5, 3, 0, 0),
                                   'etime': dt.datetime(2002, 11, 1, 0, 0)},
                                  'si13_north_circle.eab':
                                  {'instrument': 'si13', 'hemisphere': 1,
                                   'stime': dt.datetime(2000, 5, 3, 0, 0),
                                   'etime': dt.datetime(2002, 11, 1, 0, 0)},
                                  'si12_north_circle.eab':
                                  {'instrument': 'si12', 'hemisphere': 1,
                                   'stime': dt.datetime(2000, 5, 4, 0, 0),
                                   'etime': dt.datetime(2002, 11, 1, 0, 0)}}}

        self.short_to_long = {"amp": "ampere", "si12": "image",
                              "si13": "image", "wic": "image", "": "image"}
        self.long_to_short = {"ampere": "amp", "": "image", "dmsp-ssj": None}
        self.inst = {1: ['', 'si13', 'si12', 'wic', 'amp', 'image', 'ampere',
                         'dmsp-ssj'],
                     -1: ['', 'amp', 'ampere', 'dmsp-ssj']}
        self.hemi = 1
        self.ikey = ''
        self.fname = None
        return

    def tearDown(self):
        """Clean the test environment."""
        if ocbpy.__file__ != self.orig_file:
            ocbpy.__file__ = self.orig_file

        del self.out, self.orig_file, self.comp_dict, self.short_to_long
        del self.inst, self.long_to_short, self.hemi, self.ikey, self.fname
        return

    def test_get_boundary_directory(self):
        """Test the default boundary directory definition."""
        self.out = files.get_boundary_directory()
        self.assertGreater(self.out.find("boundaries"), 0)
        return

    def test_get_boundary_directory_failure(self):
        """Test the failure of the default boundary directory definition."""
        ocbpy.__file__ = "/fake_dir/test_file"
        with self.assertRaisesRegex(OSError, "boundary file directory"):
            files.get_boundary_directory()
        return

    def test_get_boundary_files_unknown_boundary(self):
        """Test get_boundary_files for an unknown boundary."""
        self.out = files.get_boundary_files(bound='aaa')
        self.assertDictEqual(self.out, {})
        return

    def test_get_boundary_files(self):
        """Test the default implementation of get_boundary_files."""

        # Test only the files included with OCBpy, allow local boundary files
        # to exist.
        for bound in self.comp_dict.keys():
            with self.subTest(bound=bound):
                self.out = files.get_boundary_files(bound=bound)
                for ckey in self.comp_dict[bound].keys():
                    self.assertTrue(ckey in self.out.keys())
                    self.assertDictEqual(self.out[ckey],
                                         self.comp_dict[bound][ckey])
        return

    def test_get_default_file_bad_bound(self):
        """Test get_default_file with an unknown boundary."""
        self.out = files.get_default_file(None, None, self.hemi, bound='aaa')

        self.assertIsNone(self.out[0])
        self.assertEqual(len(self.out[1]), 0)
        return

    def test_get_default_file_none_north_any_default_inst(self):
        """Test get_default_file with no range, northern hemisphere."""
        self.out = files.get_default_file(None, None, self.hemi)

        self.assertRegex(self.out[0], 'image_north_circle.ocb')
        self.assertRegex(self.out[1], 'image')
        return

    def test_get_default_file_none_north_all(self):
        """Test get_default_file with no range, northern hemisphere."""

        # Cycle through all possible instrument names
        for ii in self.inst[self.hemi]:
            with self.subTest(ii=ii):
                self.out = files.get_default_file(None, None, self.hemi,
                                                  instrument=ii)

                if ii in self.long_to_short.keys():
                    if self.long_to_short[ii] is None:
                        # This is necessary because setUp does not run
                        # between subTests
                        self.fname = None
                    else:
                        self.fname = "{:s}_north".format(
                            self.long_to_short[ii])
                else:
                    self.fname = "{:s}_north".format(ii)

                if ii in self.short_to_long.keys():
                    iname = self.short_to_long[ii]
                else:
                    iname = ii

                if self.fname is None:
                    self.assertIsNone(self.out[0])
                else:
                    self.assertRegex(self.out[0], self.fname)
                self.assertRegex(self.out[1], iname)
        return

    def test_get_default_file_none_south_all(self):
        """Test get_default_file with no range, southern hemisphere."""
        # Set the southern hemisphere defaults
        self.hemi = -1
        self.long_to_short[''] = 'amp'
        self.short_to_long[''] = 'ampere'

        # Cycle through all possible instrument names
        for ii in self.inst[self.hemi]:
            with self.subTest(ii=ii):
                self.out = files.get_default_file(None, None, self.hemi,
                                                  instrument=ii)

                if ii in self.long_to_short.keys():
                    if self.long_to_short[ii] is None:
                        # This is necessary because setUp does not run
                        # between subTests
                        self.fname = None
                    else:
                        self.fname = "{:s}_south".format(
                            self.long_to_short[ii])
                else:
                    self.fname = "{:s}_south".format(ii)

                if ii in self.short_to_long.keys():
                    iname = self.short_to_long[ii]
                else:
                    iname = ii

                if self.fname is None:
                    self.assertIsNone(self.out[0])
                else:
                    self.assertRegex(self.out[0], self.fname)
                self.assertRegex(self.out[1], iname)
        return

    def test_get_default_good_file_times(self):
        """Test get_default_file with good ranges."""
        # Cycle through all possible instrument names
        for bound in self.comp_dict.keys():
            if bound == 'ocb':
                default_files = ['amp_north_radii.ocb',
                                 'image_north_circle.ocb']
            else:
                default_files = ['image_north_circle.eab']

            for ii in default_files:
                with self.subTest(bound=bound, ii=ii):
                    self.out = files.get_default_file(
                        self.comp_dict[bound][ii]['stime'],
                        self.comp_dict[bound][ii]['etime'], self.hemi,
                        bound=bound)

                    if(self.comp_dict[bound][ii]['instrument']
                       in self.short_to_long.keys()):
                        iname = self.short_to_long[
                            self.comp_dict[bound][ii]['instrument']]
                    else:
                        iname = self.comp_dict[bound][ii]['instrument']

                    self.assertRegex(self.out[0], ii)
                    self.assertRegex(self.out[1], iname)
        return

    def test_get_default_file_bad_file_times(self):
        """Test get_default_file with bad time ranges."""
        self.ikey = 'image_north_circle.ocb'
        self.hemi = -1
        self.out = files.get_default_file(
            self.comp_dict['ocb'][self.ikey]['stime'],
            self.comp_dict['ocb'][self.ikey]['etime'], self.hemi)

        self.assertIsNone(self.out[0])
        self.assertEqual(len(self.out[1]), 0)

        return
