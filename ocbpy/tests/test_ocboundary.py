#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
""" Tests the ocboundary class and functions
"""
from __future__ import absolute_import, unicode_literals

import datetime as dt
from io import StringIO
import logging
import numpy as np
from sys import version_info
from os import path
import unittest

import ocbpy


class TestOCBoundaryLogFailure(unittest.TestCase):
    def setUp(self):
        """ Initialize the test class"""
        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

    def tearDown(self):
        """ Tear down the test case"""
        del self.lwarn, self.lout, self.log_capture

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_bad_instrument_name(self):
        """ Test OCB initialization with bad instrument name """
        self.lwarn = u"OCB instrument must be a string"

        # Initialize the OCBoundary class with bad instrument names
        for val in [1, None, True]:
            with self.subTest(val=val):
                ocb = ocbpy.OCBoundary(instrument=val)
                self.assertIsNone(ocb.filename)
                self.assertIsNone(ocb.instrument)

                self.lout = self.log_capture.getvalue()
                # Test logging error message for each bad initialization
                self.assertTrue(self.lout.find(self.lwarn) >= 0)

        del val, ocb

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_bad_instrument_int_name(self):
        """ Test OCB initialization with a bad integer instrument name """
        self.lwarn = u"OCB instrument must be a string"

        # Initialize the OCBoundary class with bad instrument names
        ocb = ocbpy.OCBoundary(instrument=1)
        self.assertIsNone(ocb.filename)
        self.assertIsNone(ocb.instrument)

        self.lout = self.log_capture.getvalue()
        # Test logging error message for each bad initialization
        print(self.lout, self.lwarn)
        self.assertTrue(self.lout.find(self.lwarn) >= 0)

        del ocb

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_bad_instrument_none_name(self):
        """ Test OCB initialization with a bad NoneType instrument name """
        self.lwarn = u"OCB instrument must be a string"

        # Initialize the OCBoundary class with bad instrument names
        ocb = ocbpy.OCBoundary(instrument=None)
        self.assertIsNone(ocb.filename)
        self.assertIsNone(ocb.instrument)

        self.lout = self.log_capture.getvalue()
        # Test logging error message for each bad initialization
        print(self.lout, self.lwarn)
        self.assertTrue(self.lout.find(self.lwarn) >= 0)

        del ocb

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_bad_instrument_boolean_name(self):
        """ Test OCB initialization with a bad Boolean instrument name """
        self.lwarn = u"OCB instrument must be a string"

        # Initialize the OCBoundary class with bad instrument names
        ocb = ocbpy.OCBoundary(instrument=True)
        self.assertIsNone(ocb.filename)
        self.assertIsNone(ocb.instrument)

        self.lout = self.log_capture.getvalue()
        # Test logging error message for each bad initialization
        print(self.lout, self.lwarn)
        self.assertTrue(self.lout.find(self.lwarn) >= 0)

        del ocb

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_bad_file_name(self):
        """ Test OCB initialization with bad file name """
        self.lwarn = u"filename is not a string"

        # Initialize the OCBoundary class with bad instrument names
        for val in [1, True]:
            with self.subTest(val=val):
                ocb = ocbpy.OCBoundary(filename=val)
                self.assertIsNone(ocb.filename)

                self.lout = self.log_capture.getvalue()
                # Test logging error message for each bad initialization
                self.assertTrue(self.lout.find(self.lwarn) >= 0)

        del val, ocb

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_bad_file_name_int(self):
        """ Test OCB initialization with a bad file name that's an integer"""
        self.lwarn = u"filename is not a string"

        # Initialize the OCBoundary class with bad instrument names
        ocb = ocbpy.OCBoundary(filename=1)
        self.assertIsNone(ocb.filename)

        self.lout = self.log_capture.getvalue()
        # Test logging error message for each bad initialization
        self.assertTrue(self.lout.find(self.lwarn) >= 0)

        del ocb

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_bad_filename_bool(self):
        """ Test OCB initialization with a bad file name that's a Boolean"""
        self.lwarn = u"filename is not a string"

        # Initialize the OCBoundary class with bad instrument names
        ocb = ocbpy.OCBoundary(filename=True)
        self.assertIsNone(ocb.filename)

        self.lout = self.log_capture.getvalue()
        # Test logging error message for each bad initialization
        self.assertTrue(self.lout.find(self.lwarn) >= 0)

        del ocb

    def test_bad_filename(self):
        """ Test OCB initialization with a bad default file/instrument pairing
        """
        self.lwarn = u"name provided is not a file\ncannot open OCB file [hi]"

        # Try to load AMPERE data with an IMAGE file
        ocb = ocbpy.OCBoundary(filename="hi")
        self.assertIsNone(ocb.filename)

        self.lout = self.log_capture.getvalue()
        # Test logging error message for each bad initialization
        self.assertTrue(self.lout.find(self.lwarn) >= 0)

        del ocb

    def test_bad_time_structure(self):
        """ Test OCB initialization without complete time data in file
        """
        self.lwarn = u"missing time columns in"

        # Initialize without a file so that custom loading is performed
        ocb = ocbpy.OCBoundary(filename=None)
        self.assertIsNone(ocb.filename)

        # Set the filename
        ocb.filename = path.join(path.dirname(ocbpy.__file__), "tests",
                                 "test_data", "test_north_circle")
        self.assertTrue(path.isfile(ocb.filename))

        # Load the data, skipping the year
        ocb.load(ocb_cols="skip soy num_sectors phi_cent r_cent r a r_err")

        self.lout = self.log_capture.getvalue()
        # Test logging error message for the non-None bad initializations
        self.assertTrue(self.lout.find(self.lwarn) >= 0)

        del ocb


class TestOCBoundaryInstruments(unittest.TestCase):
    def setUp(self):
        """ Initialize the instrument information
        """
        self.test_dir = path.join(path.dirname(ocbpy.__file__), "tests",
                                  "test_data")
        self.inst_attrs = {"image": ["year", "soy", "num_sectors", "a",
                                     "r_err"],
                           "ampere": ["date", "time", "x", "y", "fom"],
                           "dmsp-ssj": ["date", "time", "sc", "x", "y", "fom",
                                        "x_1", "x_2", "y_1", "y_2"]}
        self.not_attrs = {"image": ["date", "time", "x", "y", "fom", "x_1",
                                    "x_2", "y_1", "y_2", "sc"],
                          "ampere": ["year", "soy", "x_1", "y_1", "x_2",
                                     "y_2", "sc", "num_sectors", "a",
                                     "r_err"],
                          "dmsp-ssj": ["year", "soy", "num_sectors", "a",
                                       "r_err"]}
        self.inst_init = [{"instrument": "image", "hemisphere": 1,
                           "filename": path.join(self.test_dir,
                                                 "test_north_circle")},
                          {"instrument": "dmsp-ssj", "hemisphere": 1,
                           "filename": path.join(self.test_dir,
                                                 "dmsp-ssj_north_out.ocb")},
                          {"instrument": "dmsp-ssj", "hemisphere": -1,
                           "filename": path.join(self.test_dir,
                                                 "dmsp-ssj_south_out.ocb")},
                          {"instrument": "ampere", "hemisphere": -1,
                           "filename": path.join(self.test_dir,
                                                 "test_south_circle")}]
        self.ocb = None

    def tearDown(self):
        del self.test_dir, self.inst_attrs, self.inst_init, self.ocb

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_instrument_loading(self):
        """ Test OCB initialization with good instrument names """
        for ocb_kwargs in self.inst_init:
            with self.subTest(ocb_kwargs=ocb_kwargs):
                self.ocb = ocbpy.OCBoundary(**ocb_kwargs)

                for tattr in self.inst_attrs[ocb_kwargs['instrument']]:
                    self.assertTrue(hasattr(self.ocb, tattr))

                for tattr in self.not_attrs[ocb_kwargs['instrument']]:
                    self.assertFalse(hasattr(self.ocb, tattr))

        del ocb_kwargs, tattr

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_image_loading(self):
        """ Test OCB initialization for IMAGE names """
        self.ocb = ocbpy.OCBoundary(**self.inst_init[0])

        for tattr in self.inst_attrs['image']:
            self.assertTrue(hasattr(self.ocb, tattr))

        for tattr in self.not_attrs['image']:
            self.assertFalse(hasattr(self.ocb, tattr))

        del tattr

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_ampere_loading(self):
        """ Test OCB initialization for AMPERE names """
        self.ocb = ocbpy.OCBoundary(**self.inst_init[3])

        for tattr in self.inst_attrs['ampere']:
            self.assertTrue(hasattr(self.ocb, tattr))

        for tattr in self.not_attrs['ampere']:
            self.assertFalse(hasattr(self.ocb, tattr))

        del tattr

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_dmsp_ssj_north_loading(self):
        """ Test OCB initialization for DMSP-SSJ North names """
        self.ocb = ocbpy.OCBoundary(**self.inst_init[1])

        for tattr in self.inst_attrs['dmsp-ssj']:
            self.assertTrue(hasattr(self.ocb, tattr))

        for tattr in self.not_attrs['dmsp-ssj']:
            self.assertFalse(hasattr(self.ocb, tattr))

        del tattr

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_dmsp_ssj_south_loading(self):
        """ Test OCB initialization for DMSP-SSJ South names """
        self.ocb = ocbpy.OCBoundary(**self.inst_init[2])

        for tattr in self.inst_attrs['dmsp-ssj']:
            self.assertTrue(hasattr(self.ocb, tattr))

        for tattr in self.not_attrs['dmsp-ssj']:
            self.assertFalse(hasattr(self.ocb, tattr))

        del tattr


class TestOCBoundaryMethodsGeneral(unittest.TestCase):
    def setUp(self):
        """ Initialize the OCBoundary object using the empty file
        """
        ocb_dir = path.dirname(ocbpy.__file__)
        self.set_empty = {"filename": path.join(ocb_dir, "tests", "test_data",
                                                "test_empty"),
                          "instrument": "image"}
        self.set_default = {"filename": path.join(ocb_dir, "tests",
                                                  "test_data",
                                                  "test_north_circle"),
                            "instrument": "image"}
        self.assertTrue(path.isfile(self.set_empty['filename']))
        self.assertTrue(path.isfile(self.set_default['filename']))
        self.ocb = None

        if version_info.major == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp
            self.assertRegex = self.assertRegexpMatches

    def tearDown(self):
        del self.set_empty, self.set_default, self.ocb

    def test_default_repr(self):
        """ Test the default class representation """
        self.ocb = ocbpy.ocboundary.OCBoundary(**self.set_default)

        self.assertRegex(self.ocb.__repr__(), "Open-Closed Boundary file:")
        self.assertTrue(self.ocb.__str__() == self.ocb.__repr__())

    def test_short_repr(self):
        """ Test the default class representation """
        self.ocb = ocbpy.ocboundary.OCBoundary(**self.set_default)
        self.ocb.records = 1

        self.assertRegex(self.ocb.__repr__(), "1 records from")
        self.assertTrue(self.ocb.__str__() == self.ocb.__repr__())

    def test_bad_rfunc_inst(self):
        """Test failure setting default rfunc for unknown instrument"""

        with self.assertRaisesRegex(ValueError, "unknown instrument"):
            self.set_empty['instrument'] = 'bad'
            self.set_empty['rfunc'] = None
            self.ocb = ocbpy.ocboundary.OCBoundary(**self.set_empty)

    def test_no_file_repr(self):
        """ Test the unset class representation """

        self.ocb = ocbpy.ocboundary.OCBoundary(filename=None)

        if version_info.major == 2:
            self.assertRegexpMatches(self.ocb.__repr__(),
                                     "No Open-Closed Boundary file specified")
        else:
            self.assertRegex(self.ocb.__repr__(),
                             "No Open-Closed Boundary file specified")

    def test_empty_file_repr(self):
        """ Test the class representation with an empty data file"""

        self.ocb = ocbpy.ocboundary.OCBoundary(**self.set_empty)

        if version_info.major == 2:
            self.assertRegexpMatches(self.ocb.__repr__(), "No data loaded")
        else:
            self.assertRegex(self.ocb.__repr__(), "No data loaded")

    def test_nofile_init(self):
        """ Ensure that the class can be initialised without loading a file.
        """
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=None)

        self.assertIsNone(self.ocb.filename)
        self.assertIsNone(self.ocb.dtime)
        self.assertEqual(self.ocb.records, 0)


class TestOCBoundaryMethodsNorth(unittest.TestCase):
    def setUp(self):
        """ Initialize the OCBoundary object using the test file
        """
        self.set_north = {'filename': path.join(path.dirname(ocbpy.__file__),
                                                "tests", "test_data",
                                                "test_north_circle"),
                          'instrument': 'image'}
        self.assertTrue(path.isfile(self.set_north['filename']))
        self.ocb = ocbpy.ocboundary.OCBoundary(**self.set_north)
        self.ocb.rec_ind = 27

        self.lon = np.linspace(0.0, 360.0, num=6)
        self.lat = np.linspace(0.0, 90.0, num=len(self.lon))
        self.ocb_lat = [np.nan, 11.25588586, 30.35153908, 47.0979063,
                        66.59889231, 86.86586231]
        self.ocb_mlt = [np.nan, 4.75942194, 9.76745427, 14.61843964,
                        19.02060793, 17.832]
        self.r_corr = 0.0
        self.out = None

    def tearDown(self):
        del self.ocb, self.set_north, self.lon, self.lat, self.ocb_lat
        del self.ocb_mlt, self.r_corr, self.out

    def test_attrs(self):
        """ Test the default attributes in the north
        """
        for self.out in ["filename", "instrument", "hemisphere", "records",
                         "rec_ind", "dtime", "phi_cent", "r_cent", "r",
                         "boundary_lat"]:
            self.assertTrue(hasattr(self.ocb, self.out))

        # Ensure optional attributes are absent
        for self.out in ["aacgm_boundary_lon", "aacgm_boundary_lat"]:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_image_attrs(self):
        """ Test IMAGE attributes in the north
        """
        for self.out in ["num_sectors", "year", "soy", "r_err", "a"]:
            self.assertTrue(hasattr(self.ocb, self.out))

    def test_ampere_attrs(self):
        """ Test AMPERE attributes don't exist when IMAGE is loaded
        """
        for self.out in ['date', 'time', 'x', 'y', 'fom']:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_dmspssj_attrs(self):
        """ Test DMSP-SSJ attributes don't exist when IMAGE is loaded
        """

        for self.out in ['sc', 'date', 'time', 'x', 'y', 'fom', 'x_1', 'x_2',
                         'y_1', 'y_2']:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_load(self):
        """ Ensure correctly loaded defaults in the north
        """
        self.assertGreater(self.ocb.records, 0)
        self.assertEqual(self.ocb.boundary_lat, 74.0)

    def test_partial_load(self):
        """ Ensure limited sections of a file can be loaded in the north
        """

        stime = self.ocb.dtime[0] + dt.timedelta(seconds=1)
        etime = self.ocb.dtime[-1] - dt.timedelta(seconds=1)

        # Load all but the first and last records
        self.out = ocbpy.ocboundary.OCBoundary(filename=self.ocb.filename,
                                               instrument=self.ocb.instrument,
                                               stime=stime, etime=etime,
                                               boundary_lat=75.0)

        self.assertEqual(self.ocb.records, self.out.records + 2)
        self.assertEqual(self.out.boundary_lat, 75.0)

    def test_first_good(self):
        """ Test to see that we can find the first good point in the north
        """
        self.ocb.rec_ind = -1

        self.ocb.get_next_good_ocb_ind()

        self.assertGreater(self.ocb.rec_ind, -1)
        self.assertLess(self.ocb.rec_ind, self.ocb.records)

    def test_normal_coord_north_float(self):
        """ Test the normalisation calculation in the north
        """
        self.out = self.ocb.normal_coord(self.lat[-1], self.lon[-1]/15.0)
        self.assertAlmostEqual(float(self.out[0]), self.ocb_lat[-1])
        self.assertAlmostEqual(float(self.out[1]), self.ocb_mlt[-1])
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_north_array(self):
        """ Test the normalisation calculation in the north with arryay input
        """
        self.out = self.ocb.normal_coord(self.lat, self.lon/15.0)

        self.assertTrue(np.all(np.less(abs(self.out[0] - self.ocb_lat), 1.0e-7,
                                       where=~np.isnan(self.out[0])) |
                               np.isnan(self.out[0])))
        self.assertTrue(np.all(np.less(abs(self.out[1] - self.ocb_mlt), 1.0e-7,
                                       where=(~np.isnan(self.out[1]))) |
                               np.isnan(self.out[1])))
        self.assertTrue(np.where(np.isnan(self.out[0])) ==
                        np.where(np.isnan(self.ocb_lat)))
        self.assertTrue(np.where(np.isnan(self.out[1])) ==
                        np.where(np.isnan(self.ocb_mlt)))
        self.assertTrue(np.all(self.out[2] == self.r_corr))

    def test_normal_coord_north_alt_mag_label(self):
        """ Test the normalisation calculation with good, but odd coord label
        """
        self.out = self.ocb.normal_coord(self.lat[-1], self.lon[-1]/15.0,
                                         coords='Mag')
        self.assertAlmostEqual(float(self.out[0]), self.ocb_lat[-1])
        self.assertAlmostEqual(float(self.out[1]), self.ocb_mlt[-1])
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_north_geodetic(self):
        """ Test the geodetic normalisation calculation in the north
        """
        self.out = self.ocb.normal_coord(self.lat[-1], self.lon[-1]/15.0,
                                         coords='geodetic')
        self.assertAlmostEqual(float(self.out[0]), 79.2631, places=3)
        self.assertAlmostEqual(float(self.out[1]), 19.3839, places=3)
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_north_geocentric(self):
        """ Test the geocentric normalisation calculation in the north
        """
        self.out = self.ocb.normal_coord(self.lat[-1], self.lon[-1]/15.0,
                                         coords='geocentric')
        self.assertAlmostEqual(float(self.out[0]), 79.2654, places=3)
        self.assertAlmostEqual(float(self.out[1]), 19.3852, places=3)
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_north_w_south(self):
        """ Test the normalisation calculation in the north with southern lat
        """
        self.out = self.ocb.normal_coord(-self.lat[-1], self.lon[-1]/15.0)

        self.assertEqual(len(self.out), 3)
        self.assertTrue(np.all(np.isnan(self.out)))

    def test_normal_coord_low_rec_ind(self):
        """ Test the normalization calculation failure with low record index
        """
        self.ocb.rec_ind = -1
        self.out = self.ocb.normal_coord(self.lat[-1], self.lon[-1]/15.0)

        self.assertEqual(len(self.out), 3)
        self.assertTrue(np.all(np.isnan(self.out)))

    def test_normal_coord_high_rec_ind(self):
        """ Test the normalization calculation failure with high record index
        """
        self.ocb.rec_ind = self.ocb.records + 1
        self.out = self.ocb.normal_coord(self.lat[-1], self.lon[-1]/15.0)

        self.assertEqual(len(self.out), 3)
        self.assertTrue(np.all(np.isnan(self.out)))

    def test_revert_coord_north_float(self):
        """ Test the reversion to AACGM coordinates in the north
        """
        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr)
        self.assertAlmostEqual(self.out[0], self.lat[-2])
        self.assertAlmostEqual(self.out[1], self.lon[-2]/15.0)

    def test_revert_coord_north_array(self):
        """ Test the reversion to AACGM coordinates in the north for an array
        """
        self.out = self.ocb.revert_coord(self.ocb_lat, self.ocb_mlt,
                                         self.r_corr)

        self.assertTrue(np.all(np.less(abs(self.out[0] - self.lat), 1.0e-7,
                                       where=~np.isnan(self.out[0])) |
                               (np.isnan(self.out[0]))))
        self.assertTrue(np.all(np.less(abs(self.out[1] - self.lon/15.0),
                                       1.0e-7,
                                       where=(~np.isnan(self.out[1])
                                              & (self.lat < 90.0))) |
                               np.isnan(self.out[0]) | (self.lat >= 90.0)))
        self.assertTrue(np.where(np.isnan(self.out[0]))
                        == np.where(np.isnan(self.ocb_lat)))
        self.assertTrue(np.where(np.isnan(self.out[1]))
                        == np.where(np.isnan(self.ocb_mlt)))

    def test_revert_coord_north_coord_label(self):
        """ Test the reversion to AACGM coordinates in the north with Mag label
        """
        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr, coords='MAG')
        self.assertAlmostEqual(self.out[0], self.lat[-2])
        self.assertAlmostEqual(self.out[1], self.lon[-2]/15.0)

    def test_revert_coord_north_geodetic(self):
        """ Test the reversion to geodetic coordinates in the north
        """
        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr, coords='geodetic')
        self.assertAlmostEqual(float(self.out[0]), 77.13321838, places=3)
        self.assertAlmostEqual(float(self.out[1]), 19.18124285, places=3)

    def test_revert_coord_north_geocentric(self):
        """ Test the reversion to geocentric coordinates in the north
        """
        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr, coords='geocentric')
        self.assertAlmostEqual(float(self.out[0]), 77.05394766, places=3)
        self.assertAlmostEqual(float(self.out[1]), 19.18124285, places=3)

    def test_revert_coord_north_w_south(self):
        """ Test the reversion calculation in the north with southern lat
        """
        self.out = self.ocb.revert_coord(-self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr)
        self.assertEqual(len(self.out), 2)
        self.assertTrue(np.all(np.isnan(self.out)))

    def test_revert_coord_low_rec_ind(self):
        """ Test the reversion calculation failure with low record index
        """
        self.ocb.rec_ind = -1

        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr)
        self.assertEqual(len(self.out), 2)
        self.assertTrue(np.all(np.isnan(self.out)))

    def test_revert_coord_high_rec_ind(self):
        """ Test the reversion calculation failure with high record index
        """
        self.ocb.rec_ind = self.ocb.records + 1

        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr)
        self.assertEqual(len(self.out), 2)
        self.assertTrue(np.all(np.isnan(self.out)))

    def test_default_boundary_input(self):
        """ Test to see that the boundary latitude has the correct sign
        """
        self.assertEqual(self.ocb.boundary_lat, 74.0)

    def test_mismatched_boundary_input(self):
        """ Test to see that the boundary latitude has the incorrect sign
        """
        self.set_north['hemisphere'] = -1
        self.out = ocbpy.ocboundary.OCBoundary(**self.set_north)
        self.assertEqual(self.out.boundary_lat, -74.0)

    def test_mismatched_boundary_input_correction(self):
        """ Test to see that the boundary latitude corrects the sign
        """
        self.set_north['boundary_lat'] = -70.0
        self.out = ocbpy.ocboundary.OCBoundary(**self.set_north)
        self.assertEqual(self.out.boundary_lat, 70.0)

    def test_retrieve_all_good_ind(self):
        """ Test routine that retrieves all good indices, record set at start
        """
        self.ocb.rec_ind = -1
        self.out = ocbpy.ocboundary.retrieve_all_good_indices(self.ocb)

        self.assertEqual(self.out[0], 27)
        self.assertEqual(self.out[1], 31)
        self.assertEqual(len(self.out), 36)
        self.assertEqual(self.ocb.rec_ind, -1)

    def test_retrieve_all_good_ind_init_middle(self):
        """ Test routine that retrieves all good indices, record set at middle
        """
        self.ocb.rec_ind = 65
        self.out = ocbpy.ocboundary.retrieve_all_good_indices(self.ocb)

        self.assertEqual(self.out[0], 27)
        self.assertEqual(self.out[1], 31)
        self.assertEqual(len(self.out), 36)
        self.assertEqual(self.ocb.rec_ind, 65)

    def test_retrieve_all_good_ind_empty(self):
        """ Test routine that retrieves all good indices, no data loaded
        """
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=None)
        self.out = ocbpy.ocboundary.retrieve_all_good_indices(self.ocb)

        self.assertEqual(len(self.out), 0)

    def test_aacgm_boundary_location(self):
        """ Test the calculation of the OCB in AACGM coordinates in the north
        """
        # Add new attributes
        self.ocb.get_aacgm_boundary_lat(aacgm_lon=self.lon)

        # Ensure new attriutes were added
        self.assertTrue(hasattr(self.ocb, "aacgm_boundary_lon"))
        self.assertTrue(hasattr(self.ocb, "aacgm_boundary_lat"))

        # Test shape of new attributes
        self.assertEqual(len(self.ocb.aacgm_boundary_lon), self.ocb.records)
        self.assertEqual(len(self.ocb.aacgm_boundary_lon[0]), len(self.lon))
        self.assertEqual(len(self.ocb.aacgm_boundary_lat[0]), len(self.lon))

        # Test value of longitude attribute
        self.assertEqual(sum(self.lon[:-1]
                             - self.ocb.aacgm_boundary_lon[0][:-1]), 0)
        self.assertEqual(sum(self.lon[:-1]
                             - self.ocb.aacgm_boundary_lon[-1][:-1]), 0)

        # Test the value of the latitude attriubte at the good record location
        # Also tests that specifying the same longitude locations twice is ok
        self.test_aacgm_boundary_location_good()

    def test_aacgm_boundary_location_good(self):
        """ Test the calculation of the OCB in AACGM coordinates in the north
        """
        rind = 27

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(aacgm_lon=self.lon, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(np.all(self.ocb.aacgm_boundary_lat[rind] > 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].min(),
                               73.26939247752293)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].max(),
                               78.52813223696786)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmax(), 4)

    def test_aacgm_boundary_location_bad(self):
        """ Test the calclation of the OCB in AACGM coordinates for limited MLTs
        """
        rind = 2

        # Add the attriubte at the bad location
        self.ocb.get_aacgm_boundary_lat(aacgm_lon=self.lon, rec_ind=rind)

        # Test value of latitude attribute
        self.assertFalse(np.all(self.ocb.aacgm_boundary_lat[rind] > 0.0))
        self.assertTrue(np.any(self.ocb.aacgm_boundary_lat[rind] > 0.0))
        self.assertTrue(np.any(np.isnan(self.ocb.aacgm_boundary_lat[rind])))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind][1],
                               72.82502115387997)

    def test_aacgm_boundary_location_no_input(self):
        """ Test failure of OCB AACGM location calculation for no input
        """
        with self.assertRaises(TypeError):
            self.ocb.get_aacgm_boundary_lat()

    def test_aacgm_boundary_location_no_overwrite(self):
        """ Ensure no overwrite when re-calculating OCB AACGM locations
        """
        log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        # Initialize the attributes with values for the good location
        rind = 27
        self.test_aacgm_boundary_location_good()
        # This should not raise a warning
        self.ocb.get_aacgm_boundary_lat(150.0, rec_ind=rind-1)
        # This should raise a warning
        self.ocb.get_aacgm_boundary_lat(150.0, rec_ind=rind)

        self.out = log_capture.getvalue()
        # Test logging error message for only one warning about boundary update
        self.assertTrue(self.out.find(u"unable to update AACGM boundary") >= 0)

        del log_capture

    def test_aacgm_boundary_location_overwrite(self):
        """ Test ability to overwrite OCB AACGM location
        """
        # Initialize the attributes with values for the good location
        self.test_aacgm_boundary_location_good()

        # Specify a new longitude for that location
        rind = 27
        self.ocb.get_aacgm_boundary_lat(150.0, rec_ind=rind, overwrite=True)

        # Test value of latitude attribute
        self.assertFalse(hasattr(self.ocb.aacgm_boundary_lat[rind], "shape"))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind],
                               74.8508209365)

    def test_aacgm_boundary_location_lon_range(self):
        """ Test failure of OCB AACGM location with different valued longitude
        """
        self.lon[self.lon > 180.0] -= 360.0
        self.ocb.get_aacgm_boundary_lat(aacgm_lon=self.lon)

        # Test the attributes with values for the good location
        self.test_aacgm_boundary_location_good()


class TestOCBoundaryMethodsSouth(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file
        """

        self.set_south = {"filename": path.join(path.dirname(ocbpy.__file__),
                                                "tests", "test_data",
                                                "test_south_circle"),
                          "instrument": "ampere",
                          "hemisphere": -1,
                          "rfunc": ocbpy.ocb_correction.circular}
        self.ocb = ocbpy.ocboundary.OCBoundary(**self.set_south)
        self.ocb.rec_ind = 8

        self.lon = np.linspace(0.0, 360.0, num=6)
        self.lat = np.linspace(-90.0, 0.0, num=len(self.lon))
        self.ocb_lat = [-86.8, -58.14126906, -30.46277504, -5.44127327,
                        22.16097829, np.nan]
        self.ocb_mlt = [6.0, 4.91857824, 9.43385497, 14.28303702, 19.23367655,
                        np.nan]
        self.r_corr = 0.0
        self.out = None

    def tearDown(self):
        del self.ocb, self.set_south, self.lon, self.lat, self.ocb_lat
        del self.ocb_mlt, self.r_corr, self.out

    def test_attrs(self):
        """ Test the default attributes in the south """

        for self.out in ["filename", "instrument", "hemisphere", "records",
                         "rec_ind", "dtime", "phi_cent", "r_cent", "r",
                         "boundary_lat"]:
            self.assertTrue(hasattr(self.ocb, self.out))

        # Ensure optional attributes are absent
        for self.out in ["aacgm_boundary_lon", "aacgm_boundary_lat"]:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_image_attrs(self):
        """ Test that IMAGE attributes are not available in the south"""

        for self.out in ["num_sectors", "year", "soy", "r_err", "a"]:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_dmspssj_attrs(self):
        """ Test that DMSP-SSJ attributes are not available in the south"""

        for self.out in ["sc", "x_1", "x_2", "y_1", "y_2"]:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_ampere_attrs(self):
        """ Test that AMPERE attributes are available in the south"""

        for self.out in ['date', 'time', 'x', 'y', 'fom']:
            self.assertTrue(hasattr(self.ocb, self.out))

    def test_load(self):
        """ Ensure that the default options were correctly set in the south
        """
        self.assertGreater(self.ocb.records, 0)
        self.assertEqual(self.ocb.boundary_lat, -74.0)

    def test_first_good(self):
        """ Test to see that we can find the first good point in the south
        """
        self.ocb.rec_ind = -1
        self.ocb.get_next_good_ocb_ind()

        self.assertGreater(self.ocb.rec_ind, -1)
        self.assertLess(self.ocb.rec_ind, self.ocb.records)

    def test_normal_coord_south(self):
        """ Test to see that the normalisation calculation in the south
        """
        self.out = self.ocb.normal_coord(self.lat[1], self.lon[1]/15.0)

        self.assertAlmostEqual(self.out[0], self.ocb_lat[1])
        self.assertAlmostEqual(self.out[1], self.ocb_mlt[1])
        self.assertEqual(self.out[2], self.r_corr)

    def test_normal_coord_south_array(self):
        """ Test the AACGM coordinate conversion in the south with arrays
        """
        self.out = self.ocb.normal_coord(self.lat, self.lon/15.0)

        self.assertTrue(np.all(np.less(abs(self.out[0] - self.ocb_lat), 1.0e-7,
                                       where=~np.isnan(self.out[0]))))
        self.assertTrue(np.all(np.less(abs(self.out[1] - self.ocb_mlt), 1.0e-7,
                                       where=~np.isnan(self.out[1]))))
        self.assertTrue(np.where(np.isnan(self.out[0])) ==
                        np.where(np.isnan(self.ocb_lat)))
        self.assertTrue(np.where(np.isnan(self.out[1])) ==
                        np.where(np.isnan(self.ocb_mlt)))
        self.assertTrue(np.all(self.out[2] == self.r_corr))

    def test_normal_coord_south_geocentric(self):
        """ Test the geocentric normalisation calculation in the south
        """
        self.out = self.ocb.normal_coord(self.lat[0], self.lon[0]/15.0,
                                         coords='geocentric')

        self.assertAlmostEqual(float(self.out[0]), -68.58362251, places=3)
        self.assertAlmostEqual(float(self.out[1]), 20.56981238, places=3)
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_south_geodetic(self):
        """ Test the geodetic normalisation calculation in the south
        """
        self.out = self.ocb.normal_coord(self.lat[0], self.lon[0]/15.0,
                                         coords='geodetic')

        self.assertAlmostEqual(float(self.out[0]), -68.53149555, places=3)
        self.assertAlmostEqual(float(self.out[1]), 20.57270224, places=3)
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_south_corrected(self):
        """ Test the normalisation calculation in the south with a corrected OCB
        """
        self.r_corr = 1.0
        self.ocb.rfunc_kwargs[self.ocb.rec_ind]['r_add'] = self.r_corr
        self.out = self.ocb.normal_coord(self.lat[0], self.lon[0]/15.0)

        self.assertAlmostEqual(float(self.out[0]), -87.0909090909091, places=3)
        self.assertAlmostEqual(float(self.out[1]), 6.0, places=3)
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_revert_coord_south(self):
        """ Test the reversion to AACGM coordinates in the south
        """
        self.out = self.ocb.revert_coord(self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr)
        self.assertAlmostEqual(float(self.out[0]), self.lat[1])
        self.assertAlmostEqual(float(self.out[1]), self.lon[1]/15.0)

    def test_revert_coord_south_array(self):
        """ Test the reversion to AACGM coordinates in the south with arrays
        """
        self.out = self.ocb.revert_coord(self.ocb_lat, self.ocb_mlt,
                                         self.r_corr)

        self.assertTrue(np.all(np.less(abs(self.out[0] - self.lat), 1.0e-7,
                                       where=~np.isnan(self.out[0])) |
                               np.isnan(self.out[0])))
        self.assertTrue(np.all(np.less(abs(self.out[1] - self.lon/15.0),
                                       1.0e-7,
                                       where=(~np.isnan(self.out[1])
                                              & (self.lat > -90.0))) |
                               np.isnan(self.out[0]) | (self.lat <= -90.0)))
        self.assertTrue(np.where(np.isnan(self.out[0]))
                        == np.where(np.isnan(self.ocb_lat)))
        self.assertTrue(np.where(np.isnan(self.out[1]))
                        == np.where(np.isnan(self.ocb_mlt)))

    def test_revert_coord_south_coord_label(self):
        """ Test the reversion to AACGM coordinates in the south with Mag label
        """
        self.out = self.ocb.revert_coord(self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr, coords='MAG')
        self.assertAlmostEqual(float(self.out[0]), self.lat[1])
        self.assertAlmostEqual(float(self.out[1]), self.lon[1]/15.0)

    def test_revert_coord_south_geodetic(self):
        """ Test the reversion to geodetic coordinates in the south
        """
        self.out = self.ocb.revert_coord(self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr, coords='geodetic')
        self.assertAlmostEqual(float(self.out[0]), -59.17923691, places=3)
        self.assertAlmostEqual(float(self.out[1]), 6.61724772, places=3)

    def test_revert_coord_south_geocentric(self):
        """ Test the reversion to geocentric coordinates in the south
        """
        self.out = self.ocb.revert_coord(self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr, coords='geocentric')
        self.assertAlmostEqual(float(self.out[0]), -59.01868904, places=3)
        self.assertAlmostEqual(float(self.out[1]), 6.61724772, places=3)

    def test_revert_coord_south_w_north(self):
        """ Test the reversion calculation in the sorth with northern lat
        """
        self.out = self.ocb.revert_coord(-self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr)
        self.assertEqual(len(self.out), 2)
        self.assertTrue(np.all(np.isnan(self.out)))

    def test_default_boundary_input(self):
        """ Test to see that the boundary latitude has the correct sign
        """
        self.assertEqual(self.ocb.boundary_lat, -74.0)

    def test_mismatched_boundary_input(self):
        """ Test to see that the boundary latitude has the correct sign
        """
        self.set_south['hemisphere'] = 1
        self.ocb = ocbpy.ocboundary.OCBoundary(**self.set_south)
        self.assertEqual(self.ocb.boundary_lat, 74.0)

    def test_aacgm_boundary_location_good_south(self):
        """ Test finding the OCB in AACGM coordinates in the south
        """
        rind = 8

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(aacgm_lon=self.lon, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(np.all(self.ocb.aacgm_boundary_lat[rind] < 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].min(),
                               -81.92122960532046)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].max(),
                               -78.11700354013985)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmax(), 4)

    def test_aacgm_boundary_location_good_south_corrected_func_arr(self):
        """ Test func array init with good, southern, corrected OCB
        """
        rind = 8
        self.set_south['rfunc'] = np.full(
            shape=self.ocb.r.shape, fill_value=ocbpy.ocb_correction.circular)
        self.set_south['rfunc_kwargs'] = np.full(shape=self.ocb.r.shape,
                                                 fill_value={"r_add": 1.0})
        self.ocb = ocbpy.ocboundary.OCBoundary(**self.set_south)

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(aacgm_lon=self.lon, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(np.all(self.ocb.aacgm_boundary_lat[rind] < 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].min(),
                               -80.91948884759928)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].max(),
                               -77.11526278241867)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmax(), 4)

        del rind

    def test_aacgm_boundary_location_good_south_corrected_kwarg_arr(self):
        """ Test kwarg array init with good, southern, corrected OCB
        """
        rind = 8
        self.set_south['rfunc_kwargs'] = np.full(shape=self.ocb.r.shape,
                                                 fill_value={"r_add": 1.0})
        self.ocb = ocbpy.ocboundary.OCBoundary(**self.set_south)

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(aacgm_lon=self.lon, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(np.all(self.ocb.aacgm_boundary_lat[rind] < 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].min(),
                               -80.91948884759928)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].max(),
                               -77.11526278241867)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmax(), 4)

        del rind

    def test_aacgm_boundary_location_good_south_corrected_dict(self):
        """ Test dict init with good, southern, corrected OCB
        """
        rind = 8
        self.set_south['rfunc_kwargs'] = {"r_add": 1.0}
        self.ocb = ocbpy.ocboundary.OCBoundary(**self.set_south)

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(aacgm_lon=self.lon, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(np.all(self.ocb.aacgm_boundary_lat[rind] < 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].min(),
                               -80.91948884759928)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].max(),
                               -77.11526278241867)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmax(), 4)

    def test_aacgm_boundary_location_good_south_corrected(self):
        """ Test finding the corrected OCB in AACGM coordinates in the south
        """
        self.out = self.ocb.rec_ind
        self.ocb.rfunc_kwargs[self.ocb.rec_ind]['r_add'] = 1.0

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(aacgm_lon=self.lon, rec_ind=self.out)

        # Test value of latitude attribute
        self.assertTrue(np.all(self.ocb.aacgm_boundary_lat[self.out] < 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[self.out].min(),
                               -80.91948884759928)
        self.assertEqual(self.ocb.aacgm_boundary_lat[self.out].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[self.out].max(),
                               -77.11526278241867)
        self.assertEqual(self.ocb.aacgm_boundary_lat[self.out].argmax(), 4)

    def test_aacgm_boundary_location_partial_fill(self):
        """ Test the partial filling when some indices are specified
        """
        self.out = 8
        self.test_aacgm_boundary_location_good_south()

        for i in range(self.ocb.records):
            if i != self.out:
                self.assertTrue(self.ocb.aacgm_boundary_lat[i] is None)
                self.assertTrue(self.ocb.aacgm_boundary_lon[i] is None)
            else:
                self.assertEqual(self.ocb.aacgm_boundary_lat[i].shape,
                                 self.ocb.aacgm_boundary_lon[i].shape)
                self.assertEqual(self.ocb.aacgm_boundary_lon[i].shape,
                                 self.lon.shape)


class TestOCBoundaryMatchData(unittest.TestCase):
    def setUp(self):
        """ Initialize the OCBoundary object using the test file
        """
        set_north = {"filename": path.join(path.dirname(ocbpy.__file__),
                                           "tests", "test_data",
                                           "test_north_circle"),
                     "instrument": "image"}
        self.ocb = ocbpy.ocboundary.OCBoundary(**set_north)
        self.ocb.rec_ind = -1
        self.idat = 0

        # Initialize logging
        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)
        del set_north

    def tearDown(self):
        del self.ocb, self.lwarn, self.lout, self.log_capture, self.idat

    def test_match(self):
        """ Test to see that the data matching works properly
        """
        # Build a array of times for a test dataset
        self.ocb.rec_ind = 27
        test_times = np.arange(self.ocb.dtime[self.ocb.rec_ind],
                               self.ocb.dtime[self.ocb.rec_ind + 5],
                               dt.timedelta(seconds=600)).astype(dt.datetime)

        # Because the array starts at the first good OCB, will return zero
        self.idat = ocbpy.ocboundary.match_data_ocb(self.ocb, test_times,
                                                    idat=self.idat)
        self.assertEqual(self.idat, 0)
        self.assertEqual(self.ocb.rec_ind, 27)

        # The next test time will cause the OCB to cycle forward to a new
        # record
        idat = ocbpy.ocboundary.match_data_ocb(self.ocb, test_times, idat=1)
        self.assertEqual(idat, 1)
        self.assertEqual(self.ocb.rec_ind, 31)
        self.assertLess(
            abs((test_times[idat]
                 - self.ocb.dtime[self.ocb.rec_ind]).total_seconds()), 600.0)
        del test_times

    def test_good_first_match(self):
        """ Test ability to find the first good OCB
        """
        # Change the logging level
        ocbpy.logger.setLevel(logging.INFO)

        # Because the array starts at the first good OCB, will return zero
        self.idat = ocbpy.ocboundary.match_data_ocb(self.ocb,
                                                    [self.ocb.dtime[27]],
                                                    idat=self.idat)
        self.assertEqual(self.idat, 0)
        self.assertEqual(self.ocb.rec_ind, 27)

        # The first match will be announced in the log
        self.lwarn = u"found first good OCB record at"
        self.lout = self.log_capture.getvalue()
        self.assertTrue(self.lout.find(self.lwarn) >= 0)

    def test_bad_first_match(self):
        """ Test ability to not find a good OCB
        """
        # Set requirements for good OCB so high that none will pass
        self.idat = ocbpy.ocboundary.match_data_ocb(self.ocb,
                                                    [self.ocb.dtime[27]],
                                                    idat=self.idat,
                                                    min_sectors=24)
        self.assertEqual(self.idat, 0)
        self.assertGreaterEqual(self.ocb.rec_ind, self.ocb.records)

        # The first match will be announced in the log
        self.lwarn = u"unable to find a good OCB record"
        self.lout = self.log_capture.getvalue()
        self.assertTrue(self.lout.find(self.lwarn) >= 0)

    def test_bad_ocb_ind(self):
        """ Test ability to exit if ocb record counter is too high
        """
        # Set the OCB record index to the end
        self.ocb.rec_ind = self.ocb.records
        self.idat = ocbpy.ocboundary.match_data_ocb(self.ocb,
                                                    [self.ocb.dtime[27]],
                                                    idat=-1)
        self.assertEqual(self.idat, -1)
        self.assertGreaterEqual(self.ocb.rec_ind, self.ocb.records)

    def test_bad_dat_ind(self):
        """ Test ability to exit if data record counter is too high
        """
        # Set the OCB record index to the end
        self.idat = ocbpy.ocboundary.match_data_ocb(self.ocb,
                                                    [self.ocb.dtime[27]],
                                                    idat=2)
        self.assertEqual(self.idat, 2)
        self.assertGreaterEqual(self.ocb.rec_ind, -1)

    def test_bad_first_data_time(self):
        """ Test ability to cycle past data times not close enough to match
        """
        # Set the OCB record index to the beginning and match
        self.idat = ocbpy.ocboundary.match_data_ocb(self.ocb,
                                                    [self.ocb.dtime[27]
                                                     - dt.timedelta(days=1),
                                                     self.ocb.dtime[27]],
                                                    idat=self.idat)
        self.assertEqual(self.idat, 1)
        self.assertEqual(self.ocb.rec_ind, 27)

    def test_data_all_before_first_ocb_record(self):
        """ Test failure when data occurs before boundaries"""
        # Change the logging level
        ocbpy.logger.setLevel(logging.ERROR)

        # Set the OCB record index to the beginning and match
        self.idat = ocbpy.ocboundary.match_data_ocb(self.ocb,
                                                    [self.ocb.dtime[27]
                                                     - dt.timedelta(days=1)],
                                                    idat=self.idat)
        self.assertIsNone(self.idat)
        self.assertGreaterEqual(self.ocb.rec_ind, 27)

        # Check the log output
        self.lwarn = u"no input data close enough to first record"
        self.lout = self.log_capture.getvalue()
        self.assertTrue(self.lout.find(self.lwarn) >= 0)

    def test_late_data_time_alignment(self):
        """ Test failure when data occurs after boundaries"""
        # Change the logging level
        ocbpy.logger.setLevel(logging.INFO)

        # Match OCB with data that occurs after the boundaries end
        self.idat = ocbpy.ocboundary.match_data_ocb(
            self.ocb, [self.ocb.dtime[self.ocb.records-1]
                       + dt.timedelta(days=2)], idat=self.idat)
        self.assertEqual(self.idat, 0)
        self.assertGreaterEqual(self.ocb.rec_ind, self.ocb.records)

        # Check the log output
        self.lwarn = u"no OCB data available within"
        self.lout = self.log_capture.getvalue()
        self.assertTrue(self.lout.find(self.lwarn) >= 0)
        self.lwarn = u"of first measurement"
        self.assertTrue(self.lout.find(self.lwarn) > 0)

    def test_no_data_time_alignment(self):
        """ Test failure when data occurs between boundaries """
        # Change the logging level
        ocbpy.logger.setLevel(logging.INFO)

        # Match OCBs with misaligned input data
        self.idat = ocbpy.ocboundary.match_data_ocb(
            self.ocb, [self.ocb.dtime[37] - dt.timedelta(seconds=601)],
            idat=self.idat)
        self.assertEqual(self.idat, 1)
        self.assertGreaterEqual(self.ocb.rec_ind, 37)

        # Check the log output
        self.lwarn = u"no OCB data available within"
        self.lout = self.log_capture.getvalue()
        self.assertTrue(self.lout.find(self.lwarn) >= 0)
        self.lwarn = u"of input measurement"
        self.assertTrue(self.lout.find(self.lwarn) >= 0)


class TestOCBoundaryFailure(unittest.TestCase):
    def setUp(self):
        """ Initialize the OCBoundary object using the test file
        """
        if version_info.major == 2:
            self.assertRaisesRegex = self.assertRaisesRegexp

    def tearDown(self):
        pass

    def test_bad_instrument_input(self):
        """ Test failure when bad instrument value is input"""

        test_north = path.join(path.dirname(ocbpy.__file__), "tests",
                               "test_data", "test_north_circle")
        self.assertTrue(path.isfile(test_north))
        with self.assertRaisesRegex(ValueError, "unknown instrument"):
            ocbpy.ocboundary.OCBoundary(instrument="hi", filename=test_north)

        del test_north

    def test_bad_hemisphere_input(self):
        """ Test failure when incorrect hemisphere value is input"""
        with self.assertRaisesRegex(ValueError, "hemisphere must be 1"):
            ocbpy.ocboundary.OCBoundary(hemisphere=0)

    def test_bad_shape_rfunc_input(self):
        """ Test failure when badly shaped radial correction function"""
        with self.assertRaisesRegex(ValueError,
                                    "Misshaped correction function array"):
            ocbpy.ocboundary.OCBoundary(
                rfunc=np.array([ocbpy.ocb_correction.circular]))

    def test_bad_shape_rfunc_kwarg_input(self):
        """ Test failure when badly shaped radial correction function kwargs"""
        with self.assertRaisesRegex(ValueError,
                                    "Misshaped correction function keyword"):
            ocbpy.ocboundary.OCBoundary(rfunc_kwargs=np.array([{}]))

    def test_bad_rfunc_input(self):
        """ Test failure with bad radial correction function input"""
        with self.assertRaisesRegex(
                ValueError, "Unknown input type for correction function"):
            ocbpy.ocboundary.OCBoundary(rfunc="rfunc")

    def test_bad_rfunc_kwarg_input(self):
        """ Test failure with bad radial correction function kwarg input"""
        with self.assertRaisesRegex(
                ValueError, "Unknown input type for correction keywords"):
            ocbpy.ocboundary.OCBoundary(rfunc_kwargs="rfunc")
