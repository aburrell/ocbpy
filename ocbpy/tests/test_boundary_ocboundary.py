#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the boundary OCBoundary class."""

import datetime as dt
from io import StringIO
import logging
import numpy
from os import path
import unittest

import ocbpy


class TestOCBoundaryLogFailure(unittest.TestCase):
    """Test the logging messages raised by the OCBoundary class."""

    def setUp(self):
        """Initialize the test class."""
        self.test_class = ocbpy.OCBoundary
        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

    def tearDown(self):
        """Tear down the test case."""
        del self.lwarn, self.lout, self.log_capture, self.test_class

    def test_bad_instrument_name(self):
        """Test OCB initialization with bad instrument name."""
        self.lwarn = u"OCB instrument must be a string"

        # Initialize the OCBoundary class with bad instrument names
        for val in [1, None, True]:
            with self.subTest(val=val):
                ocb = self.test_class(instrument=val)
                self.assertIsNone(ocb.filename)
                self.assertIsNone(ocb.instrument)

                self.lout = self.log_capture.getvalue()
                # Test logging error message for each bad initialization
                self.assertRegex(self.lout.find, self.lwarn)

        del val, ocb

    def test_bad_file_name(self):
        """Test OCB initialization with bad file name."""
        self.lwarn = "filename is not a string"

        # Initialize the OCBoundary class with bad instrument names
        for val in [1, True]:
            with self.subTest(val=val):
                ocb = self.test_class(filename=val)
                self.assertIsNone(ocb.filename)

                self.lout = self.log_capture.getvalue()
                # Test logging error message for each bad initialization
                self.assertRegex(self.lout, self.lwarn)

        del val, ocb

    def test_bad_filename(self):
        """Test initialization with a bad default file/instrument pairing."""
        self.lwarn = "name provided is not a file\ncannot open OCB file [hi]"

        # Try to load AMPERE data with an IMAGE file
        ocb = self.test_class(filename="hi")
        self.assertIsNone(ocb.filename)

        self.lout = self.log_capture.getvalue()

        # Test logging error message for each bad initialization
        self.assertRegex(self.lout, self.lwarn)

        del ocb

    def test_bad_time_structure(self):
        """Test initialization without complete time data in file."""
        self.lwarn = u"missing time columns in"

        # Initialize without a file so that custom loading is performed
        ocb = self.test_class(filename=None)
        self.assertIsNone(ocb.filename)

        # Set the filename
        ocb.filename = path.join(path.dirname(ocbpy.__file__), "tests",
                                 "test_data", "test_north_circle")
        self.assertTrue(path.isfile(ocb.filename))

        # Load the data, skipping the year
        ocb.load(ocb_cols="skip soy num_sectors phi_cent r_cent r a r_err")

        self.lout = self.log_capture.getvalue()
        # Test logging error message for the non-None bad initializations
        self.assertRegex(self.lout, self.lwarn)

        del ocb


class TestOCBoundaryInstruments(unittest.TestCase):
    def setUp(self):
        """Initialize the instrument information."""
        self.test_class = ocbpy.OCBoundary
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
        """Clean up the test environment."""
        del self.test_dir, self.inst_attrs, self.inst_init, self.ocb
        del self.test_class

    def test_instrument_loading(self):
        """Test OCB initialization with good instrument names."""
        for ocb_kwargs in self.inst_init:
            with self.subTest(ocb_kwargs=ocb_kwargs):
                self.ocb = self.test_class(**ocb_kwargs)

                for tattr in self.inst_attrs[ocb_kwargs['instrument']]:
                    self.assertTrue(hasattr(self.ocb, tattr))

                for tattr in self.not_attrs[ocb_kwargs['instrument']]:
                    self.assertFalse(hasattr(self.ocb, tattr))

        del ocb_kwargs, tattr


class TestOCBoundaryMethodsGeneral(unittest.TestCase):
    def setUp(self):
        """Initialize the test environment."""
        self.test_class = ocbpy.OCBoundary
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

    def tearDown(self):
        """Clean up the test environment."""
        del self.set_empty, self.set_default, self.ocb, self.test_class

    def test_repr_string(self):
        """Test __repr__ method string."""
        # Initalize the class object
        self.ocb = self.test_class(**self.set_default)

        # Set the different types of correction function attributes
        rfuncs = [self.ocb.rfunc, None,
                  numpy.array([ocbpy.ocb_correction.circular if i == 0 else
                               ocbpy.ocb_correction.elliptical
                               for i, val in enumerate(self.ocb.dtime)])]
        rfunc_kwargs = [self.ocb.rfunc_kwargs, None,
                        numpy.array([{'r_add': 1} if i == 0 else
                                     {'method': 'gaussian'}
                                     for i, val in enumerate(self.ocb.dtime)])]

        # Test each type of correction function attribute
        for i, val in enumerate(rfuncs):
            with self.subTest(val=val, i=i):
                self.ocb.rfunc = val
                self.ocb.rfunc_kwargs = rfunc_kwargs[i]
                self.assertRegex(repr(self.ocb), "ocbpy.OCBoundary")
                if val is None:
                    self.assertRegex(repr(self.ocb), repr(val))
                else:
                    self.assertRegex(repr(self.ocb), val[0].__name__)
                    if i == 2:
                        self.assertRegex(repr(self.ocb), 'numpy.array')
                        self.assertRegex(repr(self.ocb), 'dtype=object')

    def test_repr_eval(self):
        """Test __repr__ method's ability to reproduce a class."""
        test_ocb = eval(repr(self.ocb))
        self.assertEqual(repr(self.ocb), repr(test_ocb))

    def test_default_str(self):
        """Test the default class print output."""
        self.ocb = self.test_class(**self.set_default)
        self.assertRegex(self.ocb.__str__(), "Open-Closed Boundary file:")

    def test_short_str(self):
        """Test the default class print output."""
        self.ocb = self.test_class(**self.set_default)
        self.ocb.records = 1

        self.assertRegex(self.ocb.__str__(), "1 records from")

    def test_bad_rfunc_inst(self):
        """Test failure setting default rfunc for unknown instrument."""

        with self.assertRaisesRegex(ValueError, "unknown instrument"):
            self.set_empty['instrument'] = 'bad'
            self.set_empty['rfunc'] = None
            self.ocb = self.test_class(**self.set_empty)

    def test_no_file_str(self):
        """Test the unset class print output."""

        self.ocb = self.test_class(filename=None)
        self.assertRegex(self.ocb.__str__(),
                         "No Open-Closed Boundary file specified")

    def test_empty_file_str(self):
        """Test the class print output with an empty data file."""

        self.ocb = self.test_class(**self.set_empty)
        self.assertRegex(self.ocb.__str__(), "No data loaded")

    def test_nofile_init(self):
        """Ensure that the class can be initialised without loading a file."""
        self.ocb = self.test_class(filename=None)

        self.assertIsNone(self.ocb.filename)
        self.assertIsNone(self.ocb.dtime)
        self.assertEqual(self.ocb.records, 0)


class TestOCBoundaryMethodsNorth(unittest.TestCase):
    """Unit tests for the OCBoundary class in the northern hemisphere."""
    
    def setUp(self):
        """Initialize the test environment."""
        self.test_class = ocbpy.OCBoundary
        self.set_north = {'filename': path.join(path.dirname(ocbpy.__file__),
                                                "tests", "test_data",
                                                "test_north_circle"),
                          'instrument': 'image'}
        self.assertTrue(path.isfile(self.set_north['filename']))
        self.ocb = self.test_class(**self.set_north)
        self.ocb.rec_ind = 27

        self.mlt = numpy.linspace(0.0, 24.0, num=6)
        self.lat = numpy.linspace(0.0, 90.0, num=len(self.mlt))
        self.ocb_lat = [numpy.nan, 11.25588586, 30.35153908, 47.0979063,
                        66.59889231, 86.86586231]
        self.ocb_mlt = [numpy.nan, 4.75942194, 9.76745427, 14.61843964,
                        19.02060793, 17.832]
        self.r_corr = 0.0
        self.out = None

    def tearDown(self):
        """Clean up the test environment."""
        del self.ocb, self.set_north, self.mlt, self.lat, self.ocb_lat
        del self.ocb_mlt, self.r_corr, self.out, self.test_class

    def test_attrs(self):
        """Test the default attributes in the north."""
        for self.out in ["filename", "instrument", "hemisphere", "records",
                         "rec_ind", "dtime", "phi_cent", "r_cent", "r",
                         "boundary_lat"]:
            self.assertTrue(hasattr(self.ocb, self.out))

        # Ensure optional attributes are absent
        for self.out in ["aacgm_boundary_lon", "aacgm_boundary_lat"]:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_image_attrs(self):
        """Test IMAGE attributes in the north."""
        for self.out in ["num_sectors", "year", "soy", "r_err", "a"]:
            self.assertTrue(hasattr(self.ocb, self.out))

    def test_ampere_attrs(self):
        """Test AMPERE attributes don't exist when IMAGE is loaded."""
        for self.out in ['date', 'time', 'x', 'y', 'fom']:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_dmspssj_attrs(self):
        """Test DMSP-SSJ attributes don't exist when IMAGE is loaded."""

        for self.out in ['sc', 'date', 'time', 'x', 'y', 'fom', 'x_1', 'x_2',
                         'y_1', 'y_2']:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_load(self):
        """Ensure correctly loaded defaults in the north."""
        self.assertGreater(self.ocb.records, 0)
        self.assertEqual(self.ocb.boundary_lat, 74.0)

    def test_partial_load(self):
        """Ensure limited sections of a file can be loaded in the north."""

        stime = self.ocb.dtime[0] + dt.timedelta(seconds=1)
        etime = self.ocb.dtime[-1] - dt.timedelta(seconds=1)

        # Load all but the first and last records
        self.out = self.test_class(filename=self.ocb.filename,
                                               instrument=self.ocb.instrument,
                                               stime=stime, etime=etime,
                                               boundary_lat=75.0)

        self.assertEqual(self.ocb.records, self.out.records + 2)
        self.assertEqual(self.out.boundary_lat, 75.0)

    def test_first_good(self):
        """Test to see that we can find the first good point in the north."""
        self.ocb.rec_ind = -1

        self.ocb.get_next_good_ocb_ind()

        self.assertGreater(self.ocb.rec_ind, -1)
        self.assertLess(self.ocb.rec_ind, self.ocb.records)

    def test_normal_coord_north_float(self):
        """Test the normalisation calculation in the north."""
        self.out = self.ocb.normal_coord(self.lat[-1], self.mlt[-1])
        self.assertAlmostEqual(float(self.out[0]), self.ocb_lat[-1])
        self.assertAlmostEqual(float(self.out[1]), self.ocb_mlt[-1])
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_north_array(self):
        """Test normalisation calculation in the north with arryay input."""
        self.out = self.ocb.normal_coord(self.lat, self.mlt)

        self.assertTrue(numpy.all(numpy.less(abs(self.out[0] - self.ocb_lat),
                                             1.0e-7,
                                             where=~numpy.isnan(self.out[0]))
                                  | numpy.isnan(self.out[0])))
        self.assertTrue(numpy.all(numpy.less(abs(self.out[1] - self.ocb_mlt),
                                             1.0e-7,
                                             where=(~numpy.isnan(self.out[1])))
                                  | numpy.isnan(self.out[1])))
        self.assertTrue(numpy.where(numpy.isnan(self.out[0]))
                        == numpy.where(numpy.isnan(self.ocb_lat)))
        self.assertTrue(numpy.where(numpy.isnan(self.out[1]))
                        == numpy.where(numpy.isnan(self.ocb_mlt)))
        self.assertTrue(numpy.all(self.out[2] == self.r_corr))

    def test_normal_coord_north_alt_mag_label(self):
        """Test normalisation calculation with good, but odd coord label."""
        self.out = self.ocb.normal_coord(self.lat[-1], self.mlt[-1],
                                         coords='Mag')
        self.assertAlmostEqual(float(self.out[0]), self.ocb_lat[-1])
        self.assertAlmostEqual(float(self.out[1]), self.ocb_mlt[-1])
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_north_geodetic(self):
        """Test the geodetic normalisation calculation in the north."""
        self.out = self.ocb.normal_coord(self.lat[-1], self.mlt[-1],
                                         coords='geodetic')
        self.assertAlmostEqual(float(self.out[0]), 79.2631, places=3)
        self.assertAlmostEqual(float(self.out[1]), 19.3839, places=3)
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_north_geocentric(self):
        """Test the geocentric normalisation calculation in the north."""
        self.out = self.ocb.normal_coord(self.lat[-1], self.mlt[-1],
                                         coords='geocentric')
        self.assertAlmostEqual(float(self.out[0]), 79.2654, places=3)
        self.assertAlmostEqual(float(self.out[1]), 19.3852, places=3)
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_north_w_south(self):
        """Test normalisation calculation in the north with southern lat."""
        self.out = self.ocb.normal_coord(-self.lat[-1], self.mlt[-1])

        self.assertEqual(len(self.out), 3)
        self.assertTrue(numpy.all(numpy.isnan(self.out)))

    def test_normal_coord_low_rec_ind(self):
        """Test the normalization calculation failure with low record index."""
        self.ocb.rec_ind = -1
        self.out = self.ocb.normal_coord(self.lat[-1], self.mlt[-1])

        self.assertEqual(len(self.out), 3)
        self.assertTrue(numpy.all(numpy.isnan(self.out)))

    def test_normal_coord_high_rec_ind(self):
        """Test normalization calculation failure with high record index."""
        self.ocb.rec_ind = self.ocb.records + 1
        self.out = self.ocb.normal_coord(self.lat[-1], self.mlt[-1])

        self.assertEqual(len(self.out), 3)
        self.assertTrue(numpy.all(numpy.isnan(self.out)))

    def test_revert_coord_north_float(self):
        """Test the reversion to AACGM coordinates in the north."""
        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr)
        self.assertAlmostEqual(self.out[0], self.lat[-2])
        self.assertAlmostEqual(self.out[1], self.mlt[-2])

    def test_revert_coord_north_array(self):
        """Test reversion to AACGM coordinates in the north for an array."""
        self.out = self.ocb.revert_coord(self.ocb_lat, self.ocb_mlt,
                                         self.r_corr)

        self.assertTrue(numpy.all(numpy.less(abs(self.out[0] - self.lat),
                                             1.0e-7,
                                             where=~numpy.isnan(self.out[0]))
                                  | (numpy.isnan(self.out[0]))))
        self.assertTrue(numpy.all(numpy.less(abs(self.out[1] - self.mlt),
                                             1.0e-7,
                                             where=(~numpy.isnan(self.out[1])
                                                    & (self.lat < 90.0)))
                                  | numpy.isnan(self.out[0])
                                  | (self.lat >= 90.0)))
        self.assertTrue(numpy.where(numpy.isnan(self.out[0]))
                        == numpy.where(numpy.isnan(self.ocb_lat)))
        self.assertTrue(numpy.where(numpy.isnan(self.out[1]))
                        == numpy.where(numpy.isnan(self.ocb_mlt)))

    def test_revert_coord_north_coord_label(self):
        """Test reversion to AACGM coordinates in the north with Mag label."""
        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr, coords='MAG')
        self.assertAlmostEqual(self.out[0], self.lat[-2])
        self.assertAlmostEqual(self.out[1], self.mlt[-2])

    def test_revert_coord_north_geodetic(self):
        """Test the reversion to geodetic coordinates in the north."""
        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr, coords='geodetic')
        self.assertAlmostEqual(float(self.out[0]), 77.13321838, places=3)
        self.assertAlmostEqual(float(self.out[1]), 19.18124285, places=3)

    def test_revert_coord_north_geocentric(self):
        """Test the reversion to geocentric coordinates in the north."""
        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr, coords='geocentric')
        self.assertAlmostEqual(float(self.out[0]), 77.05394766, places=3)
        self.assertAlmostEqual(float(self.out[1]), 19.18124285, places=3)

    def test_revert_coord_north_w_south(self):
        """Test the reversion calculation in the north with southern lat."""
        self.out = self.ocb.revert_coord(-self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr)
        self.assertEqual(len(self.out), 2)
        self.assertTrue(numpy.all(numpy.isnan(self.out)))

    def test_revert_coord_low_rec_ind(self):
        """Test the reversion calculation failure with low record index
        """
        self.ocb.rec_ind = -1

        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr)
        self.assertEqual(len(self.out), 2)
        self.assertTrue(numpy.all(numpy.isnan(self.out)))

    def test_revert_coord_high_rec_ind(self):
        """Test the reversion calculation failure with high record index."""
        self.ocb.rec_ind = self.ocb.records + 1

        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr)
        self.assertEqual(len(self.out), 2)
        self.assertTrue(numpy.all(numpy.isnan(self.out)))

    def test_default_boundary_input(self):
        """Test to see that the boundary latitude has the correct sign."""
        self.assertEqual(self.ocb.boundary_lat, 74.0)

    def test_mismatched_boundary_input(self):
        """Test to see that the boundary latitude has the incorrect sign."""
        self.set_north['hemisphere'] = -1
        self.out = self.test_class(**self.set_north)
        self.assertEqual(self.out.boundary_lat, -74.0)

    def test_mismatched_boundary_input_correction(self):
        """Test to see that the boundary latitude corrects the sign."""
        self.set_north['boundary_lat'] = -70.0
        self.out = self.test_class(**self.set_north)
        self.assertEqual(self.out.boundary_lat, 70.0)

    def test_aacgm_boundary_location(self):
        """Test the calculation of the OCB in AACGM coordinates in the north."""
        # Add new attributes
        self.ocb.get_aacgm_boundary_lat(self.mlt)

        # Ensure new attriutes were added
        self.assertTrue(hasattr(self.ocb, "aacgm_boundary_lon"))
        self.assertTrue(hasattr(self.ocb, "aacgm_boundary_lat"))

        # Test shape of new attributes
        self.assertEqual(len(self.ocb.aacgm_boundary_mlt), self.ocb.records)
        self.assertEqual(len(self.ocb.aacgm_boundary_mlt[0]), len(self.mlt))
        self.assertEqual(len(self.ocb.aacgm_boundary_lat[0]), len(self.mlt))
        self.assertEqual(len(self.ocb.aacgm_boundary_lon[0]), len(self.mlt))

        # Test value of longitude attribute
        self.assertEqual(sum(self.mlt[:-1]
                             - self.ocb.aacgm_boundary_mlt[0][:-1]), 0)
        self.assertEqual(sum(self.mlt[:-1]
                             - self.ocb.aacgm_boundary_mlt[-1][:-1]), 0)

        # Test the value of the latitude attriubte at the good record location
        # Also tests that specifying the same longitude locations twice is ok
        self.test_aacgm_boundary_location_good()

    def test_aacgm_boundary_location_good(self):
        """Test calculation of the OCB in AACGM coordinates in the north."""
        rind = 27

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(numpy.all(self.ocb.aacgm_boundary_lat[rind] > 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].min(),
                               73.26939247752293)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].max(),
                               78.52813223696786)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmax(), 4)

    def test_aacgm_boundary_location_bad(self):
        """Test calclation of the OCB in AACGM coordinates for limited MLTs."""
        rind = 2

        # Add the attriubte at the bad location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=rind)

        # Test value of latitude attribute
        self.assertFalse(numpy.all(self.ocb.aacgm_boundary_lat[rind] > 0.0))
        self.assertTrue(numpy.any(self.ocb.aacgm_boundary_lat[rind] > 0.0))
        self.assertTrue(numpy.any(numpy.isnan(
            self.ocb.aacgm_boundary_lat[rind])))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind][1],
                               72.82502115387997)

    def test_aacgm_boundary_location_no_input(self):
        """Test failure of OCB AACGM location calculation for no input."""
        with self.assertRaises(TypeError):
            self.ocb.get_aacgm_boundary_lat()

    def test_aacgm_boundary_location_no_overwrite(self):
        """Ensure no overwrite when re-calculating OCB AACGM locations."""
        log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        # Initialize the attributes with values for the good location
        rind = 27
        self.test_aacgm_boundary_location_good()

        # This should not raise a warning
        self.ocb.get_aacgm_boundary_lat(150.0, rec_ind=rind - 1)

        # This should raise a warning
        self.ocb.get_aacgm_boundary_lat(150.0, rec_ind=rind)

        self.out = log_capture.getvalue()
        # Test logging error message for only one warning about boundary update
        self.assertRegex(self.out, "unable to update AACGM boundary")

        del log_capture

    def test_aacgm_boundary_location_overwrite(self):
        """Test ability to overwrite OCB AACGM location."""

        # Initialize the attributes with values for the good location
        self.test_aacgm_boundary_location_good()

        # Specify a new longitude for that location
        rind = 27
        self.ocb.get_aacgm_boundary_lat(10.0, rec_ind=rind, overwrite=True)

        # Test value of latitude attribute
        self.assertFalse(hasattr(self.ocb.aacgm_boundary_lat[rind], "shape"))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind],
                               74.8508209365)

    def test_aacgm_boundary_location_mlt_range(self):
        """Test failure of OCB AACGM location with different valued MLT."""
        self.mlt[self.mlt > 12.0] -= 24.0
        self.ocb.get_aacgm_boundary_lat(self.mlt)

        # Test the attributes with values for the good location
        self.test_aacgm_boundary_location_good()


class TestOCBoundaryMethodsSouth(unittest.TestCase):
    """Unit tests for the OCBoundary methods in the southern hemisphere."""

    def setUp(self):
        """Initialize the test environment."""
        self.test_class = ocbpy.OCBoundary

        self.set_south = {"filename": path.join(path.dirname(ocbpy.__file__),
                                                "tests", "test_data",
                                                "test_south_circle"),
                          "instrument": "ampere",
                          "hemisphere": -1,
                          "rfunc": ocbpy.ocb_correction.circular}
        self.ocb = self.test_class(**self.set_south)
        self.ocb.rec_ind = 8

        self.mlt = numpy.linspace(0.0, 24.0, num=6)
        self.lat = numpy.linspace(-90.0, 0.0, num=len(self.mlt))
        self.ocb_lat = [-86.8, -58.14126906, -30.46277504, -5.44127327,
                        22.16097829, numpy.nan]
        self.ocb_mlt = [6.0, 4.91857824, 9.43385497, 14.28303702, 19.23367655,
                        numpy.nan]
        self.r_corr = 0.0
        self.out = None

    def tearDown(self):
        """Clean up the test environment."""
        del self.ocb, self.set_south, self.mlt, self.lat, self.ocb_lat
        del self.ocb_mlt, self.r_corr, self.out, self.test_class

    def test_attrs(self):
        """Test the default attributes in the south."""

        for self.out in ["filename", "instrument", "hemisphere", "records",
                         "rec_ind", "dtime", "phi_cent", "r_cent", "r",
                         "boundary_lat"]:
            self.assertTrue(hasattr(self.ocb, self.out))

        # Ensure optional attributes are absent
        for self.out in ["aacgm_boundary_lon", "aacgm_boundary_lat"]:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_image_attrs(self):
        """Test that IMAGE attributes are not available in the south."""

        for self.out in ["num_sectors", "year", "soy", "r_err", "a"]:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_dmspssj_attrs(self):
        """Test that DMSP-SSJ attributes are not available in the south."""

        for self.out in ["sc", "x_1", "x_2", "y_1", "y_2"]:
            self.assertFalse(hasattr(self.ocb, self.out))

    def test_ampere_attrs(self):
        """Test that AMPERE attributes are available in the south."""

        for self.out in ['date', 'time', 'x', 'y', 'fom']:
            self.assertTrue(hasattr(self.ocb, self.out))

    def test_load(self):
        """Ensure that the default options were correctly set in the south."""
        self.assertGreater(self.ocb.records, 0)
        self.assertEqual(self.ocb.boundary_lat, -74.0)

    def test_first_good(self):
        """Test to see that we can find the first good point in the south."""
        self.ocb.rec_ind = -1
        self.ocb.get_next_good_ocb_ind()

        self.assertGreater(self.ocb.rec_ind, -1)
        self.assertLess(self.ocb.rec_ind, self.ocb.records)

    def test_normal_coord_south(self):
        """Test to see that the normalisation calculation in the south."""
        self.out = self.ocb.normal_coord(self.lat[1], self.mlt[1])

        self.assertAlmostEqual(self.out[0], self.ocb_lat[1])
        self.assertAlmostEqual(self.out[1], self.ocb_mlt[1])
        self.assertEqual(self.out[2], self.r_corr)

    def test_normal_coord_south_array(self):
        """Test the AACGM coordinate conversion in the south with arrays."""
        self.out = self.ocb.normal_coord(self.lat, self.mlt)

        self.assertTrue(numpy.all(numpy.less(abs(self.out[0] - self.ocb_lat),
                                             1.0e-7,
                                             where=~numpy.isnan(self.out[0]))))
        self.assertTrue(numpy.all(numpy.less(abs(self.out[1] - self.ocb_mlt),
                                             1.0e-7,
                                             where=~numpy.isnan(self.out[1]))))
        self.assertTrue(numpy.where(numpy.isnan(self.out[0]))
                        == numpy.where(numpy.isnan(self.ocb_lat)))
        self.assertTrue(numpy.where(numpy.isnan(self.out[1]))
                        == numpy.where(numpy.isnan(self.ocb_mlt)))
        self.assertTrue(numpy.all(self.out[2] == self.r_corr))

    def test_normal_coord_south_geocentric(self):
        """Test the geocentric normalisation calculation in the south."""
        self.out = self.ocb.normal_coord(self.lat[0], self.mlt[0],
                                         coords='geocentric')

        self.assertAlmostEqual(float(self.out[0]), -68.58362251, places=3)
        self.assertAlmostEqual(float(self.out[1]), 20.56981238, places=3)
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_south_geodetic(self):
        """Test the geodetic normalisation calculation in the south."""
        self.out = self.ocb.normal_coord(self.lat[0], self.mlt[0],
                                         coords='geodetic')

        self.assertAlmostEqual(float(self.out[0]), -68.53149555, places=3)
        self.assertAlmostEqual(float(self.out[1]), 20.57270224, places=3)
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_normal_coord_south_corrected(self):
        """Test normalisation calculation in the south with a corrected OCB."""
        self.r_corr = 1.0
        self.ocb.rfunc_kwargs[self.ocb.rec_ind]['r_add'] = self.r_corr
        self.out = self.ocb.normal_coord(self.lat[0], self.mlt[0])

        self.assertAlmostEqual(float(self.out[0]), -87.0909090909091, places=3)
        self.assertAlmostEqual(float(self.out[1]), 6.0, places=3)
        self.assertEqual(float(self.out[2]), self.r_corr)

    def test_revert_coord_south(self):
        """Test the reversion to AACGM coordinates in the south."""
        self.out = self.ocb.revert_coord(self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr)
        self.assertAlmostEqual(float(self.out[0]), self.lat[1])
        self.assertAlmostEqual(float(self.out[1]), self.mlt[1])

    def test_revert_coord_south_array(self):
        """Test the reversion to AACGM coordinates in the south with arrays."""
        self.out = self.ocb.revert_coord(self.ocb_lat, self.ocb_mlt,
                                         self.r_corr)

        self.assertTrue(numpy.all(numpy.less(abs(self.out[0] - self.lat),
                                             1.0e-7,
                                             where=~numpy.isnan(self.out[0]))
                                  | numpy.isnan(self.out[0])))
        self.assertTrue(numpy.all(numpy.less(abs(self.out[1] - self.mlt),
                                             1.0e-7,
                                             where=(~numpy.isnan(self.out[1])
                                                    & (self.lat > -90.0)))
                                  | numpy.isnan(self.out[0])
                                  | (self.lat <= -90.0)))
        self.assertTrue(numpy.where(numpy.isnan(self.out[0]))
                        == numpy.where(numpy.isnan(self.ocb_lat)))
        self.assertTrue(numpy.where(numpy.isnan(self.out[1]))
                        == numpy.where(numpy.isnan(self.ocb_mlt)))

    def test_revert_coord_south_coord_label(self):
        """Test reversion to AACGM coordinates in the south with Mag label."""
        self.out = self.ocb.revert_coord(self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr, coords='MAG')
        self.assertAlmostEqual(float(self.out[0]), self.lat[1])
        self.assertAlmostEqual(float(self.out[1]), self.mlt[1])

    def test_revert_coord_south_geodetic(self):
        """Test the reversion to geodetic coordinates in the south."""
        self.out = self.ocb.revert_coord(self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr, coords='geodetic')
        self.assertAlmostEqual(float(self.out[0]), -59.17923691, places=3)
        self.assertAlmostEqual(float(self.out[1]), 6.61724772, places=3)

    def test_revert_coord_south_geocentric(self):
        """Test the reversion to geocentric coordinates in the south."""
        self.out = self.ocb.revert_coord(self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr, coords='geocentric')
        self.assertAlmostEqual(float(self.out[0]), -59.01868904, places=3)
        self.assertAlmostEqual(float(self.out[1]), 6.61724772, places=3)

    def test_revert_coord_south_w_north(self):
        """Test the reversion calculation in the sorth with northern lat."""
        self.out = self.ocb.revert_coord(-self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr)
        self.assertEqual(len(self.out), 2)
        self.assertTrue(numpy.all(numpy.isnan(self.out)))

    def test_default_boundary_input(self):
        """Test to see that the boundary latitude has the correct sign."""
        self.assertEqual(self.ocb.boundary_lat, -74.0)

    def test_mismatched_boundary_input(self):
        """Test to see that the boundary latitude has the correct sign
        """
        self.set_south['hemisphere'] = 1
        self.ocb = self.test_class(**self.set_south)
        self.assertEqual(self.ocb.boundary_lat, 74.0)

    def test_aacgm_boundary_location_good_south(self):
        """Test finding the OCB in AACGM coordinates in the south."""
        rind = 8

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(numpy.all(self.ocb.aacgm_boundary_lat[rind] < 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].min(),
                               -81.92122960532046)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].max(),
                               -78.11700354013985)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmax(), 4)

    def test_aacgm_boundary_location_good_south_corrected_func_arr(self):
        """Test func array init with good, southern, corrected OCB."""
        rind = 8
        self.set_south['rfunc'] = numpy.full(
            shape=self.ocb.r.shape, fill_value=ocbpy.ocb_correction.circular)
        self.set_south['rfunc_kwargs'] = numpy.full(shape=self.ocb.r.shape,
                                                    fill_value={"r_add": 1.0})
        self.ocb = self.test_class(**self.set_south)

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(numpy.all(self.ocb.aacgm_boundary_lat[rind] < 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].min(),
                               -80.91948884759928)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].max(),
                               -77.11526278241867)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmax(), 4)

        del rind

    def test_aacgm_boundary_location_good_south_corrected_kwarg_arr(self):
        """Test kwarg array init with good, southern, corrected OCB."""
        rind = 8
        self.set_south['rfunc_kwargs'] = numpy.full(shape=self.ocb.r.shape,
                                                    fill_value={"r_add": 1.0})
        self.ocb = self.test_class(**self.set_south)

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(numpy.all(self.ocb.aacgm_boundary_lat[rind] < 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].min(),
                               -80.91948884759928)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].max(),
                               -77.11526278241867)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmax(), 4)

        del rind

    def test_aacgm_boundary_location_good_south_corrected_dict(self):
        """Test dict init with good, southern, corrected OCB."""
        rind = 8
        self.set_south['rfunc_kwargs'] = {"r_add": 1.0}
        self.ocb = self.test_class(**self.set_south)

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(numpy.all(self.ocb.aacgm_boundary_lat[rind] < 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].min(),
                               -80.91948884759928)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[rind].max(),
                               -77.11526278241867)
        self.assertEqual(self.ocb.aacgm_boundary_lat[rind].argmax(), 4)

    def test_aacgm_boundary_location_good_south_corrected(self):
        """Test finding the corrected OCB in AACGM coordinates in the south."""
        self.out = self.ocb.rec_ind
        self.ocb.rfunc_kwargs[self.ocb.rec_ind]['r_add'] = 1.0

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=self.out)

        # Test value of latitude attribute
        self.assertTrue(numpy.all(self.ocb.aacgm_boundary_lat[self.out] < 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[self.out].min(),
                               -80.91948884759928)
        self.assertEqual(self.ocb.aacgm_boundary_lat[self.out].argmin(), 1)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[self.out].max(),
                               -77.11526278241867)
        self.assertEqual(self.ocb.aacgm_boundary_lat[self.out].argmax(), 4)

    def test_aacgm_boundary_location_partial_fill(self):
        """Test the partial filling when some indices are specified."""
        self.out = 8
        self.test_aacgm_boundary_location_good_south()

        for i in range(self.ocb.records):
            if i != self.out:
                self.assertTrue(self.ocb.aacgm_boundary_lat[i] is None)
                self.assertTrue(self.ocb.aacgm_boundary_mlt[i] is None)
                self.assertTrue(self.ocb.aacgm_boundary_lon[i] is None)
            else:
                self.assertEqual(self.ocb.aacgm_boundary_lat[i].shape,
                                 self.ocb.aacgm_boundary_mlt[i].shape)
                self.assertEqual(self.ocb.aacgm_boundary_lat[i].shape,
                                 self.ocb.aacgm_boundary_lon[i].shape)
                self.assertEqual(self.ocb.aacgm_boundary_mlt[i].shape,
                                 self.mlt.shape)


class TestOCBoundaryFailure(unittest.TestCase):
    """Test the OCBoundary class failures raise appropriate errors."""

    def setUp(self):
        """Set up the test environment."""
        self.test_class = ocbpy.OCBoundary
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.test_class
        return

    def test_bad_instrument_input(self):
        """Test failure when bad instrument value is input."""

        test_north = path.join(path.dirname(ocbpy.__file__), "tests",
                               "test_data", "test_north_circle")
        self.assertTrue(path.isfile(test_north))
        with self.assertRaisesRegex(ValueError, "unknown instrument"):
            self.test_class(instrument="hi", filename=test_north)

        del test_north

    def test_bad_hemisphere_input(self):
        """Test failure when incorrect hemisphere value is input."""
        with self.assertRaisesRegex(ValueError, "hemisphere must be 1"):
            self.test_class(hemisphere=0)

    def test_bad_shape_rfunc_input(self):
        """Test failure when badly shaped radial correction function."""
        with self.assertRaisesRegex(ValueError,
                                    "Misshaped correction function array"):
            self.test_class(
                rfunc=numpy.array([ocbpy.ocb_correction.circular]))

    def test_bad_shape_rfunc_kwarg_input(self):
        """Test failure when badly shaped radial correction function kwargs."""
        with self.assertRaisesRegex(ValueError,
                                    "Misshaped correction function keyword"):
            self.test_class(rfunc_kwargs=numpy.array([{}]))

    def test_bad_rfunc_input(self):
        """Test failure with bad radial correction function input."""
        with self.assertRaisesRegex(
                ValueError, "Unknown input type for correction function"):
            self.test_class(rfunc="rfunc")

    def test_bad_rfunc_kwarg_input(self):
        """Test failure with bad radial correction function kwarg input."""
        with self.assertRaisesRegex(
                ValueError, "Unknown input type for correction keywords"):
            self.test_class(rfunc_kwargs="rfunc")
