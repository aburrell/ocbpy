#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocboundary class and functions
"""

import unittest
import numpy as np
import sys

import logbook

import ocbpy

class TestOCBoundaryLogFailure(unittest.TestCase):
    def setUp(self):
        """ Initialize the test class"""
        self.log_handler = logbook.TestHandler()
        self.log_handler.push_thread()

    def tearDown(self):
        """ Tear down the test case"""
        self.log_handler.pop_thread()

        del self.log_handler

    def test_bad_instrument_name(self):
        """ Test OCB initialization with bad instrument name
        """
        # Initialize the OCBoundary class with bad instrument names
        for val in [1, None, True]:
            ocb = ocbpy.OCBoundary(instrument=val)
            self.assertIsNone(ocb.filename)
            self.assertIsNone(ocb.instrument)

        log_rec = self.log_handler.formatted_records
        # Test logging error message for each bad initialization
        self.assertEqual(len(log_rec), 3)

        for val in log_rec:
            self.assertTrue(val.find("OCB instrument must be a string") > 0)

        del log_rec, val, ocb

    def test_bad_file_name(self):
        """ Test OCB initialization with bad file name
        """
        # Initialize the OCBoundary class with bad instrument names
        for val in [1, None, True]:
            ocb = ocbpy.OCBoundary(filename=val)
            self.assertIsNone(ocb.filename)

        log_rec = self.log_handler.formatted_records
        # Test logging error message for the non-None bad initializations
        self.assertEqual(len(log_rec), 2)

        for val in log_rec:
            self.assertTrue(val.find("file is not a string") > 0)

        del log_rec, val, ocb

    def test_bad_default_file_name(self):
        """ Test OCB initialization with a bad default file name
        """
        # Set a bad default boundary file name
        ocbpy.__default_file__ = "hi"

        ocb = ocbpy.OCBoundary()
        self.assertIsNone(ocb.filename)

        log_rec = self.log_handler.formatted_records
        # Test logging error message for the non-None bad initializations
        self.assertTrue(log_rec[-1].find("problem with default OC Boundary")>0)

        del log_rec, ocb

    def test_bad_default_pairing(self):
        """ Test OCB initialization with a bad default file/instrument pairing
        """
        # Try to load AMPERE data with an IMAGE file
        ocb = ocbpy.OCBoundary(instrument="ampere")
        self.assertIsNone(ocb.filename)

        log_rec = self.log_handler.formatted_records
        # Test logging error message for the non-None bad initializations
        self.assertTrue(log_rec[-1].find("default OC Boundary file uses IMAGE")
                        > 0)

        del log_rec, ocb

    def test_bad_filename(self):
        """ Test OCB initialization with a bad default file/instrument pairing
        """
        # Try to load AMPERE data with an IMAGE file
        ocb = ocbpy.OCBoundary(filename="hi")
        self.assertIsNone(ocb.filename)

        log_rec = self.log_handler.formatted_records
        # Test logging error message for the non-None bad initializations
        self.assertEqual(len(log_rec), 2)
        self.assertTrue(log_rec[0].find("name provided is not a file") > 0)
        self.assertTrue(log_rec[1].find("cannot open OCB file [hi]") > 0)

        del log_rec, ocb

    def test_bad_time_structure(self):
        """ Test OCB initialization without complete time data in file
        """
        from os import path

        # Initialize without a file so that custom loading is performed
        ocb = ocbpy.OCBoundary(filename=None)
        self.assertIsNone(ocb.filename)

        # Set the filename
        ocb.filename = path.join(path.split(ocbpy.__file__)[0], "tests",
                                 "test_data", "test_north_circle")
        self.assertTrue(path.isfile(ocb.filename))

        # Load the data, skipping the year
        ocb.load(ocb_cols="skip soy num_sectors phi_cent r_cent r a r_err")

        log_rec = self.log_handler.formatted_records
        # Test logging error message for the non-None bad initializations
        self.assertEqual(len(log_rec), 1)
        self.assertTrue(log_rec[0].find("missing time columns in") > 0)

        del log_rec, ocb

class TestOCBoundaryMethodsNorth(unittest.TestCase):
    def setUp(self):
        """ Initialize the OCBoundary object using the test file
        """
        from os import path

        ocb_dir = path.split(ocbpy.__file__)
        self.test_north = path.join(ocb_dir[0], "tests", "test_data",
                                    "test_north_circle")
        self.assertTrue(path.isfile(self.test_north))
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=self.test_north)
        
        self.lon = np.linspace(0.0, 360.0, num=6)

    def tearDown(self):
        del self.ocb, self.test_north, self.lon

    def test_bad_rfunc_inst(self):
        """Test failure setting default rfunc for unknown instrument"""
        with self.assertRaisesRegexp(ValueError, "unknown instrument"):
            self.ocb.instrument = "bad"
            self.ocb.rfunc = None
            self.ocb.load()
        
    def test_default_repr(self):
        """ Test the default class representation """
        out = self.ocb.__repr__()

        if sys.version_info.major == 2:
            self.assertRegexpMatches(out, "Open-Closed Boundary file:")
        else:
            self.assertRegex(out, "Open-Closed Boundary file:")

        del out

    def test_empty_repr(self):
        """ Test the unset class representation """

        self.ocb = ocbpy.ocboundary.OCBoundary(filename=None)
        out = self.ocb.__repr__()

        if sys.version_info.major == 2:
            self.assertRegexpMatches(out, "No Open-Closed Boundary file ")
        else:
            self.assertRegex(out, "No Open-Closed Boundary file specified")

        del out

    def test_default_str(self):
        """ Test the default class representation for string output"""

        self.assertTrue(self.ocb.__str__() == self.ocb.__repr__())

    def test_attrs(self):
        """ Test the default attributes in the north
        """

        for tattr in ["filename", "instrument", "hemisphere", "records",
                      "rec_ind", "dtime", "phi_cent", "r_cent", "r",
                      "boundary_lat"]:
            self.assertTrue(hasattr(self.ocb, tattr))

        # Ensure optional attributes are absent
        for tattr in ["aacgm_boundary_lon", "aacgm_boundary_lat"]:
            self.assertFalse(hasattr(self.ocb, tattr))

    def test_image_attrs(self):
        """ Test IMAGE attributes in the north
        """

        for tattr in ["num_sectors", "year", "soy", "r_err", "a"]:
            self.assertTrue(hasattr(self.ocb, tattr))

    def test_ampere_attrs(self):
        """ Test AMPERE attributes don't exist when IMAGE is loaded
        """

        for tattr in ['date', 'time', 'x', 'y', 'j_mag']:
            self.assertFalse(hasattr(self.ocb, tattr))
        
    def test_nofile_init(self):
        """ Ensure that the class can be initialised without loading a file.
        """
        nofile_ocb = ocbpy.ocboundary.OCBoundary(filename=None)

        self.assertIsNone(nofile_ocb.filename)
        self.assertIsNone(nofile_ocb.dtime)
        self.assertEqual(nofile_ocb.records, 0)
        del nofile_ocb

    def test_wrong_instrument(self):
        """ test failure when default file and instrument disagree
        """

        nofile_ocb = ocbpy.ocboundary.OCBoundary(instrument="AMPERE")

        self.assertIsNone(nofile_ocb.filename)
        self.assertIsNone(nofile_ocb.dtime)
        self.assertEqual(nofile_ocb.records, 0)
        del nofile_ocb
        
    def test_load(self):
        """ Ensure correctly loaded defaults in the north
        """
        self.assertGreater(self.ocb.records, 0)
        self.assertEqual(self.ocb.boundary_lat, 74.0)

    def test_partial_load(self):
        """ Ensure limited sections of a file can be loaded in the north
        """
        import datetime as dt

        stime = self.ocb.dtime[0] + dt.timedelta(seconds=1)
        etime = self.ocb.dtime[-1] - dt.timedelta(seconds=1)

        # Load all but the first and last records
        part_ocb = ocbpy.ocboundary.OCBoundary(filename=self.ocb.filename,
                                               stime=stime, etime=etime,
                                               boundary_lat=75.0)

        self.assertEqual(self.ocb.records, part_ocb.records + 2)
        self.assertEqual(part_ocb.boundary_lat, 75.0)
        del part_ocb

    def test_first_good(self):
        """ Test to see that we can find the first good point in the north
        """
        self.ocb.rec_ind = -1

        self.ocb.get_next_good_ocb_ind()

        self.assertGreater(self.ocb.rec_ind, -1)
        self.assertLess(self.ocb.rec_ind, self.ocb.records)

    def test_normal_coord_north(self):
        """ Test the normalisation calculation in the north
        """
        self.ocb.rec_ind = 27
        
        ocb_lat, ocb_mlt, r_corr = self.ocb.normal_coord(90.0, 0.0)
        self.assertAlmostEqual(ocb_lat, 86.8658623137)
        self.assertAlmostEqual(ocb_mlt, 17.832)
        self.assertEqual(r_corr, 0.0)
        del ocb_lat, ocb_mlt, r_corr

    def test_normal_coord_north_w_south(self):
        """ Test the normalisation calculation in the north with southern lat
        """
        self.ocb.rec_ind = 27
        
        out = self.ocb.normal_coord(-80.0, 0.0)
        self.assertEqual(len(out), 3)

        for val in out:
            self.assertTrue(np.isnan(val))

        del out, val

    def test_normal_coord_low_rec_ind(self):
        """ Test the normalization calculation failure with low record index
        """
        self.ocb.rec_ind = -1
        
        out = self.ocb.normal_coord(80.0, 0.0)
        self.assertEqual(len(out), 3)

        for val in out:
            self.assertTrue(np.isnan(val))

        del out, val

    def test_normal_coord_high_rec_ind(self):
        """ Test the normalization calculation failure with high record index
        """
        self.ocb.rec_ind = self.ocb.records + 1
        
        out = self.ocb.normal_coord(80.0, 0.0)
        self.assertEqual(len(out), 3)

        for val in out:
            self.assertTrue(np.isnan(val))

        del out, val

    def test_revert_coord_north(self):
        """ Test the reversion to AACGM coordinates in the north
        """
        self.ocb.rec_ind = 27
        
        ocb_lat, ocb_mlt, r_corr = self.ocb.normal_coord(80.0, 0.0)
        aacgm_lat, aacgm_mlt = self.ocb.revert_coord(ocb_lat, ocb_mlt, r_corr)
        self.assertAlmostEqual(aacgm_lat, 80.0)
        self.assertAlmostEqual(aacgm_mlt, 0.0)
        del ocb_lat, ocb_mlt, r_corr, aacgm_lat, aacgm_mlt

    def test_revert_coord_north_w_south(self):
        """ Test the reversion calculation in the north with southern lat
        """
        self.ocb.rec_ind = 27
        
        out = self.ocb.revert_coord(-80.0, 0.0)
        self.assertEqual(len(out), 2)

        for val in out:
            self.assertTrue(np.isnan(val))

        del out, val

    def test_revert_coord_low_rec_ind(self):
        """ Test the reversion calculation failure with low record index
        """
        self.ocb.rec_ind = -1
        
        out = self.ocb.revert_coord(80.0, 0.0)
        self.assertEqual(len(out), 2)

        for val in out:
            self.assertTrue(np.isnan(val))

        del out, val

    def test_revert_coord_high_rec_ind(self):
        """ Test the reversion calculation failure with high record index
        """
        self.ocb.rec_ind = self.ocb.records + 1
        
        out = self.ocb.revert_coord(80.0, 0.0)
        self.assertEqual(len(out), 2)

        for val in out:
            self.assertTrue(np.isnan(val))

        del out, val

    def test_default_boundary_input(self):
        """ Test to see that the boundary latitude has the correct sign
        """
        self.assertEqual(self.ocb.boundary_lat, 74.0)

    def test_mismatched_boundary_input(self):
        """ Test to see that the boundary latitude has the correct sign
        """
        ocb_n = ocbpy.ocboundary.OCBoundary(filename=self.test_north,
                                            hemisphere=-1)
        self.assertEqual(ocb_n.boundary_lat, -74.0)
        del ocb_n

    def test_retrieve_all_good_ind(self):
        """ Test routine that retrieves all good indices, record set at start
        """

        igood = ocbpy.ocboundary.retrieve_all_good_indices(self.ocb)

        self.assertEqual(igood[0], 27)
        self.assertEqual(igood[1], 31)
        self.assertEqual(len(igood), 36)
        self.assertEqual(self.ocb.rec_ind, -1)

    def test_retrieve_all_good_ind_init_middle(self):
        """ Test routine that retrieves all good indices, record set at middle
        """
        self.ocb.rec_ind = 65
        igood = ocbpy.ocboundary.retrieve_all_good_indices(self.ocb)

        self.assertEqual(igood[0], 27)
        self.assertEqual(igood[1], 31)
        self.assertEqual(len(igood), 36)
        self.assertEqual(self.ocb.rec_ind, 65)

    def test_retrieve_all_good_ind_empty(self):
        """ Test routine that retrieves all good indices, no data loaded
        """
        ocb = ocbpy.ocboundary.OCBoundary(filename=None)
        igood = ocbpy.ocboundary.retrieve_all_good_indices(ocb)

        self.assertEqual(len(igood), 0)

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
                             - self.ocb.aacgm_boundary_lon[-1][:-1]),0)

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
        log_handler = logbook.TestHandler()
        log_handler.push_thread()

        # Initialize the attributes with values for the good location
        rind = 27
        self.test_aacgm_boundary_location_good()
        # This should not raise a warning
        self.ocb.get_aacgm_boundary_lat(150.0, rec_ind=rind-1)
        # This should raise a warning
        self.ocb.get_aacgm_boundary_lat(150.0, rec_ind=rind)

        log_rec = log_handler.formatted_records
        # Test logging error message for only one warning about boundary update
        self.assertEqual(len(log_rec), 1)
        self.assertTrue(log_rec[0].find("unable to update AACGM boundary") > 0)
        log_handler.pop_thread()
        
        del log_rec, log_handler

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
        from os import path

        ocb_dir = path.split(ocbpy.__file__)
        self.test_south = path.join(ocb_dir[0], "tests", "test_data",
                                    "test_south_circle")
        self.assertTrue(path.isfile(self.test_south))
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=self.test_south,
                                               instrument="Ampere",
                                               hemisphere=-1, \
                                            rfunc=ocbpy.ocb_correction.circular)
        
        self.lon = np.linspace(0.0, 360.0, num=6)

    def tearDown(self):
        del self.ocb, self.test_south, self.lon

    def test_attrs(self):
        """ Test the default attributes in the south """

        for tattr in ["filename", "instrument", "hemisphere", "records",
                      "rec_ind", "dtime", "phi_cent", "r_cent", "r",
                      "boundary_lat"]:
            self.assertTrue(hasattr(self.ocb, tattr))

        # Ensure optional attributes are absent
        for tattr in ["aacgm_boundary_lon", "aacgm_boundary_lat"]:
            self.assertFalse(hasattr(self.ocb, tattr))

    def test_image_attrs(self):
        """ Test that IMAGE attributes are not available in the south"""

        for tattr in ["num_sectors", "year", "soy", "r_err", "a"]:
            self.assertFalse(hasattr(self.ocb, tattr))

    def test_ampere_attrs(self):
        """ Test that AMPERE attributes are available in the south"""

        for tattr in ['date', 'time', 'x', 'y', 'j_mag']:
            self.assertTrue(hasattr(self.ocb, tattr))
        
    def test_load(self):
        """ Ensure that the default options were correctly set in the south
        """
        self.assertGreater(self.ocb.records, 0)
        self.assertEqual(self.ocb.boundary_lat, -72.0)

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
        self.ocb.rec_ind = 8

        ocb_lat, ocb_mlt, r_corr = self.ocb.normal_coord(-90.0, 0.0)
        self.assertAlmostEqual(ocb_lat, -86.4)
        self.assertAlmostEqual(ocb_mlt, 6.0)
        self.assertEqual(r_corr, 0.0)
        del ocb_lat, ocb_mlt, r_corr

    def test_normal_coord_south_corrected(self):
        """ Test the normalisation calculation in the south with a corrected OCB
        """
        self.ocb.rec_ind = 8
        self.ocb.rfunc_kwargs[self.ocb.rec_ind]['r_add'] = 1.0

        ocb_lat, ocb_mlt, r_corr = self.ocb.normal_coord(-90.0, 0.0)
        self.assertAlmostEqual(ocb_lat, -86.72727272)
        self.assertAlmostEqual(ocb_mlt, 6.0)
        self.assertEqual(r_corr, 1.0)
        del ocb_lat, ocb_mlt, r_corr

    def test_default_boundary_input(self):
        """ Test to see that the boundary latitude has the correct sign
        """
        self.assertEqual(self.ocb.boundary_lat, -72.0)

    def test_mismatched_boundary_input(self):
        """ Test to see that the boundary latitude has the correct sign
        """
        ocb_s = ocbpy.ocboundary.OCBoundary(filename=self.test_south,
                                            instrument="ampere",
                                            hemisphere=1)
        self.assertEqual(ocb_s.boundary_lat, 72.0)
        del ocb_s

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
        rfuncs = np.full(shape=self.ocb.r.shape,
                         fill_value=ocbpy.ocb_correction.circular)
        rkwargs = np.full(shape=self.ocb.r.shape, fill_value={"r_add": 1.0})
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=self.test_south,
                                               instrument="Ampere",
                                               hemisphere=-1, rfunc=rfuncs,
                                               rfunc_kwargs=rkwargs)

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

        del rind, rkwargs

    def test_aacgm_boundary_location_good_south_corrected_kwarg_arr(self):
        """ Test kwarg array init with good, southern, corrected OCB
        """
        rind = 8
        rkwargs = np.full(shape=self.ocb.r.shape, fill_value={"r_add": 1.0})
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=self.test_south,
                                               instrument="Ampere",
                                               hemisphere=-1, \
                                            rfunc=ocbpy.ocb_correction.circular,
                                               rfunc_kwargs=rkwargs)

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

        del rind, rkwargs

    def test_aacgm_boundary_location_good_south_corrected_dict(self):
        """ Test dict init with good, southern, corrected OCB
        """
        rind = 8
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=self.test_south,
                                               instrument="Ampere",
                                               hemisphere=-1, \
                                            rfunc=ocbpy.ocb_correction.circular,
                                               rfunc_kwargs={"r_add": 1.0})

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
        rind = 8
        self.ocb.rfunc_kwargs[self.ocb.rec_ind]['r_add'] = 1.0

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

    def test_aacgm_boundary_location_partial_fill(self):
        """ Test the partial filling when some indices are specified
        """
        rind = 8
        self.test_aacgm_boundary_location_good_south()

        for i in range(self.ocb.records):
            if i != rind:
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
        from os import path

        ocb_dir = path.split(ocbpy.__file__)
        self.test_north = path.join(ocb_dir[0], "tests", "test_data",
                                    "test_north_circle")
        self.assertTrue(path.isfile(self.test_north))
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=self.test_north)
        
        # Initialize logging
        self.log_handler = logbook.TestHandler()
        self.log_handler.push_thread()

    def tearDown(self):
        self.log_handler.pop_thread()
        del self.ocb, self.test_north, self.log_handler

    def test_match(self):
        """ Test to see that the data matching works properly
        """
        import numpy as np
        import datetime as dt
    
        # Build a array of times for a test dataset
        self.ocb.rec_ind = 27
        test_times = np.arange(self.ocb.dtime[self.ocb.rec_ind],
                               self.ocb.dtime[self.ocb.rec_ind + 5],
                               dt.timedelta(seconds=600)).astype(dt.datetime)

        # Because the array starts at the first good OCB, will return zero
        idat = ocbpy.ocboundary.match_data_ocb(self.ocb, test_times, idat=0)
        self.assertEqual(idat, 0)
        self.assertEqual(self.ocb.rec_ind, 27)

        # The next test time will cause the OCB to cycle forward to a new
        # record
        idat = ocbpy.ocboundary.match_data_ocb(self.ocb, test_times, idat=1)
        self.assertEqual(idat, 1)
        self.assertEqual(self.ocb.rec_ind, 31)
        self.assertLess(abs((test_times[idat] -
                             self.ocb.dtime[self.ocb.rec_ind]).total_seconds()),
                        600.0)
        del test_times, idat

    def test_good_first_match(self):
        """ Test ability to find the first good OCB
        """
        # Because the array starts at the first good OCB, will return zero
        self.ocb.rec_ind = -1
        idat = ocbpy.ocboundary.match_data_ocb(self.ocb, [self.ocb.dtime[27]],
                                               idat=0)
        self.assertEqual(idat, 0)
        self.assertEqual(self.ocb.rec_ind, 27)

        # The first match will be announced in the log
        log_rec = self.log_handler.formatted_records
        self.assertEqual(len(log_rec), 1)
        self.assertTrue(log_rec[0].find("found first good OCB record at") > 0)
       
        del idat, log_rec

    def test_bad_first_match(self):
        """ Test ability to not find a good OCB
        """
        # Set requirements for good OCB so high that none will pass
        self.ocb.rec_ind = -1
        idat = ocbpy.ocboundary.match_data_ocb(self.ocb, [self.ocb.dtime[27]],
                                               idat=0, min_sectors=24)
        self.assertEqual(idat, 0)
        self.assertGreaterEqual(self.ocb.rec_ind, self.ocb.records)

        # The first match will be announced in the log
        log_rec = self.log_handler.formatted_records
        self.assertEqual(len(log_rec), 1)
        self.assertTrue(log_rec[0].find("unable to find a good OCB record") > 0)
        
        del idat, log_rec

    def test_late_data_time_alignment(self):
        """ Test failure when data occurs after boundaries
        """
        import datetime as dt

        # Build a array of times for a test dataset
        test_times = [self.ocb.dtime[self.ocb.records-1] + dt.timedelta(days=2)]

        # Set requirements for good OCB so high that none will pass
        self.ocb.rec_ind = -1
        idat = ocbpy.ocboundary.match_data_ocb(self.ocb, test_times, idat=0)
        self.assertEqual(idat, 0)
        self.assertGreaterEqual(self.ocb.rec_ind, self.ocb.records)

        # Check the log output
        log_rec = self.log_handler.formatted_records
        self.assertTrue(log_rec[-1].find("no OCB data available within") > 0)
        self.assertTrue(log_rec[-1].find("of first measurement") > 0)

        del test_times, idat, log_rec

    def test_no_data_time_alignment(self):
        """ Test failure when data occurs between boundaries
        """
        import datetime as dt

        # Build a array of times for a test dataset
        test_times = [self.ocb.dtime[37] - dt.timedelta(seconds=601)]

        # Set requirements for good OCB so high that none will pass
        self.ocb.rec_ind = -1
        idat = ocbpy.ocboundary.match_data_ocb(self.ocb, test_times, idat=0)
        self.assertEqual(idat, 1)
        self.assertGreaterEqual(self.ocb.rec_ind, 37)

        # Check the log output
        log_rec = self.log_handler.formatted_records
        self.assertTrue(log_rec[-1].find("no OCB data available within") > 0)
        self.assertTrue(log_rec[-1].find("of input measurement") > 0)

        del test_times, idat, log_rec

class TestOCBoundaryFailure(unittest.TestCase):
    def setUp(self):
        """ Initialize the OCBoundary object using the test file
        """
        pass

    def tearDown(self):
        pass

    def test_bad_hemisphere_input(self):
        """ Test failure when incorrect hemisphere value is input"""
        with self.assertRaisesRegexp(ValueError, "hemisphere must be 1"):
            ocbpy.ocboundary.OCBoundary(hemisphere=0)

    def test_bad_shape_rfunc_input(self):
        """ Test failure when badly shaped radial correction function"""
        with self.assertRaisesRegexp(ValueError,
                                     "Misshaped correction function array"):
            ocbpy.ocboundary.OCBoundary(rfunc= \
                                np.array([ocbpy.ocb_correction.circular]))

    def test_bad_shape_rfunc_kwarg_input(self):
        """ Test failure when badly shaped radial correction function kwargs"""
        with self.assertRaisesRegexp(ValueError,
                                     "Misshaped correction function keyword"):
            ocbpy.ocboundary.OCBoundary(rfunc_kwargs=np.array([{}]))

    def test_bad_rfunc_input(self):
        """ Test failure with bad radial correction function input"""
        with self.assertRaisesRegexp(ValueError, \
                                "Unknown input type for correction function"):
            ocbpy.ocboundary.OCBoundary(rfunc="rfunc")

    def test_bad_rfunc_kwarg_input(self):
        """ Test failure with bad radial correction function kwarg input"""
        with self.assertRaisesRegexp(ValueError, \
                                "Unknown input type for correction keywords"):
            ocbpy.ocboundary.OCBoundary(rfunc_kwargs="rfunc")

if __name__ == '__main__':
    unittest.main()
