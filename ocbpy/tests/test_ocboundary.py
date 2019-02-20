#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocboundary class and functions
"""

import unittest
import numpy as np

import ocbpy

class TestOCBoundaryMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file
        """
        from os import path

        ocb_dir = path.split(ocbpy.__file__)
        self.test_north = path.join(ocb_dir[0], "tests", "test_data",
                                    "test_north_circle")
        self.test_south = path.join(ocb_dir[0], "tests", "test_data",
                                    "test_south_circle")
        self.assertTrue(path.isfile(self.test_north))
        self.assertTrue(path.isfile(self.test_south))
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=self.test_north)
        self.ocb_south = ocbpy.ocboundary.OCBoundary(filename=self.test_south,
                                                     instrument="Ampere",
                                                     hemisphere=-1)
        # Set a custom correction for AMPERE
        for i in range(self.ocb_south.records):
            self.ocb_south.rfunc[i] = ocbpy.ocb_correction.circular
            self.ocb_south.rfunc_kwargs[i] = {"r_add": 0.0}
        
        self.lon = np.linspace(0.0, 360.0, num=6)

    def tearDown(self):
        del self.ocb, self.ocb_south, self.test_north, self.test_south, self.lon

    def test_attrs(self):
        """ Test the default attributes
        """

        for tattr in ["filename", "instrument", "hemisphere", "records",
                      "rec_ind", "dtime", "phi_cent", "r_cent", "r",
                      "boundary_lat"]:
            self.assertTrue(hasattr(self.ocb, tattr))
            self.assertTrue(hasattr(self.ocb_south, tattr))

        # Ensure optional attributes are absent
        for tattr in ["aacgm_boundary_lon", "aacgm_boundary_lat"]:
            self.assertFalse(hasattr(self.ocb, tattr))
            self.assertFalse(hasattr(self.ocb_south, tattr))

    def test_image_attrs(self):
        """ Test IMAGE attributes
        """

        for tattr in ["num_sectors", "year", "soy", "r_err", "a"]:
            self.assertTrue(hasattr(self.ocb, tattr))
            self.assertFalse(hasattr(self.ocb_south, tattr))

    def test_ampere_attrs(self):
        """ Test AMPERE attributes
        """

        for tattr in ['date', 'time', 'x', 'y', 'j_mag']:
            self.assertTrue(hasattr(self.ocb_south, tattr))
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
        """ Ensure that no file is loaded if user wants an instrument other
        than image, but asks for default file
        """

        nofile_ocb = ocbpy.ocboundary.OCBoundary(instrument="AMPERE")

        self.assertIsNone(nofile_ocb.filename)
        self.assertIsNone(nofile_ocb.dtime)
        self.assertEqual(nofile_ocb.records, 0)
        del nofile_ocb
        
    def test_load(self):
        """ Ensure that records from the default file were loaded and the
        default latitude boundary was set
        """
        self.assertGreater(self.ocb.records, 0)
        self.assertEqual(self.ocb.boundary_lat, 74.0)

        self.assertGreater(self.ocb_south.records, 0)
        self.assertEqual(self.ocb_south.boundary_lat, -72.0)

    def test_partial_load(self):
        """ Ensure limited sections of a file can be loaded
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
        """ Test to see that we can find the first good point
        """
        self.ocb.rec_ind = -1
        self.ocb_south.rec_ind = -1

        self.ocb.get_next_good_ocb_ind()
        self.ocb_south.get_next_good_ocb_ind()

        self.assertGreater(self.ocb.rec_ind, -1)
        self.assertLess(self.ocb.rec_ind, self.ocb.records)

        self.assertGreater(self.ocb_south.rec_ind, -1)
        self.assertLess(self.ocb_south.rec_ind, self.ocb_south.records)

    def test_normal_coord_north(self):
        """ Test to see that the normalisation is performed properly in the
        northern hemisphere
        """
        self.ocb.rec_ind = 27
        
        ocb_lat, ocb_mlt = self.ocb.normal_coord(90.0, 0.0)
        self.assertAlmostEqual(ocb_lat, 86.8658623137)
        self.assertAlmostEqual(ocb_mlt, 17.832)
        del ocb_lat, ocb_mlt

    def test_revert_coord_north(self):

        """ Test to see that the reversion to AACGM coordinates is performed
        properly
        """
        self.ocb.rec_ind = 27
        
        ocb_lat, ocb_mlt = self.ocb.normal_coord(80.0, 0.0)
        aacgm_lat, aacgm_mlt = self.ocb.revert_coord(ocb_lat, ocb_mlt)
        self.assertAlmostEqual(aacgm_lat, 80.0)
        self.assertAlmostEqual(aacgm_mlt, 0.0)
        del ocb_lat, ocb_mlt, aacgm_lat, aacgm_mlt

    def test_normal_coord_south(self):
        """ Test to see that the normalisation is performed properly in the
        southern hemisphere
        """
        self.ocb_south.rec_ind = 8

        ocb_lat, ocb_mlt = self.ocb_south.normal_coord(-90.0, 0.0)
        self.assertAlmostEqual(ocb_lat, -86.4)
        self.assertAlmostEqual(ocb_mlt, 6.0)
        del ocb_lat, ocb_mlt

    def test_normal_coord_south_corrected(self):
        """ Test to see that the normalisation is performed properly in the
        southern hemisphere with a corrected OCB 
        """
        self.ocb_south.rec_ind = 8
        self.ocb_south.rfunc_kwargs[self.ocb_south.rec_ind]['r_add'] = 1.0

        ocb_lat, ocb_mlt = self.ocb_south.normal_coord(-90.0, 0.0)
        self.assertAlmostEqual(ocb_lat, -86.72727272)
        self.assertAlmostEqual(ocb_mlt, 6.0)
        del ocb_lat, ocb_mlt

    def test_default_boundary_input(self):
        """ Test to see that the boundary latitude has the correct sign
        """
        self.assertEqual(self.ocb.boundary_lat, 74.0)
        self.assertEqual(self.ocb_south.boundary_lat, -72.0)

    def test_mismatched_boundary_input(self):
        """ Test to see that the boundary latitude has the correct sign
        """
        ocb_n = ocbpy.ocboundary.OCBoundary(filename=self.test_north,
                                            hemisphere=-1)
        ocb_s = ocbpy.ocboundary.OCBoundary(filename=self.test_south,
                                            instrument="ampere",
                                            hemisphere=1)
        self.assertEqual(ocb_n.boundary_lat, -74.0)
        self.assertEqual(ocb_s.boundary_lat, 72.0)
        del ocb_n, ocb_s

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
        """ Test the calculation of the OCB in AACGM coordinates
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
        """ Test the calculation of the OCB in AACGM coordinates at the first
        good location
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

    def test_aacgm_boundary_location_good_south(self):
        """ Test the calculation of the OCB in AACGM coordinates at the first
        good location for the southern hemisphere
        """
        rind = 8

        # Add the attribute at the good location
        self.ocb_south.get_aacgm_boundary_lat(aacgm_lon=self.lon, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(np.all(self.ocb_south.aacgm_boundary_lat[rind] < 0.0))
        self.assertAlmostEqual(self.ocb_south.aacgm_boundary_lat[rind].min(),
                               -81.92122960532046)
        self.assertEqual(self.ocb_south.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb_south.aacgm_boundary_lat[rind].max(),
                               -78.11700354013985)
        self.assertEqual(self.ocb_south.aacgm_boundary_lat[rind].argmax(), 4)

    def test_aacgm_boundary_location_good_south_corrected(self):
        """ Test the calculation of the OCB in AACGM coordinates at the first
        good location for the southern hemisphere with a corrected OCB
        """
        rind = 8
        self.ocb_south.rfunc_kwargs[self.ocb_south.rec_ind]['r_add'] = 1.0

        # Add the attribute at the good location
        self.ocb_south.get_aacgm_boundary_lat(aacgm_lon=self.lon, rec_ind=rind)

        # Test value of latitude attribute
        self.assertTrue(np.all(self.ocb_south.aacgm_boundary_lat[rind] < 0.0))
        self.assertAlmostEqual(self.ocb_south.aacgm_boundary_lat[rind].min(),
                               -81.92122960532046)
        self.assertEqual(self.ocb_south.aacgm_boundary_lat[rind].argmin(), 1)
        self.assertAlmostEqual(self.ocb_south.aacgm_boundary_lat[rind].max(),
                               -78.11700354013985)
        self.assertEqual(self.ocb_south.aacgm_boundary_lat[rind].argmax(), 4)

    def test_aacgm_boundary_location_bad(self):
        """ Test the calclation of the OCB in AACGM coordinates for a boundary
        that doesn't span all MLT sectors
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

    def test_aacgm_boundary_location_partial_fill(self):
        """ Test the partial filling when some indices are specified
        """
        rind = 8
        self.test_aacgm_boundary_location_good_south()

        for i in range(self.ocb_south.records):
            if i != rind:
                self.assertTrue(self.ocb_south.aacgm_boundary_lat[i] is None)
                self.assertTrue(self.ocb_south.aacgm_boundary_lon[i] is None)
            else:
                self.assertEqual(self.ocb_south.aacgm_boundary_lat[i].shape,
                                 self.ocb_south.aacgm_boundary_lon[i].shape)
                self.assertEqual(self.ocb_south.aacgm_boundary_lon[i].shape,
                                 self.lon.shape)

    def test_aacgm_boundary_location_no_overwrite(self):
        """ Ensure OCB AACGM location will not overwrite calculated AACGM
        boundary locations
        """
        import logbook

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

if __name__ == '__main__':
    unittest.main()
