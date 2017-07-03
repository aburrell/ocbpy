#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocboundary class and functions
"""

import ocbpy
import unittest

class TestOCBoundaryMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file
        """
        from os.path import isfile

        ocb_dir = ocbpy.__file__.split("/")
        test_file = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                       "tests/test_data/test_north_circle")
        self.assertTrue(isfile(test_file))
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=test_file)

    def tearDown(self):
        del self.ocb

    def test_load(self):
        """ Ensure that records from the default file were loaded
        """
        self.assertGreater(self.ocb.records, 0)

    def test_first_good(self):
        """ Test to see that we can find the first good point
        """
        self.ocb.rec_ind = -1

        self.ocb.get_next_good_ocb_ind()
        self.assertGreater(self.ocb.rec_ind, -1)
        self.assertLess(self.ocb.rec_ind, self.ocb.records)

    def test_normal_coord(self):
        """ Test to see that the normalisation is performed properly
        """
        self.ocb.rec_ind = 27
        
        ocb_lat, ocb_mlt = self.ocb.normal_coord(90.0, 0.0)
        self.assertAlmostEquals(ocb_lat, 86.8658623137)
        self.assertAlmostEquals(ocb_mlt, 17.832)

    def test_year_soy_to_datetime(self):
        """ Test to see that the seconds of year conversion works
        """
        import datetime as dt

        self.assertEquals(ocbpy.ocboundary.year_soy_to_datetime(2001, 0),
                          dt.datetime(2001,1,1))

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
        self.assertEquals(idat, 0)
        self.assertEquals(self.ocb.rec_ind, 27)

        # The next test time will cause the OCB to cycle forward to a new
        # record
        idat = ocbpy.ocboundary.match_data_ocb(self.ocb, test_times, idat=1)
        self.assertEquals(idat, 1)
        self.assertEquals(self.ocb.rec_ind, 31)
        self.assertLess(abs((test_times[idat] -
                             self.ocb.dtime[self.ocb.rec_ind]).total_seconds()),
                        600.0)

if __name__ == '__main__':
    unittest.main()
