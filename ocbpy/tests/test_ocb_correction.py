#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocboundary class and functions
"""

import unittest

from ocbpy import ocb_correction as ocb_cor

class TestOCBCorrection(unittest.TestCase):
    def setUp(self):
        """ Set up test runs """
        from os import path
        import ocbpy

        ocb_dir = path.split(ocbpy.__file__)
        self.test_file = path.join(ocb_dir[0], "tests", "test_data",
                                   "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=self.test_file)
        self.ocb.rec_ind = 27
        self.aacgm_mlt = 0.0

    def tearDown(self):
        """ Clean up after each test """
        del self.test_file, self.ocb, self.aacgm_mlt

    def test_circular_default(self):
        """ Test the circular boundary function with default options """

        self.assertEqual(ocb_cor.circular(self.ocb, self.aacgm_mlt),
                         self.ocb.r[self.ocb.rec_ind])

    def test_circular_ocb(self):
        """ Test the circular boundary function for OCB MLT input """

        self.assertEqual(ocb_cor.circular(self.ocb, self.aacgm_mlt,
                                          mlt_coords="ocb"),
                         self.ocb.r[self.ocb.rec_ind])

    def test_circular_offset(self):
        """ Test the circular boundary function with an offset """

        self.assertEqual(ocb_cor.circular(self.ocb, self.aacgm_mlt, r_add=1.0),
                         self.ocb.r[self.ocb.rec_ind] + 1.0)

    def test_circular_low_index(self):
        """ Test the circular boundary function with an index below zero """
        import numpy as np
        self.ocb.rec_ind = -1
        self.assertTrue(np.isnan(ocb_cor.circular(self.ocb, self.aacgm_mlt)))

    def test_circular_high_index(self):
        """ Test the circular boundary function with an index below zero """
        import numpy as np
        self.ocb.rec_ind = self.ocb.records
        self.assertTrue(np.isnan(ocb_cor.circular(self.ocb, self.aacgm_mlt)))

    def test_ampere_harmonic_default(self):
        """ Test the ampere_harmonic boundary function with default options """

        self.assertAlmostEqual(ocb_cor.ampere_harmonic(self.ocb,
                                                       self.aacgm_mlt),
                               15.625666423138952)

    def test_ampere_harmonic_ocb(self):
        """ Test the ampere_harmonic boundary function for OCB MLT input """

        with self.assertRaisesRegexp(ValueError, "routine lacks OCB to AACGM"):
            ocb_cor.ampere_harmonic(self.ocb, self.aacgm_mlt, mlt_coords="ocb")

    def test_ampere_harmonic_gaussian(self):
        """ Test the gaussian ampere_harmonic boundary function """

        self.assertAlmostEqual(ocb_cor.ampere_harmonic(self.ocb, self.aacgm_mlt,
                                                       method="gaussian"),
                               16.362036355151048)

    def test_ampere_harmonic_low_index(self):
        """ Test the ampere_harmonic boundary function with an index below zero
        """
        import numpy as np

        self.ocb.rec_ind = -1
        self.assertTrue(np.isnan(ocb_cor.ampere_harmonic(self.ocb,
                                                         self.aacgm_mlt)))

    def test_ampere_harmonic_high_index(self):
        """ Test the ampere_harmonic boundary function with an index below zero
        """
        import numpy as np

        self.ocb.rec_ind = self.ocb.records
        self.assertTrue(np.isnan(ocb_cor.ampere_harmonic(self.ocb,
                                                         self.aacgm_mlt)))

if __name__ == '__main__':
    unittest.main()
