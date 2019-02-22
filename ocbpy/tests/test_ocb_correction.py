#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocboundary class and functions
"""

import numpy as np
import unittest

from ocbpy import ocb_correction as ocb_cor

class TestOCBCorrection(unittest.TestCase):
    def setUp(self):
        """ Set up test runs """

        self.aacgm_mlt_val = 0.0
        self.aacgm_mlt_arr = np.arange(0.0, 24.0, 12.0)

    def tearDown(self):
        """ Clean up after each test """
        del self.aacgm_mlt_val, self.aacgm_mlt_arr

    def test_circular_default_float(self):
        """ Test the default circular boundary function with a float"""

        self.assertEqual(ocb_cor.circular(self.aacgm_mlt_val), 0.0)

    def test_circular_default_arr(self):
        """ Test the default circular boundary function with an array"""

        self.assertTrue(np.all(ocb_cor.circular(self.aacgm_mlt_arr) ==
                               np.zeros(shape=self.aacgm_mlt_arr.shape)))

    def test_circular_offset(self):
        """ Test the circular boundary function with an offset """

        self.assertEqual(ocb_cor.circular(self.aacgm_mlt_val, r_add=1.0), 1.0)

    def test_ampere_harmonic_float(self):
        """ Test the default_ampere_harmonic boundary function for a value"""

        self.assertAlmostEqual(ocb_cor.ampere_harmonic(self.aacgm_mlt_val),
                               1.53566642)

    def test_ampere_harmonic_arr(self):
        """ Test the default ampere_harmonic boundary function for an array"""

        href = np.array([1.53566642, 2.53483664])
        
        self.assertTrue(np.all(abs(ocb_cor.ampere_harmonic(self.aacgm_mlt_arr)
                                   - href) < 1.0e-7))

    def test_ampere_harmonic_gaussian(self):
        """ Test the gaussian ampere_harmonic boundary function """

        self.assertAlmostEqual(ocb_cor.ampere_harmonic(self.aacgm_mlt_val,
                                                       method="gaussian"),
                               2.27203636)


if __name__ == '__main__':
    unittest.main()
