#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the functions for vector transformation."""

import numpy as np
import unittest

from ocbpy import vectors


class TestOCBVectors(unittest.TestCase):
    """Unit tests for the vectors functions."""

    def setUp(self):
        """Initialize the test class."""
        self.out = None
        self.comp = None
        return

    def tearDown(self):
        """Clean up the test class."""
        del self.out, self.comp
        return

    def test_get_pole_loc_float(self):
        """Test convertion of one pole loc from relative to lt/lat coords."""
        # Cycle through a range of inputs and outputs
        for phi, rad, self.comp in [(0, 5, (0.0, 85.0)), (90, 4, (6.0, 86.0)),
                                    (180, 0, (12.0, 90.0)),
                                    (270, 1, (18.0, 89.0)),
                                    (360, 1, (0.0, 89.0))]:
            with self.subTest(phi=phi, rad=rad):
                # Convert the input
                self.out = vectors.get_pole_loc(phi, rad)

                # Evaluate the output
                self.assertTupleEqual(self.out, self.comp)
        return

    def test_get_pole_loc_array(self):
        """Test convertion of one pole loc from relative to lt/lat coords."""
        # Set the inputs and expected outputs
        phi = [0.0, 90.0, 180.0, 270.0, 360.0, 450.0]
        rad = [5.0, 4.0, 3.0, 2.0, 1.0, 6.0, 0.0]
        self.comp = (np.array([0.0, 6.0, 12.0, 18.0, 0.0, 6.0]),
                     np.array([85.0, 86.0, 87.0, 88.0, 89.0, 84.0, 90.0]))
        
        # Convert the input
        self.out = vectors.get_pole_loc(phi, rad)

        # Evaluate the outputs
        for i, comp_array in enumerate(self.comp):
            self.assertTrue(np.all(self.out[i] == comp_array),
                            msg="unexpected array output: {:} != {:}".format(
                                self.out[i], comp_array))
        return
