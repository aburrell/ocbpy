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

    def test_calc_vec_pole_angle_float(self):
        """Test angle calc between base pole, data, and new pole for floats."""
        # Set locations designed to test no angle, flat lines, acute angles,
        # isocelese triangles, and oblique angles
        lt = [0.0, 24.0, 6.0, 21.22, 22.0, 6.0, 6.0, 4.832]
        lat = [50.0, 50.0, 50.0, 87.2, 75.0, 88.62, 87.24, 88.62]
        pole_lt = [0.0, 12.0, 18.0, 1.260677777, 5.832, 6.0, 0.0, 5.832]
        pole_lat = [88.0, 88.0, 88.0, 83.99, 87.24, 87.24, 87.24, 87.24]
        self.comp = [0.0, 0.0, 0.0, 91.72024697182087, 8.67527923, 180.0,
                     45.03325090532819, 150.9561733411]

        for hemi in [-1, 1]:
            for i, cval in enumerate(self.comp):
                # Set the input arguments
                args = [lt[i], hemi * lat[i], pole_lt[i], hemi * pole_lat[i]]

                with self.subTest(args=args):
                    # Calculate the pole angle
                    self.out = vectors.calc_vec_pole_angle(*args)

                    # Test the output
                    self.assertAlmostEqual(self.out, cval)
        return

    def test_calc_vec_pole_angle_array(self):
        """Test angle calc between base pole, data, and new pole for arrays."""
        # Set locations designed to test no angle, flat lines, acute angles,
        # isocelese triangles, and oblique angles
        lt = np.array([0.0, 24.0, 6.0, 21.22, 22.0, 6.0, 6.0, 4.832])
        lat = np.array([50.0, 50.0, 50.0, 87.2, 75.0, 88.62, 87.24, 88.62])
        pole_lt = np.array([0.0, 12.0, 18.0, 1.260677777, 5.832, 6.0, 0.0,
                            5.832])
        pole_lat = np.array([88.0, 88.0, 88.0, 83.99, 87.24, 87.24, 87.24,
                             87.24])
        self.comp = np.array([0.0, 0.0, 0.0, 91.72024697182087, 8.67527923,
                              180.0, 45.03325090532819, 150.9561733411])

        for hemi in [-1, 1]:
            with self.subTest(hemi=hemi):
                # Calculate the pole angle
                self.out = vectors.calc_vec_pole_angle(
                    lt, hemi * lat, pole_lt, hemi * pole_lat)

                # Test the output
                self.assertTrue(np.all(abs(self.out - self.comp) < 1e-5),
                                msg="{:} != {:}".format(self.out, self.comp))
        return

    def test_calc_vec_pole_angle_mixed(self):
        """Test angle calc for base pole, data, new pole with float and list."""
        # Set locations designed to test no angle, flat lines, acute angles,
        # isocelese triangles, and oblique angles

        for args, self.comp in [
                ([0.0, -50.0, [0.0, 12.0], -88.0], np.zeros(shape=2)),
                ([[6.0, 21.22], [50.0, 87.2], [18.0, 1.26067777], 83.99],
                 np.array([0.0, 91.72024697182087])),
                ([6.0, [88.62, 87.24], [6.0, 0.0], 87.24],
                 np.array([180.0, 45.03325090532819]))]:
            with self.subTest(args=args):
                # Calculate the pole angle
                self.out = vectors.calc_vec_pole_angle(*args)

                # Test the output
                self.assertTrue(self.out.shape == self.comp.shape)
                self.assertTrue(np.all(abs(self.out - self.comp) < 1e-5),
                                msg="{:} != {:}".format(self.out, self.comp))
        return
        
                     
        
