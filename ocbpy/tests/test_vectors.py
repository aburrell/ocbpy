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
        self.lt = [0.0, 24.0, 6.0, 21.22, 22.0, 6.0, 6.0, 0.0]
        self.lat = [50.0, 50.0, 50.0, 87.2, 75.0, 88.62, 87.24, 89.0]
        self.pole_lt = [0.0, 12.0, 18.0, 1.260677777, 5.832, 6.0, 0.0, 22.0]
        self.pole_lat = [88.0, 88.0, 88.0, 83.99, 87.24, 87.24, 87.24, 85.0]
        self.pole_ang = [0.0, 0.0, 0.0, 91.72024697182087, 8.67527923, 180.0,
                         45.03325090532819, 143.11957692472973]
        self.out = None
        self.comp = None
        return

    def tearDown(self):
        """Clean up the test class."""
        del self.lt, self.lat, self.out, self.comp
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
        for hemi in [-1, 1]:
            for i, self.comp in enumerate(self.pole_ang):
                # Set the input arguments
                args = [self.lt[i], hemi * self.lat[i], self.pole_lt[i],
                        hemi * self.pole_lat[i]]

                with self.subTest(args=args):
                    # Calculate the pole angle
                    self.out = vectors.calc_vec_pole_angle(*args)

                    # Test the output
                    self.assertAlmostEqual(self.out, self.comp)
        return

    def test_calc_vec_pole_angle_array(self):
        """Test angle calc between base pole, data, and new pole for arrays."""
        # Set locations designed to test no angle, flat lines, acute angles,
        # isocelese triangles, and oblique angles
        self.lt = np.asarray(self.lt)
        self.lat = np.asarray(self.lat)
        self.pole_lt = np.asarray(self.pole_lt)
        self.pole_lat = np.asarray(self.pole_lat)
        self.comp = np.array(self.pole_ang)

        for hemi in [-1, 1]:
            with self.subTest(hemi=hemi):
                # Calculate the pole angle
                self.out = vectors.calc_vec_pole_angle(
                    self.lt, hemi * self.lat, self.pole_lt,
                    hemi * self.pole_lat)

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
                self.assertTupleEqual(self.out.shape, self.comp.shape)
                self.assertTrue(np.all(abs(self.out - self.comp) < 1e-5),
                                msg="{:} != {:}".format(self.out, self.comp))
        return

    def test_define_pole_quadrants_float(self):
        """Test the LT quadrant ID for the destination pole for floats."""
        # Set the expected pole quadrant output
        self.comp = [1, 2, 2, 4, 1, 4, 2, 3]

        # Cycle through each of the local time pairs
        for i, data_lt in enumerate(self.lt):
            # Set the function arguements
            args = [data_lt, self.pole_lt[i], self.pole_ang[i]]

            with self.subTest(args=args):
                # Get the output
                self.out = vectors.define_pole_quadrants(*args)

                # Test the integer quadrant assignment
                self.assertEqual(self.out, self.comp[i])
        return

    def test_define_pole_quadrants_array(self):
        """Test the LT quadrant ID for the destination pole for arrays."""
        # Set the expected pole quadrant output
        self.comp = np.array([1, 2, 2, 4, 1, 4, 2, 3])

        # Cycle through list-like or array-like inputs
        for is_array in [True, False]:
            # Set the function arguements
            args = [self.lt, self.pole_lt, self.pole_ang]

            if is_array:
                for i, arg in enumerate(args):
                    args[i] = np.asarray(arg)

            with self.subTest(is_array=is_array):
                # Get the output
                self.out = vectors.define_pole_quadrants(*args)

                # Test the integer quadrant assignment
                self.assertTupleEqual(self.out.shape, self.comp.shape)
                self.assertTrue(np.all(self.out == self.comp),
                                msg="{:} != {:}".format(self.out, self.comp))
        return

    def test_define_pole_quadrants_mixed(self):
        """Test the LT quadrant ID for the destination pole for mixed input."""
        # Cycle through the mixed inputs
        for args, self.comp in [
                ([self.lt[0], np.asarray(self.pole_lt)[:2],
                  np.asarray(self.pole_ang)[:2]], np.array([1, 2])),
                ([np.asarray(self.lt)[:2], self.pole_lt[0], self.pole_ang[0]],
                 np.array([1, 1]))]:
            with self.subTest(args=args):
                # Get the output
                self.out = vectors.define_pole_quadrants(*args)

                # Test the integer quadrant assignment
                self.assertTupleEqual(self.out.shape, self.comp.shape)
                self.assertTrue(np.all(self.out == self.comp),
                                msg="{:} != {:}".format(self.out, self.comp))
        return

    def test_define_pole_quadrants_neg_diff(self):
        """Test the LT quadrant ID for the dest pole with very neg LT diffs."""
        self.lt = np.asarray(self.lt) + 48.0

        # Cycle through float and array-like input
        for is_float in [True, False]:
            if is_float:
                args = [self.lt[-1], self.pole_lt[-1], self.pole_ang[-1]]
                self.comp = np.asarray(3)
            else:
                args = [self.lt, self.pole_lt, self.pole_ang]
                self.comp = np.array([1, 2, 2, 4, 1, 4, 2, 3])

            with self.subTest(args=args):
                # Get the output
                self.out = vectors.define_pole_quadrants(*args)

                # Test the integer quadrant assignment
                self.assertTupleEqual(self.out.shape, self.comp.shape)
                self.assertTrue(np.all(self.out == self.comp),
                                msg="{:} != {:}".format(self.out, self.comp))
        return

    def test_define_pole_quadrants_large_diff(self):
        """Test the LT quadrant ID for the dest pole with very pos LT diffs."""
        self.pole_lt = np.asarray(self.pole_lt) + 48.0

        # Cycle through float and array-like input
        for is_float in [True, False]:
            if is_float:
                args = [self.lt[-1], self.pole_lt[-1], self.pole_ang[-1]]
                self.comp = np.asarray(3)
            else:
                args = [self.lt, self.pole_lt, self.pole_ang]
                self.comp = np.array([1, 2, 2, 4, 1, 4, 2, 3])

            with self.subTest(args=args):
                # Get the output
                self.out = vectors.define_pole_quadrants(*args)

                # Test the integer quadrant assignment
                self.assertTupleEqual(self.out.shape, self.comp.shape)
                self.assertTrue(np.all(self.out == self.comp),
                                msg="{:} != {:}".format(self.out, self.comp))
        return

    def test_define_vect_quadrants_float(self):
        """Test the vector direction quadrant ID for floats."""
        # Set the expected pole quadrant output
        self.comp = {1: {1: 1, -1: 2}, -1: {1: 4, -1: 3}}

        # Cycle through each of the North directions
        for vect_n in [1.0, 0.0, -1.0]:
            nkey = 1 if vect_n >= 0.0 else -1

            # Cycle through each of the East directions
            for vect_e in [1.0, 0.0, -1.0]:
                ekey = 1 if vect_e >= 0 else -1

                with self.subTest(vect_n=vect_n, vect_e=vect_e):
                    # Get the output
                    self.out = vectors.define_vect_quadrants(vect_n, vect_e)

                    # Test the integer quadrant assignment
                    self.assertEqual(self.out, self.comp[nkey][ekey])
        return

    def test_define_vect_quadrants_array(self):
        """Test the vector direction quadrant ID for array-like input."""
        # Set the vector input and expected output
        self.lt = [2.0, 0.0, -1.0, 3.0, -4.5, 0.0]  # North vect component
        self.lat = [3.0, 1.0, 3.5, -1.0, -2.0, 0.0]  # East vect component
        self.comp = np.array([1, 1, 4, 2, 3, 1])

        # Cycle through list-like or array-like inputs
        for is_array in [True, False]:
            # Set the function arguements
            args = [self.lt, self.lat]

            if is_array:
                for i, arg in enumerate(args):
                    args[i] = np.asarray(arg)

            with self.subTest(is_array=is_array):
                # Get the output
                self.out = vectors.define_vect_quadrants(*args)

                # Test the integer quadrant assignment
                self.assertTupleEqual(self.out.shape, self.comp.shape)
                self.assertTrue(np.all(self.out == self.comp),
                                msg="{:} != {:}".format(self.out, self.comp))
        return

    def test_define_vect_quadrants_mixed(self):
        """Test the vector direction quadrant ID for mixed input."""
        # Cycle through the mixed inputs
        for args, self.comp in [([6.0, [3.0, -3.5]], np.array([1, 2])),
                                ([[1.0, -1.0], 0.0], np.array([1, 4]))]:
            with self.subTest(args=args):
                # Get the output
                self.out = vectors.define_vect_quadrants(*args)

                # Test the integer quadrant assignment
                self.assertTupleEqual(self.out.shape, self.comp.shape)
                self.assertTrue(np.all(self.out == self.comp),
                                msg="{:} != {:}".format(self.out, self.comp))
        return

    def test_calc_dest_polar_angle_float(self):
        """Test the north azimuth angle calculation for float inputs."""
        # Set the expected pole quadrant output
        self.comp = {1: {1: 138.11957692472973, 2: 148.11957692472973,
                         3: 148.11957692472973, 4: -138.11957692472973},
                     2: {1: 148.11957692472973, 2: 138.11957692472973,
                         3: -138.11957692472973, 4: 148.11957692472973},
                     3: {1: 148.11957692472973, 2: 138.11957692472973,
                         3: 138.11957692472973, 4: 211.88042307527027},
                     4: {1: 138.11957692472973, 2: 148.11957692472973,
                         3: 211.88042307527027, 4: 138.11957692472973}}

        # Cycle through each of the pole quadrants
        for pole_quad in np.arange(1, 5, 1):
            # Cycle through each of the vector quadrants
            for vect_quad in np.arange(1, 5, 1):
                with self.subTest(pole_quad=pole_quad, vect_quad=vect_quad):
                    # Get the output
                    self.out = vectors.calc_dest_polar_angle(
                        pole_quad, vect_quad, 5.0, self.pole_ang[-1])

                    # Test the integer quadrant assignment
                    self.assertEqual(self.out, self.comp[pole_quad][vect_quad])
        return

    def test_calc_dest_polar_angle_array(self):
        """Test the north azimuth angle calculation for array-like inputs."""
        # Set the expected inputs and outputs
        self.lt = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        self.lat = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        self.pole_lt = list(np.linspace(0.0, 360.0, num=len(self.lt)))
        self.pole_ang = list(self.pole_ang) + list(self.pole_ang)
        self.comp = np.array([0.0, 24.0, 48.0, -19.72024697, 104.67527923, 60.0,
                              98.96674909, 48.88042308, 168.0, -216.0, 240.0,
                              4.27975303, -279.32472077, -132.0, -21.03325091,
                              216.88042308])

        # Cycle through list-like or array-like inputs
        for is_array in [True, False]:
            # Set the function arguements
            args = [self.lt, self.lat, self.pole_lt, self.pole_ang]

            if is_array:
                for i, arg in enumerate(args):
                    args[i] = np.asarray(arg)

            with self.subTest(is_array=is_array):
                # Get the output
                self.out = vectors.calc_dest_polar_angle(*args)

                # Test the integer quadrant assignment
                self.assertTupleEqual(self.out.shape, self.comp.shape)
                self.assertTrue(np.all(abs(self.out - self.comp) < 1.0e-5),
                                msg="{:} != {:}".format(self.out, self.comp))
        return

    def test_calc_dest_polar_angle_mixed(self):
        """Test the north azimuth angle calculation for mixed inputs."""
        self.lt = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        self.lat = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        self.pole_lt = list(np.linspace(0.0, 360.0, num=len(self.lt)))
        self.pole_ang = list(self.pole_ang) + list(self.pole_ang)
        self.comp = np.array([0.0, 24.0, 48.0, -19.72024697, 104.67527923, 60.0,
                              98.96674909, 48.88042308, 168.0, -216.0, 240.0,
                              4.27975303, -279.32472077, -132.0, -21.03325091,
                              216.88042308])

        # Cycle through the mixed inputs
        for args, self.comp in [([1, [2, 3], [0.0, 24.0], 0.0],
                                 np.array([0.0, 24.0])),
                                ([[1, 4], 4, [72, 360], [91.720246, 143.11957]],
                                 np.array([-19.720246, 216.88043])),
                                ([[1, 2], [3, 4], 5.0, [91.720246, 143.11957]],
                                 np.array([96.720246, 148.11957])),
                                ([[3, 4], [2, 1], [0.0, 90.0], 10.0],
                                 np.array([10.0, -80.0]))]:
            with self.subTest(args=args):
                # Get the output
                self.out = vectors.calc_dest_polar_angle(*args)

                # Test the integer quadrant assignment
                self.assertTupleEqual(self.out.shape, self.comp.shape)
                self.assertTrue(np.all(abs(self.out - self.comp) < 1.0e-5),
                                msg="{:} != {:}".format(self.out, self.comp))
        return

    def test_calc_dest_polar_angle_bad_pole_quad(self):
        """Test the north azimuth angle calculation with undefined pole quad."""

        self.comp = "destination coordinate pole quadrant is undefined"
        with self.assertRaisesRegex(ValueError, self.comp):
            ocbpy.vectors.calc_dest_polar_angle(0, 1, 5.0, 5.0)
        return

    def test_calc_dest_polar_angle_bad_vect_quad(self):
        """Test the north azimuth angle calculation with undefined vect quad."""

        self.comp = "data vector quadrant is undefined"
        with self.assertRaisesRegex(ValueError, self.comp):
            ocbpy.vectors.calc_dest_polar_angle(1, 0, 5.0, 5.0)
        return
        