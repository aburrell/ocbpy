#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DOI: 10.5281/zenodo.1179230
# Full license can be found in License.md
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
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
        self.vect_n = [2.0, 0.0, -1.0, 3.0, -4.5, 0.0, 1.0, -1.0]
        self.vect_e = [3.0, 1.0, 3.5, -1.0, -2.0, 0.0, -1.0, 1.0]
        self.vect_quad = [1, 1, 4, 2, 3, 1, 2, 4]
        self.pole_lt = [0.0, 12.0, 18.0, 1.260677777, 5.832, 6.0, 0.0, 22.0]
        self.pole_lat = [88.0, 88.0, 88.0, 83.99, 87.24, 87.24, 87.24, 85.0]
        self.pole_ang = [0.0, 0.0, 0.0, 91.72024697182087, 8.67527923, 180.0,
                         45.03325090532819, 143.11957692472973]
        self.pole_quad = [1, 2, 2, 4, 1, 4, 2, 3]
        self.out = None
        self.comp = None
        return

    def tearDown(self):
        """Clean up the test class."""
        del self.lt, self.lat, self.vect_n, self.vect_e, self.vect_quad
        del self.pole_lt, self.pole_lat, self.pole_ang, self.pole_quad
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

        # Cycle through each of the local time pairs
        for i, data_lt in enumerate(self.lt):
            # Set the function arguements
            args = [data_lt, self.pole_lt[i], self.pole_ang[i]]

            with self.subTest(args=args):
                # Get the output
                self.out = vectors.define_pole_quadrants(*args)

                # Test the integer quadrant assignment
                self.assertEqual(self.out, self.pole_quad[i])
        return

    def test_define_pole_quadrants_array(self):
        """Test the LT quadrant ID for the destination pole for arrays."""
        # Set the expected pole quadrant output
        self.comp = np.array(self.pole_quad)

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
                self.comp = np.array(self.pole_quad)

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
                self.comp = np.array(self.pole_quad)

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
        # Set the vector quadrant output
        self.comp = np.array(self.vect_quad)

        # Cycle through list-like or array-like inputs
        for is_array in [True, False]:
            # Set the function arguements
            args = [self.vect_n, self.vect_e]

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
        self.comp = {1: {1: [138.11957692472973, 132.0],
                         2: [148.11957692472973, -132.0],
                         3: [148.11957692472973, -132.0],
                         4: [-138.11957692472973, -132.0]},
                     2: {1: [148.11957692472973, -132.0],
                         2: [138.11957692472973, 132.0],
                         3: [-138.11957692472973, -132.0],
                         4: [148.11957692472973, -132.0]},
                     3: {1: [148.11957692472973, -132.0],
                         2: [138.11957692472973, 132.0],
                         3: [138.11957692472973, 132.0],
                         4: [211.88042307527027, -132.0]},
                     4: {1: [138.11957692472973, 132.0],
                         2: [148.11957692472973, -132.0],
                         3: [211.88042307527027, -132.0],
                         4: [138.11957692472973, 132.0]}}

        # Cycle through each of the pole quadrants
        for pole_quad in np.arange(1, 5, 1):
            # Cycle through each of the vector quadrants
            for vect_quad in np.arange(1, 5, 1):
                # Cycle through small and big angles
                for i, ang_args in enumerate([(5.0, self.pole_ang[-1]),
                                              (180.0, 312.0)]):
                    args = [pole_quad, vect_quad, ang_args[0], ang_args[1]]
                    with self.subTest(args=args):
                        # Get the output
                        self.out = vectors.calc_dest_polar_angle(*args)

                        # Test the integer quadrant assignment
                        self.assertEqual(self.out,
                                         self.comp[pole_quad][vect_quad][i],
                                         msg="unexpected output: {:}".format(
                                             self.out))
        return

    def test_calc_dest_polar_angle_array(self):
        """Test the north azimuth angle calculation for array-like inputs."""
        # Set the expected inputs and outputs
        self.pole_quad = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        self.vect_quad = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        self.pole_lt = list(np.linspace(0.0, 360.0, num=len(self.pole_quad)))
        self.pole_ang = list(self.pole_ang) + list(self.pole_ang)
        self.comp = np.array([0.0, 24.0, 48.0, -19.72024697, 104.67527923, 60.0,
                              98.96674909, 48.88042308, 168.0, -216.0, 240.0,
                              4.27975303, -279.32472077, -132.0, -21.03325091,
                              216.88042308])

        # Cycle through list-like or array-like inputs
        for is_array in [True, False]:
            # Set the function arguements
            args = [self.pole_quad, self.vect_quad, self.pole_lt, self.pole_ang]

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
        # Cycle through the mixed inputs
        for args, self.comp in [([1, [2, 3], [0.0, 24.0], 0.0],
                                 np.array([0.0, 24.0])),
                                ([[1, 4], 4, [72, 360], [91.720246, 143.11957]],
                                 np.array([-19.720246, 216.88043])),
                                ([[1, 2], [3, 4], 5.0, [91.720246, 143.11957]],
                                 np.array([96.720246, 148.11957])),
                                ([[3, 4], [2, 1], [0.0, 90.0], 10.0],
                                 np.array([10.0, -80.0])),
                                ([[1, 1, 2, 3, 4], [1, 2, 3, 4, 1], 180, 312],
                                 np.array([132, -132, -132, -132, 132]))]:
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
            vectors.calc_dest_polar_angle(0, 1, 5.0, 5.0)
        return

    def test_calc_dest_polar_angle_bad_vect_quad(self):
        """Test the north azimuth angle calculation with undefined vect quad."""

        self.comp = "data vector quadrant is undefined"
        with self.assertRaisesRegex(ValueError, self.comp):
            vectors.calc_dest_polar_angle(1, 0, 5.0, 5.0)
        return

    def test_calc_dest_vec_sign_float(self):
        """Test the vector sign calculation for float inputs."""
        # Set the expected vector sign output
        self.comp = {
            1: {1: {'north': 1, 'east': -1}, 2: {'north': -1, 'east': -1},
                3: {'north': -1, 'east': -1}, 4: {'north': 1, 'east': 1}},
            2: {1: {'north': -1, 'east': 1}, 2: {'north': 1, 'east': 1},
                3: {'north': 1, 'east': -1}, 4: {'north': -1, 'east': 1}},
            3: {1: {'north': -1, 'east': 1}, 2: {'north': -1, 'east': 1},
                3: {'north': 1, 'east': 1}, 4: {'north': -1, 'east': -1}},
            4: {1: {'north': -1, 'east': -1}, 2: {'north': -1, 'east': -1},
                3: {'north': -1, 'east': 1}, 4: {'north': 1, 'east': -1}}}

        # Cycle through each of the pole quadrants
        for pole_quad in np.arange(1, 5, 1):
            # Cycle through each of the vector quadrants
            for vect_quad in np.arange(1, 5, 1):
                for nout in [True, False]:
                    for eout in [True, False]:
                        with self.subTest(
                                pole_quad=pole_quad, vect_quad=vect_quad,
                                north_out=nout, east_out=eout):
                            # Get the output
                            self.out = vectors.calc_dest_vec_sign(
                                pole_quad, vect_quad, 5.0, self.pole_ang[-1],
                                north=nout, east=eout)

                            # Update the comparison data
                            sub_comp = dict(self.comp[pole_quad][vect_quad])
                            if not nout:
                                sub_comp['north'] = 0
                            if not eout:
                                sub_comp['east'] = 0

                            # Test the integer quadrant assignment
                            self.assertDictEqual(self.out, sub_comp)
        return

    def test_calc_dest_vec_sign_array(self):
        """Test the vector sign calculation for array inputs."""
        # Set the expected vector sign input and output
        self.pole_quad = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        self.vect_quad = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        self.pole_lt = list(np.linspace(0.0, 360.0, num=len(self.pole_quad)))
        self.pole_ang = list(self.pole_ang) + list(self.pole_ang)
        self.comp = {
            'north': np.array([1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1,
                               1, 1]),
            'east': np.array([1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1,
                              1, 1])}

        # Cycle through list-like or array-like inputs
        for is_array in [True, False]:
            # Set the function arguements
            args = [self.pole_quad, self.vect_quad, self.pole_lt, self.pole_ang]

            if is_array:
                for i, arg in enumerate(args):
                    args[i] = np.asarray(arg)

            # Cycle through different output types
            for nout in [True, False]:
                for eout in [True, False]:
                    # Update the comparison data
                    sub_comp = {key: np.array(self.comp[key])
                                for key in self.comp.keys()}
                    if not nout:
                        sub_comp['north'] *= 0
                    if not eout:
                        sub_comp['east'] *= 0

                    with self.subTest(is_array=is_array, north=nout, east=eout):
                        # Get the output
                        self.out = vectors.calc_dest_vec_sign(
                            *args, north=nout, east=eout)

                        # Test the vector sign assignment
                        self.assertListEqual(list(self.out.keys()),
                                             list(sub_comp.keys()))
                        for key in self.out.keys():
                            self.assertTrue(
                                np.all(self.out[key] == sub_comp[key]),
                                msg="{:}: {:} != {:}".format(
                                    key, self.out[key], sub_comp[key]))
        return

    def test_calc_dest_vec_sign_mixed(self):
        """Test the vector sign calculation for mixed inputs."""

        # Cycle through the mixed inputs
        for args, self.comp in [([1, [2, 3], [0.0, 24.0], 0.0],
                                 {'north': np.array([1, -1]),
                                  'east': np.array([-1, -1])}),
                                ([[1, 4], 4, [72, 360], [91.720246, 143.11957]],
                                 {'north': np.array([1, 1]),
                                  'east': np.array([1, 1])}),
                                ([[1, 2], [3, 4], 5.0, [91.720246, 143.11957]],
                                 {'north': np.array([-1, -1]),
                                  'east': np.array([-1, 1])}),
                                ([[3, 4], [2, 1], [0.0, 90.0], 10.0],
                                 {'north': np.array([1, 1]),
                                  'east': np.array([1, -1])})]:
            with self.subTest(args=args):
                # Get the output
                self.out = vectors.calc_dest_vec_sign(*args, north=True,
                                                      east=True)

                # Test the vector sign assignment
                self.assertListEqual(list(self.out.keys()),
                                     list(self.comp.keys()))
                for key in self.out.keys():
                    self.assertTrue(np.all(self.out[key] == self.comp[key]),
                                    msg="{:}: {:} != {:}".format(
                                        key, self.out[key], self.comp[key]))
        return

    def test_calc_dest_vec_sign_bad_pole_quad(self):
        """Test the vector sign calculation with an undefined pole quad."""

        self.comp = "destination coordinate pole quadrant is undefined"
        with self.assertRaisesRegex(ValueError, self.comp):
            vectors.calc_dest_vec_sign(0, 1, 5.0, 5.0)
        return

    def test_calc_dest_vec_sign_bad_vect_quad(self):
        """Test the vector sign calculation with an undefined vect quad."""

        self.comp = "data vector quadrant is undefined"
        with self.assertRaisesRegex(ValueError, self.comp):
            vectors.calc_dest_vec_sign(1, 0, 5.0, 5.0)
        return

    def test_adjust_vector_float(self):
        """Test the vector adjustment with float inputs."""
        self.comp = (np.array([2.0, 0.0, -1.0, -1.0896077, -4.75018438, 0.0,
                               1.41421332, 0.19974281]),
                     np.array([3.0, 1.0, 3.5, -2.96862848, -1.29836371, 0.0,
                               0.000820721509, -1.40003672]),
                     np.array([1, 1, 1, 1, 1, 1, 1, 1]))

        # Cycle through all possible inputs
        for i, vect_lt in enumerate(self.lt):
            args = [vect_lt, self.lat[i], self.vect_n[i], self.vect_e[i], 1.0,
                    self.vect_quad[i], self.pole_lt[i], self.pole_lat[i],
                    self.pole_ang[i], self.pole_quad[i]]

            with self.subTest(args=args):
                # Get the output
                self.out = vectors.adjust_vector(*args)

                # Test the output
                self.assertTrue(len(self.out), 3)
                self.assertAlmostEqual(float(self.out[0]), self.comp[0][i],
                                       msg="bad north vector value")
                self.assertAlmostEqual(float(self.out[1]), self.comp[1][i],
                                       msg="bad east vector value")
                self.assertAlmostEqual(float(self.out[2]), self.comp[2][i],
                                       msg="bad verticle vector value")
        return

    def test_adjust_vector_array(self):
        """Test the vector adjustment with array-like inputs."""
        self.comp = (np.array([-2.0, 0.0, -1.0, -1.0896077, -4.75018438, 0.0,
                               1.41421332, 0.19974281]),
                     np.array([-3.0, 1.0, 3.5, -2.96862848, -1.29836371, 0.0,
                               0.000820721509, -1.40003672]),
                     np.array([1, 1, 1, 1, 1, 1, 1, 1]))

        # Adjust the inputs to cover all lines accessable by array-like input
        self.lat[0] = self.pole_lat[0] + 30.0

        # Cycle through list-like or array-like inputs
        for is_array in [True, False]:
            # Set the function arguements
            args = [self.lt, self.lat, self.vect_n, self.vect_e,
                    np.ones(shape=len(self.lt)), self.vect_quad, self.pole_lt,
                    self.pole_lat, self.pole_ang, self.pole_quad]

            if is_array:
                for i, arg in enumerate(args):
                    args[i] = np.asarray(arg)

            with self.subTest(is_array=is_array):
                # Get the output
                self.out = vectors.adjust_vector(*args)

                # Test the output
                self.assertTrue(len(self.out), 3)
                self.assertTrue(np.all(abs(self.out[0] - self.comp[0]) < 1e-5),
                                msg="bad north values: {:} != {:}".format(
                                    self.out[0], self.comp[0]))
                self.assertTrue(np.all(abs(self.out[1] - self.comp[1]) < 1e-5),
                                msg="bad east values: {:} != {:}".format(
                                    self.out[1], self.comp[1]))
                self.assertTrue(np.all(abs(self.out[2] - self.comp[2]) < 1e-5),
                                msg="bad vertical values: {:} != {:}".format(
                                    self.out[2], self.comp[2]))
        return

    def test_adjust_vector_mixed(self):
        """Test the vector adjustment with mixed inputs."""
        # Set the base input and output
        args = [np.asarray(self.lt), np.asarray(self.lat),
                np.asarray(self.vect_n), np.asarray(self.vect_e),
                np.ones(shape=len(self.lt)), np.asarray(self.vect_quad),
                np.asarray(self.pole_lt), np.asarray(self.pole_lat),
                np.asarray(self.pole_ang), np.asarray(self.pole_quad)]

        self.comp = (np.array([2.0, 0.0, -1.0, -1.0896077, -4.75018438, 0.0,
                               1.41421332, 0.19974281]),
                     np.array([3.0, 1.0, 3.5, -2.96862848, -1.29836371, 0.0,
                               0.000820721509, -1.40003672]),
                     np.array([1, 1, 1, 1, 1, 1, 1, 1]))

        # Cycle through compinations of float/array inputs
        for ipos, ival, islice in [(0, 0, [0, -1]), (1, 0, [0, 1, 2]),
                                   (2, 2, [2, 7]), (3, 1, [1, -1]),
                                   (4, 0, slice(None)), (5, 0, [0, 1, 5]),
                                   (6, 0, [0, 6]), (7, 5, [5, 6]),
                                   (8, 0, [0, 1, 2]), (9, 3, [3, 5])]:
            # Assign the mixed inputs
            vect_args = list()
            for i, arg in enumerate(args):
                if i == ipos:
                    vect_args.append(arg[ival])
                else:
                    vect_args.append(arg[islice])

            with self.subTest(args=vect_args):
                # Get the output
                self.out = vectors.adjust_vector(*vect_args)

                # Test the output
                self.assertTrue(len(self.out), 3)
                self.assertTrue(
                    np.all(abs(self.out[0] - self.comp[0][islice]) < 1e-5),
                    msg="bad north values: {:} != {:}".format(
                        self.out[0], self.comp[0][islice]))
                self.assertTrue(
                    np.all(abs(self.out[1] - self.comp[1][islice]) < 1e-5),
                    msg="bad east values: {:} != {:}".format(
                        self.out[1], self.comp[1][islice]))
                self.assertTrue(
                    np.all(abs(self.out[2] - self.comp[2][islice]) < 1e-5),
                    msg="bad vertical values: {:} != {:}".format(
                        self.out[2], self.comp[2][islice]))
        return
