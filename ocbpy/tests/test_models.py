#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DOI: 10.5281/zenodo.1179230
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the boundaries.models functions."""

import numpy as np
import unittest

from ocbpy.boundaries import models


class TestStarkovModel(unittest.TestCase):
    """"Unit tests for the Starkov 1994 routines."""

    def setUp(self):
        """Initialize the test case by copying over necessary files."""
        self.mlt = np.arange(0, 24, 1)
        self.coeff_out = {
            'A0': {'ocb': -.07, 'eab': 1.16, 'diffuse': 3.44},
            'A1': {'ocb': -10.06, 'eab': -9.59, 'diffuse': -2.41},
            'alpha1': {'ocb': -6.61, 'eab': -2.22, 'diffuse': -1.68},
            'A2': {'ocb': -4.44, 'eab': -12.07, 'diffuse': -0.74},
            'alpha2': {'ocb': 6.37, 'eab': -23.98, 'diffuse': 8.69},
            'A3': {'ocb': -3.77, 'eab': -6.56, 'diffuse': -2.12},
            'alpha3': {'ocb': -4.48, 'eab': -20.07, 'diffuse': 8.61}}
        self.al = [-1, -500]
        self.max_lat = {'ocb': [11.66543814, 19.2853977],
                        'eab': [24.1150303, 29.14429888],
                        'diffuse': [7.00627994, 34.72168093]}
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.mlt, self.coeff_out, self.al, self.max_lat
        return

    def test_coeff_construction(self):
        """Test coefficient calculation for an AL of -1."""

        for coeff in self.coeff_out.keys():
            for bnd in self.coeff_out[coeff].keys():
                with self.subTest(coeff=coeff, bnd=bnd):
                    # Calculate the coefficient value
                    out = models.starkov_coefficient_values(
                        self.al[0], coeff, bnd)

                    # Compare the output
                    self.assertEqual(out, self.coeff_out[coeff][bnd])
        return

    def test_coeff_bad_coeff(self):
        """Test a KeyError is raised for an unknown coeffcient name."""
        coeff = "not a coefficient"
        with self.assertRaisesRegex(KeyError, coeff):
            models.starkov_coefficient_values(self.al[0], coeff,
                                              list(self.max_lat.keys())[0])
        return

    def test_coeff_bad_bnd(self):
        """Test a KeyError is raised for an unknown boundary name."""
        bound = "not a boundary"
        with self.assertRaisesRegex(KeyError, bound):
            models.starkov_coefficient_values(
                self.al[0], list(self.coeff_out.keys())[0], bound)
        return

    def test_bound_loc_array(self):
        """Test the expected boundary location across an MLT array."""
        # Cycle through low and high AL values
        for ia, in_al in enumerate(self.al):
            for bnd in self.max_lat.keys():
                with self.subTest(al=in_al, bnd=bnd):
                    lat = models.starkov_auroral_boundary(
                        self.mlt, al=in_al, bnd=bnd)

                    # Test the output latitude shape and values
                    self.assertTupleEqual(self.mlt.shape, lat.shape)
                    self.assertGreaterEqual(min(lat), 0)
                    self.assertAlmostEqual(max(lat), self.max_lat[bnd][ia])

        return

    def test_bound_loc_float(self):
        """Test the expected boundary location across an MLT value."""
        # Cycle through low and high AL values
        for ia, in_al in enumerate(self.al):
            for bnd in self.max_lat.keys():
                with self.subTest(al=in_al, bnd=bnd):
                    lat = models.starkov_auroral_boundary(
                        self.mlt[0], al=in_al, bnd=bnd)

                    # Test the output latitude shape and values
                    self.assertTrue(isinstance(lat, float))
                    self.assertGreaterEqual(lat, 0)
                    self.assertLessEqual(lat, self.max_lat[bnd][ia])

        return
