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
        """Initialize the test case by setting some values to test against."""
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


class TestGussenhovelModel(unittest.TestCase):
    """"Unit tests for the Gussenhoven 1983 routines."""

    def setUp(self):
        """Initialize the test case."""
        self.mlt = np.arange(0, 24, 1)
        self.bad_mlt = [2, 3, 13, 14]
        self.colat = {0: [23.9, 24.9, 22.3, 22.2, 21.8, 21.1, 20.7, 20.5, 20.4,
                          19.9, 20.6, 19.1, 18.4, 18.9, 18.8, 19.6, 20.6, 21.4,
                          22.1, 22.2],
                      9: [41.81, 38.85, 35.62, 39.03, 38.9 , 38.29, 37.53,
                          35.53, 33.09, 31.15, 28.16, 26.39, 29.92, 30.69,
                          34.46, 36.07, 37.61, 38.14, 38.12, 40.83]}
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.mlt, self.bad_mlt, self.colat
        return

    def test_gussenhoven_colatitudes_good(self):
        """Test the colat calculation for MLTs with solutions"""

        for kp in self.colat.keys():
            with self.subTest(kp=kp):
                # Calculate the colatitude values with default kwargs
                out_lat, _ = models.gussenhoven_colatitudes(kp)

                # Compare the output
                self.assertEqual(len(out_lat), len(self.colat[kp]))
                diff_lat = out_lat - np.asarray(self.colat[kp])
                self.assertLess(sum(diff_lat), 1.0e-3)
        return

    def test_gussenhoven_colatitudes_bad(self):
        """Test the colat calculation for MLTs without solutions."""
        # Cycle through each Kp
        for kp in self.colat.keys():
            with self.subTest(kp=kp):
                # Calculate the colatitude values with default kwargs
                out_lat, out_mlt = models.gussenhoven_colatitudes(
                    kp, mlt_inds=self.bad_mlt)

                # Compare the output
                self.assertTrue(np.isnan(out_lat).all())
                self.assertListEqual(list(out_mlt), self.bad_mlt)
        return

    def test_gussenhoven_colatitudes_bad_close(self):
        """Test the colat calculation for MLTs at nearest solutions."""
        # Reshape the MLT to exclude the bad MLT values
        self.mlt = list(self.mlt)
        for imlt in self.bad_mlt:
            self.mlt.pop(self.mlt.index(imlt))

        # Cycle through each Kp
        for kp in self.colat.keys():
            with self.subTest(kp=kp):
                # Calculate the colatitude values with default kwargs
                out_lat, out_mlt = models.gussenhoven_colatitudes(
                    kp, mlt_inds=self.bad_mlt, closest=True)

                # Compare the output
                self.assertFalse(np.isnan(out_lat).any())
                self.assertEqual(len(out_mlt), len(self.bad_mlt))

                for i, imlt in enumerate(out_mlt):
                    self.assertFalse(imlt in self.bad_mlt)
                    self.assertAlmostEqual(
                        out_lat[i], self.colat[kp][self.mlt.index(imlt)])
        return

    def test_bad_model(self):
        """Test a ValueError is raised for an unknown model type."""
        model = "not a model"
        with self.assertRaisesRegex(ValueError, model):
            models.gussenhoven_equatorward_auroral_boundary(
                self.mlt, model=model)
        return

    def test_gussenhoven_eab_binned(self):
        """Test the EAB for MLTs with binned solutions."""
        # Cycle through each Kp
        for kp in self.colat.keys():
            with self.subTest(kp=kp):
                # Calculate the colatitude values with default kwargs
                out_lat = models.gussenhoven_equatorward_auroral_boundary(
                    self.mlt, kp=kp, model='binned')

                # Compare the output
                self.assertEqual(len(out_lat[~np.isnan(out_lat)]),
                                 len(self.colat[kp]))
                self.assertEqual(len(out_lat[np.isnan(out_lat)]),
                                 len(self.bad_mlt))
                diff_lat = out_lat[~np.isnan(out_lat)] - np.asarray(
                    self.colat[kp])
                self.assertLess(sum(diff_lat), 1.0e-3)
        return

    def test_gussenhoven_eab_closest(self):
        """Test the EAB for MLTs with closest value solutions."""
        # Cycle through each Kp
        for kp in self.colat.keys():
            with self.subTest(kp=kp):
                # Calculate the colatitude values with default kwargs
                out_lat = models.gussenhoven_equatorward_auroral_boundary(
                    self.mlt, kp=kp, model='closest')

                # Compare the output
                self.assertEqual(len(out_lat), len(self.mlt))

                j = 0
                for i, imlt in enumerate(self.mlt):
                    if imlt in self.bad_mlt:
                        try:
                            j -= 1
                            self.assertAlmostEqual(
                                out_lat[i], self.colat[kp][j])
                            j += 1
                        except AssertionError:
                            j += 1
                            self.assertAlmostEqual(
                                out_lat[i], self.colat[kp][j],
                                msg="failed at {:} MLT with j={:}".format(
                                    imlt, j))
                    else:
                        self.assertAlmostEqual(
                            out_lat[i], self.colat[kp][j],
                            msg="failed at {:} MLT with j={:}".format(imlt, j))
                        j += 1
        return

    def test_gussenhoven_eab_circle(self):
        """Test the EAB for MLTs with circle fit solutions."""
        # Cycle through each Kp
        for kp in self.colat.keys():
            with self.subTest(kp=kp):
                # Calculate the colatitude values with default kwargs
                out_lat = models.gussenhoven_equatorward_auroral_boundary(
                    self.mlt, kp=kp, model='circle')

                # Compare the output
                self.assertEqual(len(out_lat), len(self.mlt))

                j = 0
                for i, imlt in enumerate(self.mlt):
                    if imlt in self.bad_mlt:
                        try:
                            j -= 1
                            self.assertLessEqual(
                                abs(out_lat[i] - self.colat[kp][j]), 5.0)
                            j += 1
                        except AssertionError:
                            j += 1
                            self.assertLessEqual(
                                abs(out_lat[i] - self.colat[kp][j]), 5.0,
                                msg="failed at {:} MLT with j={:}".format(
                                    imlt, j))
                    else:
                        self.assertLessEqual(
                            abs(out_lat[i] - self.colat[kp][j]), 5.0,
                            msg="failed at {:} MLT with j={:}".format(imlt, j))
                        j += 1
        return



class TestFits(unittest.TestCase):
    """"Unit tests for the fitting routines."""

    def setUp(self):
        """Initialize the test case by setting some values to test against."""
        self.mlt = np.arange(0, 24, 1)
        self.rvals = np.ones(shape=self.mlt.shape)
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.mlt, self.rvals
        return

    def test_circle_fit(self):
        """Test the circle fitting to a unit circle."""
        # Run the fitting routine
        phi_cent, r_cent, radius, r_err = models.circle_fit(
            self.mlt, self.rvals)

        # Test the output.  The anglular offset can be any value with a
        # radial offset of zero.  It should be constrained within 0-2pi
        self.assertGreaterEqual(phi_cent, 0.0)
        self.assertLessEqual(phi_cent, 2.0 * np.pi)
        self.assertAlmostEqual(r_cent, 0.0)
        self.assertAlmostEqual(radius, 1.0)
        self.assertAlmostEqual(r_err, 0.0)
        return
