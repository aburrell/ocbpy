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
                      9: [41.81, 38.85, 35.62, 39.03, 38.90, 38.29, 37.53,
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
        # radial offset of zero.  It should be constrained within +/-pi
        self.assertGreaterEqual(phi_cent, -np.pi)
        self.assertLessEqual(phi_cent, np.pi)
        self.assertAlmostEqual(r_cent, 0.0)
        self.assertAlmostEqual(radius, 1.0)
        self.assertAlmostEqual(r_err, 0.0)
        return


class TestCHAMPModel(unittest.TestCase):
    """"Unit tests for the CH-Aurora-2014 routines."""

    def setUp(self):
        """Initialize the test case by setting some values to test against."""
        self.mlt = np.arange(0, 24, 1)
        self.coeff_out = {'semix': {'ocb': {1: 12.813, -1: 13.251},
                                    'eab': {1: 18.861, -1: 18.559}},
                          'semiy': {'ocb': {1: 9.5486, -1: 11.605},
                                    'eab': {1: 20.562, -1: 19.549}},
                          'x0': {'ocb': {1: 4.5175, -1: 4.2526},
                                 'eab': {1: 4.1263, -1: 3.6946}},
                          'y0': {'ocb': {1: -0.39316, -1: -1.1330},
                                 'eab': {1: -0.32637, -1: -0.60436}},
                          'phi0': {'ocb': {1: -0.1489778, -1: 0.0646644},
                                   'eab': {1: -0.055074, -1: -0.155048}}}
        self.em = [0, 10.5]
        self.iobs = [0, 12]
        self.obs_mlt = self.mlt[self.iobs]
        self.obs_colat = {'ocb': np.array([18.0, 8.0]),
                          'eab': np.array([25.0, 16.0])}
        self.max_lat = {'ocb': {1: [17.31668454, 20.1346482],
                                -1: [17.5881323, 20.741303775]},
                        'eab': {1: [23.052935, 31.313414],
                                -1: [22.354591, 29.69814346]}}
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.mlt, self.coeff_out, self.em, self.max_lat, self.obs_mlt
        del self.obs_colat, self.iobs
        return

    def test_coeff_construction(self):
        """Test coefficient calculation for an Em of 0."""

        for coeff in self.coeff_out.keys():
            for bnd in self.coeff_out[coeff].keys():
                for hemi in [1, -1]:
                    with self.subTest(coeff=coeff, bnd=bnd, hemi=hemi):
                        # Calculate the coefficient value
                        out = models.ch_aurora_2014_coefficient_values(
                            self.em[0], bnd, hemi)

                        # Compare the output
                        for ic, coeff in enumerate(['semix', 'semiy', 'x0',
                                                    'y0']):
                            self.assertEqual(
                                out[ic], self.coeff_out[coeff][bnd][hemi],
                                msg="{:s} does not match".format(coeff))

                        # Phi0 has been converted to radians, so equality
                        # will not be exact. Use significance from paper
                        self.assertAlmostEqual(
                            out[-1], self.coeff_out['phi0'][bnd][hemi],
                            places=5, msg="phi0 does not match")
        return

    def test_coeff_bad_hemi(self):
        """Test a KeyError is raised for an unknown hemisphere."""
        hemi = "north"
        with self.assertRaisesRegex(KeyError, hemi):
            models.ch_aurora_2014_coefficient_values(
                self.em[0], list(self.max_lat.keys())[0], hemi)
        return

    def test_coeff_bad_bnd(self):
        """Test a KeyError is raised for an unknown boundary name."""
        bound = "not a boundary"
        with self.assertRaisesRegex(KeyError, bound):
            models.ch_aurora_2014_coefficient_values(self.em[0], bound, 1)
        return

    def test_bound_loc_array(self):
        """Test the expected boundary location across an MLT array."""
        # Get the boundary keys
        bnds = list(self.max_lat.keys())

        # Cycle through low and high Em values
        for ie, in_em in enumerate(self.em):
            for hemi in self.max_lat[bnds[0]].keys():
                with self.subTest(em=in_em, hemi=hemi):
                    lats = {
                        bnd: models.ch_aurora_2014_boundary(
                            self.mlt, em=in_em, bnd=bnd, hemi=hemi)
                        for bnd in bnds}

                    # Test the output latitude shape and values
                    for bnd in bnds:
                        self.assertTupleEqual(self.mlt.shape, lats[bnd].shape)
                        self.assertAlmostEqual(max(lats[bnd]),
                                               self.max_lat[bnd][hemi][ie],
                                               places=5)

                    # Test that the OCB is greater than zero and the EAB is
                    # greater than the OCB
                    self.assertGreaterEqual(min(lats['ocb']), 0)
                    self.assertTrue(np.all(lats['eab'] > lats['ocb']))

        return

    def test_bound_loc_float(self):
        """Test the expected boundary location across an MLT value."""
        # Cycle through low and high Em values
        for ie, in_em in enumerate(self.em):
            for bnd in self.max_lat.keys():
                for hemi in self.max_lat[bnd].keys():
                    with self.subTest(em=in_em, bnd=bnd, hemi=hemi):
                        lat = models.ch_aurora_2014_boundary(
                            self.mlt[0], em=in_em, bnd=bnd, hemi=hemi)

                        # Test the output latitude shape and values
                        self.assertTrue(isinstance(lat, float))
                        self.assertGreaterEqual(lat, 0)
                        self.assertLessEqual(lat, self.max_lat[bnd][hemi][ie])

        return

    def test_bound_assim(self):
        """Test the expected assimilated boundary location."""

        # Cycle through low and high Em values
        for ie, in_em in enumerate(self.em):
            for bnd in self.max_lat.keys():
                for hemi in self.max_lat[bnd].keys():
                    with self.subTest(em=in_em, bnd=bnd, hemi=hemi):
                        mlat = models.ch_aurora_2014_boundary(
                            self.mlt, em=in_em, bnd=bnd, hemi=hemi)
                        alat = models.ch_aurora_2014_boundary(
                            self.mlt, em=in_em, bnd=bnd, hemi=hemi,
                            obs_mlt=self.obs_mlt, obs_colat=self.obs_colat[bnd])

                        # Test the output latitude shape
                        self.assertTupleEqual(self.mlt.shape, alat.shape)

                        # Model and assimilation should differ
                        self.assertGreater(abs(alat - mlat).min(), 1.0e-2)

                        # Assimilated output should be closer to the provided
                        # data points than the original model in at least one
                        # of the assimilated locations
                        self.assertTrue(
                            np.any(abs(alat[self.iobs] - self.obs_colat[bnd])
                                   < abs(mlat[self.iobs]
                                         - self.obs_colat[bnd])))
        return
