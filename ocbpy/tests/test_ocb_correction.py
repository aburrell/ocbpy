#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the OCB correction functions."""

import numpy as np
import unittest

from ocbpy import ocb_correction as ocb_cor


class TestOCBCorrectionFailure(unittest.TestCase):
    """Unit tests for correction failure evaluations."""

    def setUp(self):
        """Set up the test runs."""
        self.mlt = 12.0
        self.bad_kwarg = 'bad_kwarg'
        self.functions = {'elliptical': ocb_cor.elliptical,
                          'harmonic': ocb_cor.harmonic}
        self.bound = None
        return

    def tearDown(self):
        """Tear down the test runs."""
        del self.mlt, self.bad_kwarg, self.functions, self.bound
        return

    def test_instrument_failure(self):
        """Test failure when an unknown instrument is provided."""

        for self.bound in self.functions.keys():
            with self.subTest(bound=self.bound):
                msg = "no {:s} correction for".format(self.bound)
                with self.assertRaisesRegex(ValueError, msg):
                    self.functions[self.bound](self.mlt,
                                               instrument=self.bad_kwarg)
        return

    def test_method_failure(self):
        """Test failure when an unknown method is provided."""
        msg = "unknown coefficient computation method"
        for self.bound in self.functions.keys():
            with self.subTest(bound=self.bound):
                with self.assertRaisesRegex(ValueError, msg):
                    self.functions[self.bound](self.mlt, method=self.bad_kwarg)
        return


class TestOCBCorrection(unittest.TestCase):
    """Unit tests for the boundary correction functions."""

    def setUp(self):
        """Set up test runs."""
        self.functions = {'circular': ocb_cor.circular,
                          'elliptical': ocb_cor.elliptical,
                          'harmonic': ocb_cor.harmonic}
        self.mlt = np.arange(0.0, 24.0, 12.0)
        self.def_results = {'circular': np.zeros(shape=self.mlt.shape),
                            'elliptical': np.array([-2.097939334919742,
                                                    -4.194630407939707]),
                            'harmonic': np.array([-1.5821694271422921,
                                                  -3.4392638193624325])}
        self.gaus_results = {'elliptical': -2.51643691301747,
                             'harmonic': -2.293294645880221}
        self.image_results = {'si12': {'ocb': np.array([0.67103129,
                                                        -0.10084541]),
                                       'eab': np.array([0.31691853,
                                                        2.68970685])},
                              'si13': {'ocb': np.array([0.71521016,
                                                        -0.86843608]),
                                       'eab': np.array([1.55220967,
                                                        -0.19812977])},
                              'wic': {'ocb': np.array([0.83660688,
                                                       -1.62097642]),
                                      'eab': np.array([1.89268527,
                                                       0.13983824])}}
        self.bound = None
        return

    def tearDown(self):
        """Clean up after each test."""
        del self.mlt, self.functions, self.def_results, self.gaus_results
        del self.bound, self.image_results
        return

    def test_default_float(self):
        """Test the boundary functions using a float and function defaults."""
        for self.bound in self.functions.keys():
            with self.subTest(bound=self.bound):
                self.assertEqual(self.functions[self.bound](self.mlt[0]),
                                 self.def_results[self.bound][0])
        return

    def test_default_arr(self):
        """Test the boundary functions using an array and function defaults."""
        for self.bound in self.functions.keys():
            with self.subTest(bound=self.bound):
                self.assertTrue(np.all(
                    abs(self.functions[self.bound](self.mlt)
                        - self.def_results[self.bound]) < 1.0e-7))
        return

    def test_circular_offset(self):
        """Test the circular boundary function with an offset."""
        self.assertEqual(ocb_cor.circular(self.mlt[0], r_add=1.0), 1.0)
        return

    def test_gauss_method(self):
        """Test the boundary functions using an array and function defaults."""
        for self.bound in self.gaus_results.keys():
            with self.subTest(bound=self.bound):
                self.assertAlmostEqual(
                    self.functions[self.bound](self.mlt[0], method='gaussian'),
                    self.gaus_results[self.bound])
        return

    def test_image_harmonic(self):
        """Test the IMAGE harmonic correction functions."""
        for self.bound in self.image_results.keys():
            for method in self.image_results[self.bound].keys():
                with self.subTest(bound=self.bound, method=method):
                    self.assertTrue(np.all(
                        abs(self.functions["harmonic"](
                            self.mlt, instrument=self.bound, method=method)
                            - self.image_results[self.bound][method]) < 1.0e-7))
        return
