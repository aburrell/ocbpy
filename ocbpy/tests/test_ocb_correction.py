#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
""" Tests the ocboundary class and functions
"""

import numpy as np
from sys import version_info
import unittest

from ocbpy import ocb_correction as ocb_cor


class TestOCBCorrectionFailure(unittest.TestCase):
    def setUp(self):
        """ Set up the test runs """
        self.mlt = 12.0
        self.bad_kwarg = 'bad_kwarg'
        self.functions = {'elliptical': ocb_cor.elliptical,
                          'harmonic': ocb_cor.harmonic}
        self.bound = None

    def tearDown(self):
        """ Tear down the test runs """
        del self.mlt, self.bad_kwarg, self.functions, self.bound

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_instrument_failure(self):
        """ Test failure when an unknown instrument is provided """

        for self.bound in self.functions.keys():
            with self.subTest(bound=self.bound):
                with self.assertRaises(ValueError):
                    self.functions[self.bound](self.mlt,
                                               instrument=self.bad_kwarg)

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_elliptical_instrument_failure(self):
        """ Test failure in elliptical when an unknown instrument is provided
        """
        self.bound = 'elliptical'
        with self.assertRaises(ValueError):
            self.functions[self.bound](self.mlt, instrument=self.bad_kwarg)

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_harmonic_instrument_failure(self):
        """ Test failure in harmonic when an unknown instrument is provided"""

        self.bound = 'harmonic'
        with self.assertRaises(ValueError):
            self.functions[self.bound](self.mlt, instrument=self.bad_kwarg)

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_method_failure(self):
        """ Test failure when an unknown method is provided """
        for self.bound in self.functions.keys():
            with self.subTest(bound=self.bound):
                with self.assertRaises(ValueError):
                    self.functions[self.bound](self.mlt, method=self.bad_kwarg)

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_elliptical_method_failure(self):
        """ Test failure in elliptical when an unknown method is provided"""
        self.bound = 'elliptical'
        with self.assertRaises(ValueError):
            self.functions[self.bound](self.mlt, method=self.bad_kwarg)

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_harmonic_method_failure(self):
        """ Test failure in harmonic when an unknown method is provided"""
        self.bound = 'harmonic'
        with self.assertRaises(ValueError):
            self.functions[self.bound](self.mlt, method=self.bad_kwarg)


class TestOCBCorrection(unittest.TestCase):
    def setUp(self):
        """ Set up test runs """
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
        self.bound = None

    def tearDown(self):
        """ Clean up after each test """
        del self.mlt, self.functions, self.def_results, self.gaus_results
        del self.bound

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_default_float(self):
        """ Test the boundary functions using a float and function defaults"""
        for self.bound in self.functions.keys():
            with self.subTest(bound=self.bound):
                self.assertEqual(self.functions[self.bound](self.mlt[0]),
                                 self.def_results[self.bound][0])

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_circular_default_float(self):
        """ Test the default circular boundary function with a float"""
        self.bound = 'circular'
        self.assertEqual(self.functions[self.bound](self.mlt[0]),
                         self.def_results[self.bound][0])

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_ampere_harmonic_float(self):
        """ Test the default harmonic boundary function for a value"""
        self.bound = 'harmonic'
        self.assertEqual(self.functions[self.bound](self.mlt[0]),
                         self.def_results[self.bound][0])

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_ampere_elliptical_float(self):
        """ Test the default elliptical boundary function for a value"""
        self.bound = 'elliptical'
        self.assertEqual(self.functions[self.bound](self.mlt[0]),
                         self.def_results[self.bound][0])

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_default_arr(self):
        """ Test the boundary functions using an array and function defaults"""
        for self.bound in self.functions.keys():
            with self.subTest(bound=self.bound):
                self.assertTrue(np.all(
                    abs(self.functions[self.bound](self.mlt)
                        - self.def_results[self.bound]) < 1.0e-7))

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_circular_default_arr(self):
        """ Test the default circular boundary function with an array"""
        self.bound = 'circular'
        self.assertTrue(np.all(self.functions[self.bound](self.mlt)
                               == self.def_results[self.bound]))

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_ampere_elliptical_arr(self):
        """ Test the default elliptical boundary function for an array"""
        self.bound = 'elliptical'
        self.assertTrue(np.all(abs(self.functions[self.bound](self.mlt)
                                   - self.def_results[self.bound]) < 1.0e-7))

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_ampere_harmonic_arr(self):
        """ Test the default harmonic boundary function for an array"""
        self.bound = 'harmonic'
        self.assertTrue(np.all(abs(self.functions[self.bound](self.mlt)
                                   - self.def_results[self.bound]) < 1.0e-7))

    def test_circular_offset(self):
        """ Test the circular boundary function with an offset """
        self.assertEqual(ocb_cor.circular(self.mlt[0], r_add=1.0), 1.0)

    @unittest.skipIf(version_info.major == 2,
                     'Python 2.7 does not support subTest')
    def test_gauss_method(self):
        """ Test the boundary functions using an array and function defaults"""
        for self.bound in self.gaus_results.keys():
            with self.subTest(bound=self.bound):
                self.assertAlmostEqual(
                    self.functions[self.bound](self.mlt[0], method='gaussian'),
                    self.gaus_results[self.bound])

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_ampere_harmonic_gaussian(self):
        """ Test the gaussian harmonic boundary function """
        self.bound = 'harmonic'
        self.assertAlmostEqual(self.functions[self.bound](self.mlt[0],
                                                          method="gaussian"),
                               self.gaus_results[self.bound])

    @unittest.skipIf(version_info.major == 3, 'Already tested, remove in 2020')
    def test_elliptical_gaussian(self):
        """ Test the gaussian elliptical boundary function """
        self.bound = 'elliptical'
        self.assertAlmostEqual(self.functions[self.bound](self.mlt[0],
                                                          method="gaussian"),
                               self.gaus_results[self.bound])
