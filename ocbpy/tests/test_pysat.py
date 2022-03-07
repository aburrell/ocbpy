#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the ocbpy.instruments.pysat functions."""

from io import StringIO
import logging
import numpy as np
from os import path
import unittest

import ocbpy

try:
    # Import pysat first to get the correct error message
    import pysat
    import ocbpy.instruments.pysat_instruments as ocb_pysat
    import pandas as pds
    no_pysat = False
except ImportError:
    no_pysat = True


@unittest.skipIf(no_pysat, "pysat not installed, cannot test routines")
class TestPysatUtils(unittest.TestCase):
    """Tests for using ocbpy with pysat data."""

    def setUp(self):
        """Initialize the test environment."""

        # Set the default function values
        self.meta = None
        self.ocb_key = 'ocb_test'
        self.pysat_key = 'dummy1'
        self.notes = None
        self.isvector = False
        self.test_inst = None
        self.ocb = None
        self.added_keys = list()
        self.pysat_keys = list()
        self.arevectors = list()
        self.nkeys = 0
        return

    def tearDown(self):
        """Tear down the testing environment."""
        del self.meta, self.ocb_key, self.pysat_key, self.notes, self.isvector
        del self.test_inst, self.ocb, self.added_keys, self.pysat_keys
        del self.arevectors, self.nkeys
        return

    def eval_ocb_metadata(self):
        """Evaluate new OCB metadata."""

        # Test MetaData exists
        self.assertIsNotNone(self.meta, msg="No meta data added")

        # Tests for MetaData
        self.assertIn(self.ocb_key, list(self.meta.keys()))
        if self.pysat_key is not None:
            self.assertIn(self.pysat_key, list(self.meta.keys()))

        # Test the fill value
        self.assertTrue(
            np.isnan(self.meta[self.ocb_key][self.meta.labels.fill_val]),
            msg="{:s} fill value is {:s}, not np.nan".format(
                repr(self.ocb_key),
                repr(self.meta[self.ocb_key][self.meta.labels.fill_val])))

        if self.pysat_key is not None and not self.isvector:
            # Test the elements that are identical
            for ll in [self.meta.labels.units, self.meta.labels.min_val,
                       self.meta.labels.max_val, self.meta.labels.fill_val]:
                try:
                    if np.isnan(self.meta[self.pysat_key][ll]):
                        self.assertTrue(np.isnan(self.meta[self.ocb_key][ll]))
                    else:
                        self.assertEqual(self.meta[self.ocb_key][ll],
                                         self.meta[self.pysat_key][ll])
                except TypeError:
                    if len(self.meta[self.ocb_key][ll]) == 0:
                        self.assertEqual(len(self.meta[self.pysat_key][ll]), 0)
                    else:
                        self.assertRegex(self.meta[self.ocb_key][ll],
                                         self.meta[self.pysat_key][ll])

        # Test the elements that have "OCB" appended to the text
        sline = self.meta[self.ocb_key][self.meta.labels.name].split(" ")
        self.assertRegex(sline[0], "OCB")
        if not self.isvector and self.pysat_key is not None:
            self.assertRegex(" ".join(sline[1:]),
                             self.meta[self.pysat_key][self.meta.labels.name])

        # Test the remaining elements
        self.assertEqual(self.meta[self.ocb_key][self.meta.labels.desc].find(
            "Open Closed"), 0)
        if self.notes is not None:
            self.assertRegex(self.meta[self.ocb_key][self.meta.labels.notes],
                             self.notes)
        return

    def test_ocb_added(self):
        """Test if OCB data was added correctly."""
        # If there is no test instrument, test passes
        if self.test_inst is None:
            return

        # For a test instrument, evaluate attributes
        if len(self.arevectors) < len(self.added_keys):
            self.arevectors = [False for okey in self.added_keys]

        self.assertEqual(len(self.added_keys), self.nkeys)

        for i, okey in enumerate(self.added_keys):
            # Test to see that data was added
            self.assertIn(okey, self.test_inst.variables)

            # Test the metadata
            self.meta = self.test_inst.meta
            self.pysat_key = self.pysat_keys[i]
            self.isvector = self.arevectors[i]
            self.ocb_key = okey
            self.eval_ocb_metadata()

            # Test to see that data within the time tolerance of the test OCBs
            # has OCB locations and other data is NaN
            match_data = self.test_inst[okey]
            if self.arevectors[i]:
                mask_data = np.not_equal(match_data, None)
            else:
                mask_data = np.isfinite(match_data)
            match_data = match_data[mask_data]

            self.assertGreater(len(match_data), 0)

            if hasattr(match_data, "index"):
                match_time = match_data.index
            else:
                match_time = pds.to_datetime(match_data['time'].values)

            for ii in match_time:
                check_time = abs(ii - self.ocb.dtime).min().total_seconds()
                self.assertLessEqual(check_time, 600.0)
            if self.arevectors[i]:
                self.assertTrue(isinstance(match_data[0],
                                           ocbpy.ocb_scaling.VectorData))
            elif self.pysat_keys[i] is not None:
                pysat_data = self.test_inst[self.pysat_keys[i]][mask_data]

                # Get the scaling radius
                if hasattr(self.ocb, "r"):
                    rscale = (self.ocb.r / (90.0 - self.ocb.boundary_lat))**2
                else:
                    rscale = (self.ocb.ocb.r
                              / (90.0 - self.ocb.ocb.boundary_lat))**2

                # Evaluate the data
                self.assertGreaterEqual(match_data.min(),
                                        pysat_data.min() * rscale.min())
                self.assertGreaterEqual(pysat_data.max() * rscale.max(),
                                        match_data.max())

        return


@unittest.skipIf(not no_pysat, "pysat installed, cannot test failure")
class TestPysatFailure(unittest.TestCase):
    """Unit tests for the pysat instrument functions without pysat installed."""

    def setUp(self):
        """Initialize setup, none needed."""
        return

    def tearDown(self):
        """Clean up test environment, no teardown needed."""
        return

    def test_import_failure(self):
        """Test pysat import failure."""

        with self.assertRaisesRegex(ImportError, 'unable to load the pysat'):
            import ocbpy.instruments.pysat_instruments as ocb_pysat  # NOQA 401
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatStructure(unittest.TestCase):
    """Unit tests for the pysat instrument functions."""

    def setUp(self):
        """Initialize setup, none needed."""
        return

    def tearDown(self):
        """Clean up test environment, no teardown needed."""
        return

    def test_add_ocb_to_data_defaults(self):
        """Test the add_ocb_to_data function defaults."""
        defaults = ocb_pysat.add_ocb_to_data.__defaults__

        for i in [0, 1, 8]:
            self.assertEqual(len(defaults[i]), 0)

        for i in [2, 3, 4, 6, 10, 11]:
            self.assertIsNone(defaults[i])

        self.assertEqual(defaults[5], 0)
        self.assertRegex(defaults[7], 'default')
        self.assertEqual(defaults[9], 60)
        return

    def test_add_ocb_to_metadata_defaults(self):
        """Test the add_ocb_to_metadata function defaults."""
        defaults = ocb_pysat.add_ocb_to_metadata.__defaults__

        for i in [0, 2]:
            self.assertFalse(defaults[i])

        self.assertEqual(defaults[1], '')
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatMethods(unittest.TestCase):
    """Integration tests for using ocbpy on pysat pandas data."""

    def setUp(self):
        """Initialize the test class."""

        self.test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                                   "test_data", "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb_kw = {"ocbfile": self.test_file,
                       "instrument": "image", "hemisphere": 1}
        self.ocb = ocbpy.OCBoundary(self.test_file, instrument='image',
                                    hemisphere=1)
        self.ocb.rec_ind = 27

        self.test_inst = pysat.Instrument('pysat', 'testing', num_samples=50400,
                                          clean_level='clean',
                                          update_files=True,
                                          file_date_range=pds.date_range(
                                              self.ocb.dtime[0],
                                              self.ocb.dtime[-1], freq='1D'))
        self.test_inst.load(date=self.ocb.dtime[self.ocb.rec_ind])
        self.pysat_var2 = 'dummy2'

        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        self.utils = TestPysatUtils("eval_ocb_metadata")
        self.utils.setUp()
        return

    def tearDown(self):
        """Tear down after each test."""

        self.utils.tearDown()
        del self.test_file, self.log_capture, self.ocb, self.test_inst
        del self.utils, self.lout, self.lwarn, self.ocb_kw, self.pysat_var2
        return

    def test_add_ocb_to_metadata(self):
        """Test the metadata adding routine."""
        ocb_pysat.add_ocb_to_metadata(self.test_inst, "ocb_test",
                                      self.utils.pysat_key, notes="test notes")

        self.utils.meta = self.test_inst.meta
        self.utils.notes = 'test notes'
        self.utils.eval_ocb_metadata()

    def test_add_ocb_to_metadata_vector(self):
        """Test the metadata adding routine for vector data."""
        ocb_pysat.add_ocb_to_metadata(self.test_inst, "ocb_test",
                                      self.utils.pysat_key, notes="test notes",
                                      isvector=True)

        self.utils.meta = self.test_inst.meta
        self.utils.notes = 'test notes'
        self.utils.isvector = True
        self.utils.eval_ocb_metadata()

    def test_no_overwrite_metadata(self):
        """Test the overwrite block on metadata adding routine."""
        self.test_add_ocb_to_metadata()
        ocb_pysat.add_ocb_to_metadata(self.test_inst, "ocb_test",
                                      self.utils.pysat_key,
                                      notes="test notes two",
                                      overwrite=False)

        self.lwarn = u"OCB data already has metadata"
        self.lout = self.log_capture.getvalue()
        self.assertTrue(self.lout.find(self.lwarn) >= 0)
        return

    def test_overwrite_metadata(self):
        """Test the overwrite permission on metadata adding routine."""
        self.test_add_ocb_to_metadata()
        ocb_pysat.add_ocb_to_metadata(self.test_inst, "ocb_test",
                                      self.utils.pysat_key,
                                      notes="test notes two",
                                      overwrite=True)

        meta = self.test_inst.meta
        self.assertRegex(meta['ocb_test'][meta.labels.notes], "test notes two")
        return

    def test_add_ocb_to_data_ocb_obj(self):
        """Test adding ocb to pysat data using the loaded OCB object."""
        ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                  ocb=self.ocb, max_sdiff=600)

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = [aa.split("_ocb")[0]
                                 for aa in self.utils.added_keys]
        self.assertIn('r_corr', self.utils.pysat_keys)
        self.utils.pysat_keys[self.utils.pysat_keys.index("r_corr")] = None

        self.utils.test_inst = self.test_inst
        self.utils.ocb = self.ocb
        self.utils.arevectors = [False for i in range(self.utils.nkeys)]
        self.utils.test_ocb_added()
        return

    def test_deprecated_kwargs(self):
        """Test DeprecationWarning raised for deprecation kwargs."""
        # Set the deprecated keyword arguments with standard values
        dep_inputs = {"min_sectors": 7, "rcent_dev": 8.0, "max_r": 23.0,
                      "min_r": 10.0}

        for dkey in dep_inputs.keys():
            with self.subTest(dkey=dkey):
                self.ocb_kw[dkey] = dep_inputs[dkey]
                with self.assertWarnsRegex(DeprecationWarning,
                                           "Deprecated kwarg will be removed"):
                    ocb_pysat.add_ocb_to_data(self.test_inst, "latitude",
                                              "mlt", **self.ocb_kw)
        return

    def test_add_ocb_to_data_ocb_file(self):
        """Test adding ocb to pysat data using the OCB file name."""
        ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                  max_sdiff=600, **self.ocb_kw)

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = [aa.split("_ocb")[0]
                                 for aa in self.utils.added_keys]
        self.assertIn('r_corr', self.utils.pysat_keys)
        self.utils.pysat_keys[self.utils.pysat_keys.index("r_corr")] = None
        self.arevectors = [False for i in range(self.utils.nkeys)]
        self.utils.test_ocb_added()
        return

    def test_add_ocb_to_data_south_hemisphere_selfset(self):
        """Test successful identification of southern hemisphere only."""

        # Don't set the hemisphere
        del self.ocb_kw['hemisphere']

        # Ensure all data is in the southern hemisphere and the greatest
        # value is identically zero
        new_lat = self.test_inst['latitude'].values
        new_lat[new_lat > 0] *= -1.0
        new_lat[new_lat.argmax()] = 0.0
        self.test_inst['latitude'] = new_lat

        # Add the OCB data to the Instrument and evaluate the output
        ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                  max_sdiff=600, **self.ocb_kw)
        self.utils.test_ocb_added()
        return

    def test_add_ocb_to_data_evar(self):
        """Test adding ocb to pysat with E-field related variables."""
        ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                  evar_names=[self.utils.pysat_key],
                                  ocb=self.ocb, max_sdiff=600)

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = [aa.split("_ocb")[0]
                                 for aa in self.utils.added_keys]
        self.assertIn('r_corr', self.utils.pysat_keys)
        self.utils.pysat_keys[self.utils.pysat_keys.index("r_corr")] = None
        self.arevectors = [False for i in range(self.utils.nkeys)]

        self.utils.test_ocb_added()
        return

    def test_add_ocb_to_data_curl_evar(self):
        """Test adding ocb to pysat with Curl E-field related variables."""
        ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                  curl_evar_names=[self.pysat_var2],
                                  ocb=self.ocb, max_sdiff=600)

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = [aa.split("_ocb")[0]
                                 for aa in self.utils.added_keys]
        self.assertIn('r_corr', self.utils.pysat_keys)
        self.utils.pysat_keys[self.utils.pysat_keys.index("r_corr")] = None

        self.utils.test_ocb_added()
        return

    def test_add_ocb_to_data_evar_vect(self):
        """Test adding ocb to pysat with Curl E-field related vectors."""
        ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                  evar_names=['vect_evar'],
                                  vector_names={
                                      'vect_evar':
                                      {'aacgm_n': self.utils.pysat_key,
                                       'aacgm_e': self.pysat_var2,
                                       'dat_name': 'vect',
                                       'dat_units': 'm/s'}},
                                  ocb=self.ocb, max_sdiff=600)

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = list()
        self.utils.arevectors = list()
        for aa in self.utils.added_keys:
            pp = aa.split("_ocb")[0]
            self.utils.arevectors.append(False)
            if pp == "r_corr":
                pp = None
            elif pp not in self.test_inst.variables:
                pp = self.utils.pysat_key
                self.utils.arevectors[-1] = True
            self.utils.pysat_keys.append(pp)

        self.utils.test_ocb_added()
        return

    def test_add_ocb_to_data_curl_evar_vect(self):
        """Test adding ocb to pysat with Curl E-field related vectors."""
        ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                  curl_evar_names=['vect_cevar'],
                                  vector_names={
                                      'vect_cevar':
                                      {'aacgm_n': self.utils.pysat_key,
                                       'aacgm_e': self.pysat_var2,
                                       'dat_name': 'vect',
                                       'dat_units': 'm/s'}},
                                  ocb=self.ocb, max_sdiff=600)

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = list()
        self.utils.arevectors = list()
        for aa in self.utils.added_keys:
            pp = aa.split("_ocb")[0]
            self.utils.arevectors.append(False)
            if pp == "r_corr":
                pp = None
            elif pp not in self.test_inst.variables:
                pp = self.utils.pysat_key
                self.utils.arevectors[-1] = True
            self.utils.pysat_keys.append(pp)

        self.utils.test_ocb_added()
        return

    def test_add_ocb_to_data_custom_vect(self):
        """Test adding ocb to pysat with custom scaled variables."""

        ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                  vector_names={
                                      'vect_cust':
                                      {'aacgm_n': self.utils.pysat_key,
                                       'aacgm_e': self.pysat_var2,
                                       'dat_name': 'vect',
                                       'dat_units': 'm/s',
                                       'scale_func': None}},
                                  ocb=self.ocb, max_sdiff=600)

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = list()
        self.utils.arevectors = list()
        for aa in self.utils.added_keys:
            pp = aa.split("_ocb")[0]
            self.utils.arevectors.append(False)
            if pp == "r_corr":
                pp = None
            elif pp not in self.test_inst.variables:
                pp = self.utils.pysat_key
                self.utils.arevectors[-1] = True
            self.utils.pysat_keys.append(pp)
        self.utils.test_ocb_added()
        return

    def test_add_ocb_to_data_all_types(self):
        """Test adding ocb to pysat with E-field, Curl, and Vector data."""
        ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                  evar_names=[self.utils.pysat_key],
                                  curl_evar_names=[self.pysat_var2],
                                  vector_names={
                                      'vect_cust':
                                      {'aacgm_n': self.utils.pysat_key,
                                       'aacgm_e': self.pysat_var2,
                                       'dat_name': 'vect',
                                       'dat_units': 'm/s',
                                       'scale_func': None}},
                                  ocb=self.ocb, max_sdiff=600)

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = list()
        self.utils.arevectors = list()
        for aa in self.utils.added_keys:
            pp = aa.split("_ocb")[0]
            self.utils.arevectors.append(False)
            if pp == "r_corr":
                pp = None
            elif pp not in self.test_inst.variables:
                pp = self.utils.pysat_key
                self.utils.arevectors[-1] = True
            self.utils.pysat_keys.append(pp)

        self.utils.test_ocb_added()
        return

    def test_add_ocb_to_data_no_file(self):
        """Test adding ocb to pydat when no OCB file or data is provided."""
        self.ocb_kw['ocbfile'] = None
        ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                  **self.ocb_kw, max_sdiff=600)

        self.lwarn = u"no data in Boundary file(s)"
        self.lout = self.log_capture.getvalue()
        self.assertTrue(self.lout.find(self.lwarn) >= 0)
        return

    def test_add_ocb_to_data_bad_hemisphere_selfset(self):
        """Test failure of a pysat.Instrument to specify a hemisphere."""
        del self.ocb_kw['hemisphere']

        with self.assertRaisesRegex(
                ValueError, 'cannot process observations from both '):
            ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                      **self.ocb_kw)
        return

    def test_bad_pysat_inst(self):
        """Test failure of a bad pysat.Instrument in pysat functions."""

        # Set the function and input data
        func_dict = {ocb_pysat.add_ocb_to_data: [None, "latitude", "mlt"],
                     ocb_pysat.add_ocb_to_metadata: [None, "ocb_mlt", "mlt"]}

        # Test the error for each function
        for func in func_dict.keys():
            with self.subTest(func=func):
                with self.assertRaisesRegex(
                        ValueError,
                        'unknown class, expected pysat.Instrument'):
                    func(*func_dict[func])
        return

    def test_add_ocb_to_data_bad_mlat(self):
        """Test failure of unknown mlat key in add_ocb_to_data."""

        with self.assertRaisesRegex(ValueError,
                                    'unknown magnetic latitude name mlat'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "mlat", "mlt",
                                      ocb=self.ocb)

    def test_add_ocb_to_data_bad_mlt(self):
        """Test failure of unknown mlt key in add_ocb_to_data."""
        with self.assertRaisesRegex(ValueError,
                                    'unknown magnetic local time name bad'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "bad",
                                      ocb=self.ocb)
        return

    def test_add_ocb_to_data_bad_evar(self):
        """Test failure of unknown E field key in add_ocb_to_data."""
        with self.assertRaisesRegex(ValueError,
                                    'at least one unknown E field name'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                      evar_names=["bad"], ocb=self.ocb)
        return

    def test_add_ocb_to_data_bad_curl(self):
        """Test failure of unknown E field key in add_ocb_to_data."""
        with self.assertRaisesRegex(ValueError,
                                    'at least one unknown E field name'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                      curl_evar_names=["bad"], ocb=self.ocb)
        return

    def test_add_ocb_to_data_bad_vector_scale(self):
        """Test failure of missing scaling function in add_ocb_to_data."""
        with self.assertRaisesRegex(ValueError,
                                    'missing scaling function for bad'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                      vector_names={'bad': {'aacgm_n': 'bad_n',
                                                            'aacgm_e': 'bad_e',
                                                            'dat_name': 'bad',
                                                            'dat_units': ''}},
                                      ocb=self.ocb)
        return

    def test_add_ocb_to_data_bad_vector_name(self):
        """Test failure of missing scaling function in add_ocb_to_data."""
        with self.assertRaisesRegex(ValueError,
                                    'unknown vector name bad_n'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                      evar_names=['bad'],
                                      vector_names={
                                          'bad':
                                          {'aacgm_n': 'bad_n',
                                           'aacgm_e': self.utils.pysat_key,
                                           'dat_name': 'bad',
                                           'dat_units': ''}},
                                      ocb=self.ocb)
        return


class TestPysatMethodsEAB(TestPysatMethods):
    """Integration tests for using ocbpy.EABoundary on pysat pandas data."""

    def setUp(self):
        """Initialize the test class."""

        self.test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                                   "test_data", "test_north_eab")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb_kw = {"ocbfile": self.test_file,
                       "instrument": "image", "hemisphere": 1}
        self.ocb = ocbpy.EABoundary(self.test_file, instrument='image',
                                    hemisphere=1)
        self.ocb.rec_ind = 27

        self.test_inst = pysat.Instrument('pysat', 'testing', num_samples=50400,
                                          clean_level='clean',
                                          update_files=True,
                                          file_date_range=pds.date_range(
                                              self.ocb.dtime[0],
                                              self.ocb.dtime[-1], freq='1D'))
        self.test_inst.load(date=self.ocb.dtime[self.ocb.rec_ind])
        self.pysat_var2 = 'dummy2'

        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        self.utils = TestPysatUtils("eval_ocb_metadata")
        self.utils.setUp()
        return

    def tearDown(self):
        """Tear down after each test."""

        self.utils.tearDown()
        del self.test_file, self.log_capture, self.ocb, self.test_inst
        del self.utils, self.lout, self.lwarn, self.ocb_kw, self.pysat_var2
        return


class TestPysatMethodsDual(TestPysatMethods):
    """Integration tests for using ocbpy.DualBoundary on pysat pandas data."""

    def setUp(self):
        """Initialize the test class."""

        self.test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                                   "test_data", "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb_kw = {"ocbfile": self.test_file,
                       "instrument": "image", "hemisphere": 1}
        self.ocb = ocbpy.DualBoundary(ocb_filename=self.test_file,
                                      ocb_instrument='image',
                                      eab_filename=self.test_file.replace(
                                          "north_circle", "north_eab"),
                                      eab_instrument='image',
                                      hemisphere=1)
        self.ocb.rec_ind = 0

        self.test_inst = pysat.Instrument('pysat', 'testing', num_samples=50400,
                                          clean_level='clean',
                                          update_files=True,
                                          file_date_range=pds.date_range(
                                              self.ocb.dtime[0],
                                              self.ocb.dtime[-1], freq='1D'))
        self.test_inst.load(date=self.ocb.dtime[self.ocb.rec_ind])
        self.pysat_var2 = 'dummy2'

        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        self.utils = TestPysatUtils("eval_ocb_metadata")
        self.utils.setUp()
        return

    def tearDown(self):
        """Tear down after each test."""

        self.utils.tearDown()
        del self.test_file, self.log_capture, self.ocb, self.test_inst
        del self.utils, self.lout, self.lwarn, self.ocb_kw, self.pysat_var2
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatMethods2DXarray(TestPysatMethods):
    """Integration tests for using ocbpy on pysat 2D Xarray data."""

    def setUp(self):
        """Initialize the test class."""

        self.test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                                   "test_data", "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb_kw = {"ocbfile": self.test_file,
                       "instrument": "image", "hemisphere": 1}
        self.ocb = ocbpy.OCBoundary(self.test_file, instrument='image',
                                    hemisphere=1)
        self.ocb.rec_ind = 27

        self.test_inst = pysat.Instrument('pysat', 'testing2d_xarray',
                                          num_samples=50400,
                                          clean_level='clean',
                                          update_files=True,
                                          file_date_range=pds.date_range(
                                              self.ocb.dtime[0],
                                              self.ocb.dtime[-1], freq='1D'))
        self.test_inst.load(date=self.ocb.dtime[self.ocb.rec_ind])
        self.pysat_var2 = 'dummy2'

        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        self.utils = TestPysatUtils("eval_ocb_metadata")
        self.utils.setUp()
        return

    def tearDown(self):
        """Tear down after each test."""

        self.utils.tearDown()
        del self.test_file, self.log_capture, self.ocb, self.test_inst
        del self.utils, self.lout, self.lwarn, self.ocb_kw, self.pysat_var2
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatMethodsXarray(TestPysatMethods):
    """Integration tests for using ocbpy on pysat Xarray data."""

    def setUp(self):
        """Initialize the test class."""
        self.test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                                   "test_data", "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb_kw = {"ocbfile": self.test_file,
                       "instrument": "image", "hemisphere": 1}
        self.ocb = ocbpy.OCBoundary(self.test_file, instrument='image',
                                    hemisphere=1)
        self.ocb.rec_ind = 27  # NOTE: ADD SETTER FOR THIS IN DUAL BOUNDARY

        self.test_inst = pysat.Instrument('pysat', 'testing_xarray',
                                          num_samples=50400,
                                          clean_level='clean',
                                          update_files=True,
                                          file_date_range=pds.date_range(
                                              self.ocb.dtime[0],
                                              self.ocb.dtime[-1], freq='1D'))
        self.test_inst.load(date=self.ocb.dtime[self.ocb.rec_ind])
        self.pysat_var2 = 'dummy2'

        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        self.utils = TestPysatUtils("eval_ocb_metadata")
        self.utils.setUp()
        return

    def tearDown(self):
        """Clean the test environment."""

        self.utils.tearDown()
        del self.test_file, self.log_capture, self.ocb, self.test_inst
        del self.utils, self.lout, self.lwarn, self.ocb_kw, self.pysat_var2
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatCustMethods(unittest.TestCase):
    """Integration tests for using ocbpy as a custom function with pysat pandas.

    """

    def setUp(self):
        """Initialize the unit tests for using the pysat.Custom methods."""
        self.test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                                   "test_data", "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb = ocbpy.OCBoundary(self.test_file, instrument='image',
                                    hemisphere=1)
        self.ocb.rec_ind = 27

        self.test_inst = pysat.Instrument('pysat', 'testing', num_samples=50400,
                                          clean_level='clean',
                                          update_files=True,
                                          file_date_range=pds.date_range(
                                              self.ocb.dtime[0],
                                              self.ocb.dtime[-1], freq='1D'))
        self.pysat_var2 = 'dummy2'
        self.cust_kwargs = {'mlat_name': 'latitude', 'mlt_name': 'mlt',
                            'ocb': self.ocb, 'max_sdiff': 600}

        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        self.utils = TestPysatUtils("eval_ocb_metadata")
        self.utils.setUp()
        return

    def tearDown(self):
        """Clean the test environment."""

        self.utils.tearDown()
        del self.test_file, self.log_capture, self.test_inst, self.ocb
        del self.utils, self.lwarn, self.lout, self.cust_kwargs
        del self.pysat_var2
        return

    def test_load(self):
        """Test the pysat file loading."""
        self.test_inst.load(date=self.ocb.dtime[self.ocb.rec_ind])
        self.assertFalse(self.test_inst.empty)
        self.assertIn('latitude', self.test_inst.variables)
        self.assertIn('mlt', self.test_inst.variables)
        return

    def test_cust_add_ocb_to_data_ocb_obj(self):
        """Test adding ocb to pysat data using the loaded OCB object."""
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        self.test_load()

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = [aa.split("_ocb")[0]
                                 for aa in self.utils.added_keys]
        self.assertIn('r_corr', self.utils.pysat_keys)
        self.utils.pysat_keys[self.utils.pysat_keys.index("r_corr")] = None

        self.utils.test_inst = self.test_inst
        self.utils.ocb = self.ocb
        self.utils.arevectors = [False for i in range(self.utils.nkeys)]
        self.utils.test_ocb_added()
        return

    def test_cust_add_ocb_to_data_ocb_file(self):
        """Test adding ocb to pysat data with load using the OCB file name."""
        del self.cust_kwargs['ocb']
        self.cust_kwargs['ocbfile'] = self.test_file
        self.cust_kwargs['instrument'] = 'image'
        self.cust_kwargs['hemisphere'] = 1
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        self.test_load()

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = [aa.split("_ocb")[0]
                                 for aa in self.utils.added_keys]
        self.assertIn('r_corr', self.utils.pysat_keys)
        self.utils.pysat_keys[self.utils.pysat_keys.index("r_corr")] = None
        self.arevectors = [False for i in range(self.utils.nkeys)]
        self.utils.test_ocb_added()
        return

    def test_cust_add_ocb_to_data_evar(self):
        """Test adding ocb to pysat with load including E-field variables."""
        self.cust_kwargs['evar_names'] = [self.utils.pysat_key]
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        self.test_load()

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = [aa.split("_ocb")[0]
                                 for aa in self.utils.added_keys]
        self.assertIn('r_corr', self.utils.pysat_keys)
        self.utils.pysat_keys[self.utils.pysat_keys.index("r_corr")] = None
        self.arevectors = [False for i in range(self.utils.nkeys)]

        self.utils.test_ocb_added()
        return

    def test_cust_add_ocb_to_data_curl_evar(self):
        """Test adding ocb to pysat including Curl E-field variables."""

        self.cust_kwargs['curl_evar_names'] = [self.pysat_var2]
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        self.test_load()

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = [aa.split("_ocb")[0]
                                 for aa in self.utils.added_keys]
        self.assertIn('r_corr', self.utils.pysat_keys)
        self.utils.pysat_keys[self.utils.pysat_keys.index("r_corr")] = None

        self.utils.test_ocb_added()
        return

    def test_cust_add_ocb_to_data_evar_vect(self):
        """Test adding ocb to pysat with load including Curl E-field vectors."""

        self.cust_kwargs['evar_names'] = ['vect_evar']
        self.cust_kwargs['vector_names'] = {'vect_evar':
                                            {'aacgm_n': self.utils.pysat_key,
                                             'aacgm_e': self.pysat_var2,
                                             'dat_name': 'vect',
                                             'dat_units': 'm/s'}}

        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        self.test_load()

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = list()
        self.utils.arevectors = list()
        for aa in self.utils.added_keys:
            pp = aa.split("_ocb")[0]
            self.utils.arevectors.append(False)
            if pp == "r_corr":
                pp = None
            elif pp not in self.test_inst.variables:
                pp = self.utils.pysat_key
                self.utils.arevectors[-1] = True
            self.utils.pysat_keys.append(pp)
        self.utils.test_ocb_added()
        return

    def test_cust_add_ocb_to_data_curl_evar_vect(self):
        """Test adding ocb to pysat including Curl E-field vectors."""
        self.cust_kwargs['curl_evar_names'] = ['vect_cevar']
        self.cust_kwargs['vector_names'] = {'vect_cevar':
                                            {'aacgm_n': self.utils.pysat_key,
                                             'aacgm_e': self.pysat_var2,
                                             'dat_name': 'vect',
                                             'dat_units': 'm/s'}}

        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        self.test_load()

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = list()
        self.utils.arevectors = list()
        for aa in self.utils.added_keys:
            pp = aa.split("_ocb")[0]
            self.utils.arevectors.append(False)
            if pp == "r_corr":
                pp = None
            elif pp not in self.test_inst.variables:
                pp = self.utils.pysat_key
                self.utils.arevectors[-1] = True
            self.utils.pysat_keys.append(pp)
        self.utils.test_ocb_added()
        return

    def test_cust_add_ocb_to_data_custom_vect(self):
        """Test adding ocb to pysat including custom scaled variables."""
        self.cust_kwargs['vector_names'] = {'vect_cust':
                                            {'aacgm_n': self.utils.pysat_key,
                                             'aacgm_e': self.pysat_var2,
                                             'dat_name': 'vect',
                                             'dat_units': 'm/s',
                                             'scale_func': None}}
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)
        self.test_load()

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = list()
        self.utils.arevectors = list()
        for aa in self.utils.added_keys:
            pp = aa.split("_ocb")[0]
            self.utils.arevectors.append(False)
            if pp == "r_corr":
                pp = None
            elif pp not in self.test_inst.variables:
                pp = self.utils.pysat_key
                self.utils.arevectors[-1] = True
            self.utils.pysat_keys.append(pp)

        self.utils.test_ocb_added()
        return

    def test_cust_add_ocb_to_data_all_types(self):
        """Test adding ocb to pysat including E-field, Curl, & vects."""
        self.cust_kwargs['evar_names'] = [self.utils.pysat_key]
        self.cust_kwargs['curl_evar_names'] = [self.pysat_var2]
        self.cust_kwargs['vector_names'] = {'vect_cust':
                                            {'aacgm_n': self.utils.pysat_key,
                                             'aacgm_e': self.pysat_var2,
                                             'dat_name': 'vect',
                                             'dat_units': 'm/s',
                                             'scale_func': None}}
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        self.test_load()

        self.utils.added_keys = [kk for kk in self.test_inst.meta.keys()
                                 if kk.find('_ocb') > 0]
        self.utils.nkeys = len(self.utils.added_keys)
        self.utils.pysat_keys = list()
        self.utils.arevectors = list()
        for aa in self.utils.added_keys:
            pp = aa.split("_ocb")[0]
            self.utils.arevectors.append(False)
            if pp == "r_corr":
                pp = None
            elif pp not in self.test_inst.variables:
                pp = self.utils.pysat_key
                self.utils.arevectors[-1] = True
            self.utils.pysat_keys.append(pp)

        self.utils.test_ocb_added()
        return

    def test_cust_add_ocb_to_data_no_file(self):
        """Test adding ocb to pysat with load using no OCB file or data."""
        del self.cust_kwargs['ocb']
        self.cust_kwargs['ocbfile'] = None
        self.cust_kwargs['instrument'] = 'image'
        self.cust_kwargs['hemisphere'] = 1
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        self.test_load()

        self.lwarn = u'no data in Boundary file(s)'
        self.lout = self.log_capture.getvalue()
        self.assertTrue(self.lout.find(self.lwarn) >= 0)
        return

    def test_cust_add_ocb_to_data_bad_mlat(self):
        """Test failure of unknown mlat key in add_ocb_to_data custom func."""

        with self.assertRaisesRegex(ValueError,
                                    'unknown magnetic latitude name mlat'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "mlat", "mlt",
                                      ocb=self.ocb)

    def test_cust_add_ocb_to_data_bad_mlt(self):
        """Test failure of unknown mlt key in add_ocb_to_data custom func."""
        self.cust_kwargs['mlt_name'] = 'bad'
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        with self.assertRaisesRegex(ValueError,
                                    'unknown magnetic local time name bad'):
            self.test_load()
        return

    def test_cust_add_ocb_to_data_bad_evar(self):
        """Test failure of unknown E field key in custom func."""
        self.cust_kwargs['evar_names'] = ['bad']
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        with self.assertRaisesRegex(ValueError,
                                    'at least one unknown E field name'):
            self.test_load()
        return

    def test_cust_add_ocb_to_data_bad_curl(self):
        """Test failure of unknown E field key in custom func."""
        self.cust_kwargs['curl_evar_names'] = ['bad']
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        with self.assertRaisesRegex(ValueError,
                                    'at least one unknown E field name'):
            self.test_load()
        return

    def test_cust_add_ocb_to_data_bad_vector_scale(self):
        """Test failure of missing scaling function in custom func."""
        self.cust_kwargs['vector_names'] = {'bad': {'aacgm_n': 'bad_n',
                                                    'aacgm_e': 'bad_e',
                                                    'dat_name': 'bad',
                                                    'dat_units': ''}}
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        with self.assertRaisesRegex(ValueError,
                                    'missing scaling function for bad'):
            self.test_load()
        return

    def test_cust_add_ocb_to_data_bad_vector_name(self):
        """Test failure of missing scaling function in custom func."""
        self.cust_kwargs['evar_names'] = ['bad']
        self.cust_kwargs['vector_names'] = {'bad':
                                            {'aacgm_n': 'bad_n',
                                             'aacgm_e': self.utils.pysat_key,
                                             'dat_name': 'bad',
                                             'dat_units': ''}}
        self.test_inst.custom_attach(ocb_pysat.add_ocb_to_data,
                                     kwargs=self.cust_kwargs)

        with self.assertRaisesRegex(ValueError, 'unknown vector name bad_n'):
            self.test_load()
        return


class TestPysatCustMethodsEAB(TestPysatCustMethods):
    """Integration tests for pysat pandas through custom with EABs."""

    def setUp(self):
        """Initialize the unit tests for using the pysat.Custom methods."""
        self.test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                                   "test_data", "test_north_eab")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb = ocbpy.EABoundary(self.test_file, instrument='image',
                                    hemisphere=1)
        self.ocb.rec_ind = 27

        self.test_inst = pysat.Instrument('pysat', 'testing', num_samples=50400,
                                          clean_level='clean',
                                          update_files=True,
                                          file_date_range=pds.date_range(
                                              self.ocb.dtime[0],
                                              self.ocb.dtime[-1], freq='1D'))
        self.pysat_var2 = 'dummy2'
        self.cust_kwargs = {'mlat_name': 'latitude', 'mlt_name': 'mlt',
                            'ocb': self.ocb, 'max_sdiff': 600}

        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        self.utils = TestPysatUtils("eval_ocb_metadata")
        self.utils.setUp()
        return

    def tearDown(self):
        """Clean the test environment."""

        self.utils.tearDown()
        del self.test_file, self.log_capture, self.test_inst, self.ocb
        del self.utils, self.lwarn, self.lout, self.cust_kwargs
        del self.pysat_var2
        return


class TestPysatCustMethodsDual(TestPysatCustMethods):
    """Integration tests for pysat pandas through custom with dual boundaries.

    """

    def setUp(self):
        """Initialize the unit tests for using the pysat.Custom methods."""
        self.test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                                   "test_data", "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb = ocbpy.DualBoundary(ocb_filename=self.test_file,
                                      ocb_instrument='image',
                                      eab_filename=self.test_file.replace(
                                          'north_circle', 'north_eab'),
                                      eab_instrument='image',
                                      hemisphere=1)
        self.ocb.rec_ind = 0

        self.test_inst = pysat.Instrument('pysat', 'testing', num_samples=50400,
                                          clean_level='clean',
                                          update_files=True,
                                          file_date_range=pds.date_range(
                                              self.ocb.dtime[0],
                                              self.ocb.dtime[-1], freq='1D'))
        self.pysat_var2 = 'dummy2'
        self.cust_kwargs = {'mlat_name': 'latitude', 'mlt_name': 'mlt',
                            'ocb': self.ocb, 'max_sdiff': 600}

        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        self.utils = TestPysatUtils("eval_ocb_metadata")
        self.utils.setUp()
        return

    def tearDown(self):
        """Clean the test environment."""

        self.utils.tearDown()
        del self.test_file, self.log_capture, self.test_inst, self.ocb
        del self.utils, self.lwarn, self.lout, self.cust_kwargs
        del self.pysat_var2
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatCustMethodsXarray(TestPysatCustMethods):
    """Integration tests for using ocbpy as a custom function with pysat Xarray.

    """

    def setUp(self):
        """Initialize the unit tests for using the pysat.Custom methods."""
        self.test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                                   "test_data", "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb = ocbpy.OCBoundary(self.test_file, instrument='image',
                                    hemisphere=1)
        self.ocb.rec_ind = 27

        self.test_inst = pysat.Instrument('pysat', 'testing_xarray',
                                          num_samples=50400,
                                          clean_level='clean',
                                          update_files=True,
                                          file_date_range=pds.date_range(
                                              self.ocb.dtime[0],
                                              self.ocb.dtime[-1], freq='1D'))
        self.pysat_var2 = 'dummy2'
        self.cust_kwargs = {'mlat_name': 'latitude', 'mlt_name': 'mlt',
                            'ocb': self.ocb, 'max_sdiff': 600}

        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        self.utils = TestPysatUtils("eval_ocb_metadata")
        self.utils.setUp()
        return

    def tearDown(self):
        """Clean the test environment."""

        self.utils.tearDown()
        del self.test_file, self.log_capture, self.test_inst, self.ocb
        del self.utils, self.lwarn, self.lout, self.cust_kwargs
        del self.pysat_var2
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatCustMethods2DXarray(TestPysatCustMethods):
    """Integration tests for using ocbpy with pysat 2D Xarray data."""

    def setUp(self):
        """Initialize the tests for using the pysat.Custom methods."""
        self.test_file = path.join(path.dirname(ocbpy.__file__), "tests",
                                   "test_data", "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb = ocbpy.OCBoundary(self.test_file, instrument='image',
                                    hemisphere=1)
        self.ocb.rec_ind = 27

        self.test_inst = pysat.Instrument('pysat', 'testing2d_xarray',
                                          num_samples=50400,
                                          clean_level='clean',
                                          update_files=True,
                                          file_date_range=pds.date_range(
                                              self.ocb.dtime[0],
                                              self.ocb.dtime[-1], freq='1D'))
        self.pysat_var2 = 'dummy2'
        self.cust_kwargs = {'mlat_name': 'latitude', 'mlt_name': 'mlt',
                            'ocb': self.ocb, 'max_sdiff': 600}

        self.lwarn = u""
        self.lout = u""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)

        self.utils = TestPysatUtils("eval_ocb_metadata")
        self.utils.setUp()
        return

    def tearDown(self):
        """Clean the test environment."""
        self.utils.tearDown()
        del self.test_file, self.log_capture, self.test_inst, self.ocb
        del self.utils, self.lwarn, self.lout, self.cust_kwargs
        del self.pysat_var2
        return
