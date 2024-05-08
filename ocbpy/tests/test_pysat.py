#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the ocbpy.instruments.pysat functions."""

import numpy as np
from os import path
from packaging import version
import unittest

import ocbpy
import ocbpy.tests.class_common as cc

try:
    # Import pysat first to get the correct error message
    import pysat
    import ocbpy.instruments.pysat_instruments as ocb_pysat
    import pandas as pds
    import xarray as xr
    no_pysat = False
except ImportError:
    no_pysat = True


@unittest.skipIf(not no_pysat, "pysat installed, cannot test failure")
class TestPysatFailure(unittest.TestCase):
    """Unit tests for the pysat instrument functions without pysat installed."""

    def test_import_failure(self):
        """Test pysat import failure."""

        with self.assertRaisesRegex(ImportError, 'unable to load the pysat'):
            import ocbpy.instruments.pysat_instruments as ocb_pysat  # NOQA 401
        return


@unittest.skipIf(no_pysat, "pysat not installed, cannot test routines")
class TestPysatUtils(unittest.TestCase):
    """Tests for using ocbpy with pysat data."""

    def setUp(self):
        """Initialize the test environment."""

        # Set the default function values
        self.pysat_key = 'dummy1'
        self.pysat_lat = 'latitude'
        self.pysat_alt = 'altitude'
        self.notes = None
        self.test_inst = None
        self.ocb = None
        self.added_keys = list()
        self.pysat_keys = list()
        self.del_time = 600
        self.ocb_name = "ocb"
        self.ocb_key = '{:s}_test'.format(self.ocb_name)
        return

    def tearDown(self):
        """Tear down the testing environment."""
        del self.ocb_key, self.pysat_key, self.notes, self.del_time
        del self.test_inst, self.ocb, self.added_keys, self.pysat_keys
        del self.pysat_lat, self.pysat_alt, self.ocb_name
        return

    def eval_ocb_metadata(self):
        """Evaluate new OCB metadata."""
        bound_desc = {"ocb": "Open Closed field-line Boundary",
                      "eab": "Equatorward Auroral Boundary",
                      "dualb": "Dual Boundary", "": ""}

        # Test MetaData exists
        if self.test_inst is not None:
            self.assertIsNotNone(self.test_inst.meta, msg="No meta data added")

            # Tests for MetaData
            self.assertIn(self.ocb_key, list(self.test_inst.meta.keys()))
            if self.pysat_key is not None:
                self.assertIn(self.pysat_key, list(self.test_inst.meta.keys()))

            # Test the fill value
            self.assertTrue(
                np.isnan(self.test_inst.meta[self.ocb_key][
                    self.test_inst.meta.labels.fill_val]),
                msg="".join([repr(self.ocb_key), " fill value is ",
                             repr(self.test_inst.meta[self.ocb_key][
                                 self.test_inst.meta.labels.fill_val]),
                             ", not np.nan"]))

            if self.pysat_key is not None:
                # Test the elements that are identical
                for ll in [self.test_inst.meta.labels.units,
                           self.test_inst.meta.labels.min_val,
                           self.test_inst.meta.labels.max_val,
                           self.test_inst.meta.labels.fill_val]:
                    try:
                        if np.isnan(self.test_inst.meta[self.pysat_key][ll]):
                            self.assertTrue(
                                np.isnan(self.test_inst.meta[self.ocb_key][ll]))
                        elif ll != self.test_inst.meta.labels.fill_val:
                            # The OCB fill value is NaN, regardless of prior
                            # value
                            self.assertEqual(
                                self.test_inst.meta[self.ocb_key][ll],
                                self.test_inst.meta[self.pysat_key][ll],
                                msg="unequal fill vals [{:s} and {:s}]".format(
                                    self.ocb_key, self.pysat_key))
                    except TypeError:
                        ocb_len = len(self.test_inst.meta[self.ocb_key][ll])
                        pysat_len = len(self.test_inst.meta[self.pysat_key][ll])
                        if pysat_len == 0:
                            self.assertGreaterEqual(ocb_len, pysat_len)
                        else:
                            self.assertRegex(
                                self.test_inst.meta[self.ocb_key][ll],
                                self.test_inst.meta[self.pysat_key][ll],
                                msg="".join(["Meta label ", ll, ": OCB key ",
                                             self.ocb_key, " value `",
                                             self.test_inst.meta[self.ocb_key][
                                                 ll], "` not in pysat key ",
                                             self.pysat_key, " value `",
                                             self.test_inst.meta[
                                                 self.pysat_key][ll], "`"]))

            # Test the elements that have "OCB" appended to the text
            sline = self.test_inst.meta[self.ocb_key][
                self.test_inst.meta.labels.name].split(" ")
            self.assertRegex(sline[0], self.ocb_name.upper(), msg=sline)
            note_line = self.test_inst.meta[self.ocb_key][
                self.test_inst.meta.labels.notes]
            if self.pysat_key is not None and note_line.find(
                    'scaled using') < 0:
                self.assertRegex(
                    " ".join(sline[1:]), self.test_inst.meta[self.pysat_key][
                        self.test_inst.meta.labels.name],
                    msg="Bad long name for {:}; notes are: {:}".format(
                        self.pysat_key, note_line))

            # Test the remaining elements
            self.assertRegex(self.test_inst.meta[self.ocb_key][
                self.test_inst.meta.labels.desc], bound_desc[self.ocb_name])
            if self.notes is not None:
                self.assertRegex(self.test_inst.meta[self.ocb_key][
                    self.test_inst.meta.labels.notes], self.notes)
        return

    def test_ocb_added(self):
        """Test if OCB data was added correctly."""
        # If there is no test instrument, test passes
        if self.test_inst is None:
            return

        # For a test instrument, evaluate attributes
        for i, self.ocb_key in enumerate(self.added_keys):
            # Test to see that data was added
            self.assertIn(self.ocb_key, self.test_inst.variables)

            # Test the metadata
            self.pysat_key = self.pysat_keys[i]
            self.eval_ocb_metadata()

            # Test to see that data within the time tolerance of the test OCBs
            # has OCB locations and other data is NaN
            match_data = self.test_inst[self.ocb_key]
            mask_data = np.isfinite(match_data)

            self.assertTrue(np.array(mask_data).any(),
                            msg="No OCB data for {:}".format(self.ocb_key))

            mind = np.where(np.array(mask_data))
            match_data = np.array(match_data)[mind]
            match_time = np.unique(self.test_inst.index[mind[0]])

            for ii in match_time:
                check_time = abs(pds.to_datetime(ii)
                                 - self.ocb.dtime).min().total_seconds()
                self.assertLessEqual(check_time, self.del_time,
                                     msg="".join(["bad time difference for ",
                                                  "OCB key ", self.ocb_key,
                                                  " at {:}".format(ii)]))

            if(self.pysat_key is not None
               and self.pysat_key not in [self.pysat_lat, 'mlt']):
                pysat_data = self.test_inst[self.pysat_key].where(mask_data)

                # Get the scaling radius
                if hasattr(self.ocb, "r"):
                    rscale = (self.ocb.r
                              / (90.0 - abs(self.ocb.boundary_lat)))**2
                else:
                    rscale = (self.ocb.ocb.r
                              / (90.0 - abs(self.ocb.ocb.boundary_lat)))**2

                # Evaluate the non-vector data
                if self.ocb_key.find(self.pysat_key) >= 0:
                    self.assertGreaterEqual(match_data.min(),
                                            pysat_data.min() * rscale.min(),
                                            msg="".join([
                                                "bad comparison between ",
                                                self.ocb_key, " and ",
                                                self.pysat_key]))
                    self.assertGreaterEqual(pysat_data.max() * rscale.max(),
                                            match_data.max(),
                                            msg="".join([
                                                "bad comparison between ",
                                                self.ocb_key, " and ",
                                                self.pysat_key]))

        return


class PysatBase(TestPysatUtils):
    """Base class for pysat testing."""

    def setUp(self):
        """Initialize the base class."""
        # Set the util default values
        TestPysatUtils.setUp(self)

        # Set the method default values
        self.test_file = path.join(cc.test_dir, "test_north_ocb")
        self.ocb_kw = {"filename": self.test_file,
                       "instrument": "image", "hemisphere": 1}
        self.ocb_class = ocbpy.OCBoundary
        self.rec_ind = 27
        self.pysat_var2 = 'dummy2'
        self.test_module = pysat.instruments.pysat_testing
        self.pysat_kw = {}
        return

    def tearDown(self):
        """Tear down after each test."""
        # Set the util default values
        TestPysatUtils.tearDown(self)

        del self.test_file, self.ocb_kw, self.ocb_class, self.rec_ind
        del self.pysat_var2, self.test_module, self.pysat_kw
        return

    def load_boundaries(self):
        """Load the OCB boundary class object for testing."""
        # Verify the existence of the test file
        self.assertTrue(path.isfile(self.test_file),
                        msg="'{:}' is not a file".format(self.test_file))

        # Set up the OCB object
        self.ocb = self.ocb_class(**self.ocb_kw)
        if self.rec_ind < self.ocb.records:
            self.ocb.rec_ind = self.rec_ind
        return

    def load_instrument(self):
        """Load the pysat Instrument for testing."""
        if self.ocb is None:
            self.load_boundaries()

        if 'file_date_range' not in self.pysat_kw:
            self.pysat_kw['file_date_range'] = pds.date_range(
                self.ocb.dtime[0], self.ocb.dtime[-1], freq='1D')

        if 'num_samples' not in self.pysat_kw:
            self.pysat_kw['num_samples'] = 50400

        self.test_inst = pysat.Instrument(inst_module=self.test_module,
                                          **self.pysat_kw)

        # Reduce pysat warnings
        # TODO(#130) remove version checking by updating minimum supported pysat
        load_kwargs = {'date': self.ocb.dtime[self.rec_ind]}
        if version.Version(pysat.__version__) > version.Version(
                '3.0.1') and version.Version(
                    pysat.__version__) < version.Version('3.2.0'):
            load_kwargs['use_header'] = True

        self.test_inst.load(**load_kwargs)
        return

    def set_new_keys(self, exclude_r_corr=True):
        """Set the `added_keys` and `pysat_keys` attributes."""
        self.added_keys = [kk for kk in self.test_inst.meta.keys()
                           if kk.find('_{:s}'.format(self.ocb_name)) > 0]

        if exclude_r_corr:
            self.pysat_keys = list()
            for aa in self.added_keys:
                pp = aa.split("_{:s}".format(self.ocb_name))[0]
                if pp == "r_corr":
                    pp = None
                elif pp not in self.test_inst.variables:
                    pp = self.pysat_key
                self.pysat_keys.append(pp)
        else:
            self.pysat_keys = [aa.split("_{:s}".format(self.ocb_name))[0]
                               for aa in self.added_keys]
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatStructure(unittest.TestCase):
    """Unit tests for the pysat instrument functions."""

    def test_add_ocb_to_data_defaults(self):
        """Test the add_ocb_to_data function defaults."""
        defaults = ocb_pysat.add_ocb_to_data.__defaults__

        for i in [0, 1, 2, 10]:
            self.assertEqual(len(defaults[i]), 0,
                             msg="Default {:d} value {:} != 0".format(
                                 i, defaults[i]))

        for i in [3, 4, 5, 8, 12, 13]:
            self.assertIsNone(defaults[i])

        self.assertEqual(defaults[6], 350)
        self.assertEqual(defaults[7], 0)
        self.assertRegex(defaults[9], 'ocb')
        self.assertEqual(defaults[11], 60)
        self.assertRegex(defaults[14], 'magnetic')
        self.assertRegex(defaults[15], 'magnetic')
        return

    def test_add_ocb_to_metadata_defaults(self):
        """Test the add_ocb_to_metadata function defaults."""
        defaults = ocb_pysat.add_ocb_to_metadata.__defaults__

        for i in [0, 2]:
            self.assertFalse(defaults[i])

        self.assertEqual(defaults[1], '')
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatMethods(cc.TestLogWarnings, PysatBase):
    """Integration tests for using ocbpy on pysat pandas data."""

    def setUp(self):
        """Initialize the test class."""
        PysatBase.setUp(self)
        cc.TestLogWarnings.setUp(self)
        return

    def tearDown(self):
        """Tear down after each test."""
        cc.TestLogWarnings.tearDown(self)
        PysatBase.tearDown(self)
        return

    def test_empty_ocb_warning(self):
        """Test log error raised if OCB has no records."""
        # Load the boundaries and instrument
        self.load_instrument()

        # Reload the boundary without records
        self.ocb_kw['stime'] = self.test_module._test_dates['']['']
        self.ocb_kw['etime'] = self.ocb_kw['stime']
        self.load_boundaries()

        # Test adding OCBs
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name=self.pysat_alt, ocb=self.ocb,
                                  max_sdiff=self.del_time)

        # Check for the logging error
        self.lwarn = u"no data in Boundary file"
        self.eval_logging_message()
        return

    def test_add_ocb_to_metadata(self):
        """Test the metadata adding routine."""
        # Load the data and boundaries
        self.load_instrument()

        # Test the metadata operation
        ocb_pysat.add_ocb_to_metadata(self.test_inst, self.ocb_key,
                                      self.pysat_key, notes="test notes")

        self.notes = 'test notes'
        self.eval_ocb_metadata()
        return

    def test_add_ocb_to_metadata_vector(self):
        """Test the metadata adding routine for vector data."""
        # Load the data and boundaries
        self.load_instrument()

        # Test the metadata operation
        ocb_pysat.add_ocb_to_metadata(self.test_inst, self.ocb_key,
                                      self.pysat_key,
                                      notes="test notes scaled using None",
                                      isvector=True)

        self.notes = 'test notes scaled using None'
        self.eval_ocb_metadata()
        return

    def test_no_overwrite_metadata(self):
        """Test the overwrite block on metadata adding routine."""
        # Load the data and boundaries
        self.load_instrument()

        # Test the metadata overwriting failure
        self.test_add_ocb_to_metadata()
        ocb_pysat.add_ocb_to_metadata(self.test_inst, self.ocb_key,
                                      self.pysat_key, notes="test notes two",
                                      overwrite=False)

        self.lwarn = u"Boundary data already has metadata"
        self.eval_logging_message()
        return

    def test_overwrite_metadata(self):
        """Test the overwrite permission on metadata adding routine."""
        # Load the data and boundaries
        self.load_instrument()

        # Test the metadata overwriting success
        self.test_add_ocb_to_metadata()
        ocb_pysat.add_ocb_to_metadata(self.test_inst, self.ocb_key,
                                      self.pysat_key, notes="test notes two",
                                      overwrite=True)

        meta = self.test_inst.meta
        self.assertRegex(meta[self.ocb_key][meta.labels.notes],
                         "test notes two")
        return

    def test_add_ocb_to_data_ocb_obj(self):
        """Test adding ocb to pysat data using the loaded OCB object."""
        # Load the data and boundaries
        self.load_instrument()

        # Test adding OCBs
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name=self.pysat_alt, ocb=self.ocb,
                                  max_sdiff=self.del_time)

        self.set_new_keys(exclude_r_corr=False)
        self.assertIn('r_corr', self.pysat_keys)
        self.pysat_keys[self.pysat_keys.index("r_corr")] = None
        self.test_ocb_added()
        return

    def test_add_ocb_to_data_ocb_file(self):
        """Test adding ocb to pysat data using one boundary file name."""
        # Load the data and boundaries
        self.load_instrument()
        if 'filename' in self.ocb_kw.keys():
            self.ocb_kw['ocbfile'] = self.ocb_kw['filename']
        else:
            self.ocb_kw['ocbfile'] = 'dual'

        # Test adding OCBs using filename instead of OCB object
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name=self.pysat_alt,
                                  max_sdiff=self.del_time, **self.ocb_kw)

        self.set_new_keys(exclude_r_corr=False)
        self.assertIn('r_corr', self.pysat_keys)
        self.pysat_keys[self.pysat_keys.index("r_corr")] = None
        self.test_ocb_added()
        return

    def test_add_ocb_to_data_southern_hemisphere(self):
        """Test successful identification of southern hemisphere only."""
        # Load the data and boundaries
        self.load_instrument()
        if 'filename' in self.ocb_kw.keys():
            self.ocb_kw['ocbfile'] = self.ocb_kw['filename']
        else:
            self.ocb_kw['ocbfile'] = 'dual'

        # Don't set the hemisphere
        del self.ocb_kw['hemisphere']

        # Ensure all data is in the southern hemisphere and the greatest
        # value is identically zero
        new_lat = np.array(self.test_inst[self.pysat_lat])
        new_lat[new_lat > 0] *= -1.0
        imax = np.where(np.nanmax(new_lat) == new_lat)  # Needed for model data
        new_lat[imax] = 0.0
        if(not self.test_inst.pandas_format
           and self.pysat_lat in self.test_inst.data.coords):
            self.test_inst.data = self.test_inst.data.assign_coords(
                {self.pysat_lat: new_lat})
        else:
            try:
                self.test_inst[self.pysat_lat] = new_lat
            except ValueError:
                self.test_inst[self.pysat_lat].values = new_lat

        # Add the OCB data to the Instrument and evaluate the output
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name=self.pysat_alt,
                                  max_sdiff=self.del_time, **self.ocb_kw)
        self.test_ocb_added()
        return

    def test_add_ocb_to_data_evar(self):
        """Test adding ocb to pysat with E-field related variables."""
        # Load the data and boundaries
        self.load_instrument()

        # Add the OCB with electrically scaled variables
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name=self.pysat_alt,
                                  evar_names=[self.pysat_key],
                                  ocb=self.ocb, max_sdiff=self.del_time)

        self.set_new_keys(exclude_r_corr=False)
        self.assertIn('r_corr', self.pysat_keys)
        self.pysat_keys[self.pysat_keys.index("r_corr")] = None

        self.test_ocb_added()
        return

    def test_add_ocb_to_data_curl_evar(self):
        """Test adding ocb to pysat with Curl E-field related variables."""
        # Load the data and boundaries
        self.load_instrument()

        # Add the OCB with curl-scaled variables
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name=self.pysat_alt,
                                  curl_evar_names=[self.pysat_var2],
                                  ocb=self.ocb, max_sdiff=self.del_time)

        self.set_new_keys(exclude_r_corr=False)
        self.assertIn('r_corr', self.pysat_keys)
        self.pysat_keys[self.pysat_keys.index("r_corr")] = None

        self.test_ocb_added()
        return

    def test_add_ocb_to_data_evar_vect(self):
        """Test adding ocb to pysat with Curl E-field related vectors."""
        # Load the data and boundaries
        self.load_instrument()

        # Add the OCB with electrically scaled vectors
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name=self.pysat_alt,
                                  evar_names=['vect_evar'],
                                  vector_names={
                                      'vect_evar':
                                      {'vect_n': self.pysat_key,
                                       'vect_e': self.pysat_var2,
                                       'dat_name': 'vect',
                                       'dat_units': 'm/s'}},
                                  ocb=self.ocb, max_sdiff=self.del_time)

        self.set_new_keys(exclude_r_corr=True)
        self.test_ocb_added()
        return

    def test_add_ocb_to_data_curl_evar_vect(self):
        """Test adding ocb to pysat with Curl E-field related vectors."""
        # Load the data and boundaries
        self.load_instrument()

        # Add the OCB with curl-scaled vectors
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name=self.pysat_alt,
                                  curl_evar_names=['vect_cevar'],
                                  vector_names={
                                      'vect_cevar':
                                      {'vect_n': self.pysat_key,
                                       'vect_e': self.pysat_var2,
                                       'dat_name': 'vect',
                                       'dat_units': 'm/s'}},
                                  ocb=self.ocb, max_sdiff=self.del_time)

        self.set_new_keys(exclude_r_corr=True)
        self.test_ocb_added()
        return

    def test_add_ocb_to_data_custom_vect(self):
        """Test adding ocb to pysat with custom scaled variables."""
        # Load the data and boundaries
        self.load_instrument()

        # Add the OCB with a custom scaling function
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name=self.pysat_alt, vector_names={
                                      'vect_cust':
                                      {'vect_n': self.pysat_key,
                                       'vect_e': self.pysat_var2,
                                       'dat_name': 'vect',
                                       'dat_units': 'm/s',
                                       'scale_func': None}},
                                  ocb=self.ocb, max_sdiff=self.del_time)

        self.set_new_keys(exclude_r_corr=True)
        self.test_ocb_added()
        return

    def test_add_ocb_to_data_all_types(self):
        """Test adding ocb to pysat with E-field, Curl, and Vector data."""
        # Load the data and boundaries
        self.load_instrument()

        # Add the OCB with multiple inputs
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name=self.pysat_alt,
                                  evar_names=[self.pysat_key],
                                  curl_evar_names=[self.pysat_var2],
                                  vector_names={
                                      'vect_cust':
                                      {'vect_n': self.pysat_key,
                                       'vect_e': self.pysat_var2,
                                       'dat_name': 'vect',
                                       'dat_units': 'm/s',
                                       'scale_func': None}},
                                  ocb=self.ocb, max_sdiff=self.del_time)

        self.set_new_keys(exclude_r_corr=True)
        self.test_ocb_added()
        return

    def test_add_ocb_to_data_no_file(self):
        """Test adding ocb to pydat when no OCB file or data is provided."""
        # Load the data and boundaries
        self.load_instrument()

        # Add the OCB without a filename
        self.ocb_kw['ocbfile'] = None

        with self.assertRaisesRegex(ValueError,
                                    "can't determine desired boundary type"):
            ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                      height_name=self.pysat_alt, **self.ocb_kw,
                                      max_sdiff=self.del_time)
        return

    def test_add_ocb_to_data_bad_hemisphere_selfset(self):
        """Test failure of a pysat.Instrument to specify a hemisphere."""
        # Load the data and boundaries
        self.load_instrument()

        del self.ocb_kw['hemisphere']

        with self.assertRaisesRegex(
                ValueError, 'cannot process observations from both '):
            ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                      height_name=self.pysat_alt, **self.ocb_kw)
        return

    def test_bad_pysat_inst(self):
        """Test failure of a bad pysat.Instrument in pysat functions."""
        # Load the data and boundaries
        self.load_instrument()

        # Set the function and input data
        func_dict = {
            ocb_pysat.add_ocb_to_data: [None, self.pysat_lat, "mlt"],
            ocb_pysat.add_ocb_to_metadata: [
                None, "{:s}_mlt".format(self.ocb_name), "mlt"]}

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
        # Load the data and boundaries
        self.load_instrument()

        with self.assertRaisesRegex(ValueError,
                                    'unknown magnetic latitude name bad'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "bad", "mlt",
                                      ocb=self.ocb)

    def test_add_ocb_to_data_bad_mlt(self):
        """Test failure of unknown mlt key in add_ocb_to_data."""
        # Load the data and boundaries
        self.load_instrument()

        with self.assertRaisesRegex(ValueError,
                                    'unknown magnetic local time name bad'):
            ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "bad",
                                      ocb=self.ocb)
        return

    def test_add_ocb_to_data_bad_evar(self):
        """Test failure of unknown E field key in add_ocb_to_data."""
        # Load the data and boundaries
        self.load_instrument()

        with self.assertRaisesRegex(ValueError,
                                    'at least one unknown E field name'):
            ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                      evar_names=["bad"], ocb=self.ocb)
        return

    def test_add_ocb_to_data_bad_curl(self):
        """Test failure of unknown E field key in add_ocb_to_data."""
        # Load the data and boundaries
        self.load_instrument()

        with self.assertRaisesRegex(ValueError,
                                    'at least one unknown E field name'):
            ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                      curl_evar_names=["bad"], ocb=self.ocb)
        return

    def test_add_ocb_to_data_bad_vector_scale(self):
        """Test failure of missing scaling function in add_ocb_to_data."""
        # Load the data and boundaries
        self.load_instrument()

        with self.assertRaisesRegex(ValueError,
                                    'missing scaling function for bad'):
            ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                      vector_names={'bad': {'vect_n': 'bad_n',
                                                            'vect_e': 'bad_e',
                                                            'dat_name': 'bad',
                                                            'dat_units': ''}},
                                      ocb=self.ocb)
        return

    def test_add_ocb_to_data_bad_vector_name(self):
        """Test failure of missing scaling function in add_ocb_to_data."""
        # Load the data and boundaries
        self.load_instrument()

        with self.assertRaisesRegex(ValueError,
                                    'unknown vector name bad_n'):
            ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                      height_name=self.pysat_alt,
                                      evar_names=['bad'],
                                      vector_names={
                                          'bad':
                                          {'vect_n': 'bad_n',
                                           'vect_e': self.pysat_key,
                                           'dat_name': 'bad',
                                           'dat_units': ''}},
                                      ocb=self.ocb)
        return

    def test_bad_height_array(self):
        """Test failure with a badly shaped height input."""
        # Load the data and boundaries
        self.load_instrument()

        # Define an array height input
        height = np.full(shape=(self.test_inst[self.pysat_lat].values.shape[0],
                                3), fill_value=200.0)

        # Raise the desired error
        with self.assertRaisesRegex(ValueError, 'unexpected height shape'):
            ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                      height=height, ocb=self.ocb,
                                      max_sdiff=self.del_time)
        return


@unittest.skipIf(no_pysat, "pysat not installed, cannot test routines")
class TestPysatMethodsEAB(TestPysatMethods):
    """Integration tests for using ocbpy.EABoundary on pysat pandas data."""

    def setUp(self):
        """Initialize the test class."""
        # Initalize the defaults
        super().setUp()

        # Update the class defaults
        self.test_file = path.join(cc.test_dir, "test_north_eab")
        self.ocb_kw = {"filename": self.test_file,
                       "instrument": "image", "hemisphere": 1}
        self.ocb_class = ocbpy.EABoundary
        self.ocb_name = "eab"
        self.ocb_key = "_".join([self.ocb_name, "test"])

        return


@unittest.skipIf(no_pysat, "pysat not installed, cannot test routines")
class TestPysatMethodsDual(TestPysatMethods):
    """Integration tests for using ocbpy.DualBoundary on pysat pandas data."""

    def setUp(self):
        """Initialize the test class."""
        # Initalize the defaults
        super().setUp()

        # Update the method defaults
        self.ocb_class = ocbpy.DualBoundary
        self.ocb_name = "dualb"
        self.ocb_key = "_".join([self.ocb_name, "test"])
        self.ocb_kw = {'ocb_filename': self.test_file,
                       'ocb_instrument': 'image',
                       'eab_filename': self.test_file.replace('_ocb', '_eab'),
                       'eab_instrument': 'image', 'hemisphere': 1}
        self.rec_ind = 0

        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatMethodsXarray(TestPysatMethods):
    """Integration tests for using ocbpy on pysat ND or 2D Xarray data."""

    def setUp(self):
        """Initialize the test class."""
        # Initalize the defaults
        super().setUp()

        # Update the method defaults
        # TODO(#130) remove version checking by updating minimum supported pysat
        if version.Version(pysat.__version__) < version.Version('3.1.0'):
            self.test_module = pysat.instruments.pysat_testing2d_xarray
        else:
            self.test_module = pysat.instruments.pysat_ndtesting

        return

    def test_bad_vector_shape(self):
        """Test failure with badly shaped vector data."""
        # Load the data and boundaries
        self.load_instrument()

        # Raise the desired value error
        with self.assertRaisesRegex(ValueError,
                                    'mismatched dimensions for VectorData'):
            ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                      height_name=self.pysat_alt, ocb=self.ocb,
                                      vector_names={
                                          'profile':
                                          {'vect_n': 'profiles',
                                           'vect_e': 'variable_profiles',
                                           'dat_name': 'profile',
                                           'dat_units': 'unit',
                                           'scale_func': None}},
                                      max_sdiff=self.del_time)
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatMethodsModel(TestPysatMethods):
    """Integration tests for using ocbpy on pysat model data."""

    def setUp(self):
        """Initialize the test class."""
        # Initalize the defaults
        super().setUp()

        # Update the method default
        self.test_module = pysat.instruments.pysat_testmodel
        self.pysat_alt = ''
        return

    def test_add_ocb_to_data_evar_with_alt_coord(self):
        """Test adding ocb to pysat with altitude coordinate."""
        # Load the data and boundaries
        self.load_instrument()

        # Add the OCB with E-scaled variables and the height variable
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name='altitude',
                                  evar_names=[self.pysat_var2],
                                  ocb=self.ocb, max_sdiff=self.del_time)

        self.set_new_keys(exclude_r_corr=False)
        self.assertIn('r_corr', self.pysat_keys)
        self.pysat_keys[self.pysat_keys.index("r_corr")] = None

        self.test_ocb_added()
        return

    def test_add_ocb_to_data_evar_with_alt_array(self):
        """Test adding ocb to pysat with array altitude input."""
        # Load the data and boundaries
        self.load_instrument()

        # Define an array height input
        height = np.full(shape=self.test_inst[self.pysat_key].values.shape,
                         fill_value=200.0)

        # Add the OCB with E-scaled variables
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height=height, evar_names=[self.pysat_key],
                                  ocb=self.ocb, max_sdiff=self.del_time)

        self.set_new_keys(exclude_r_corr=False)
        self.assertIn('r_corr', self.pysat_keys)
        self.pysat_keys[self.pysat_keys.index("r_corr")] = None

        self.test_ocb_added()
        return

    def test_add_ocb_to_data_evar_with_alt_value(self):
        """Test adding ocb to pysat with array altitude input."""
        # Load the data and boundaries
        self.load_instrument()

        # Define a non-coordinate height
        dims = list(self.test_inst[self.pysat_key].dims)
        dims.reverse()
        shape = list(self.test_inst[self.pysat_key].values.shape)
        shape.reverse()
        self.test_inst.data = self.test_inst.data.assign(
            {'dummy1_alt': (dims, np.full(shape=shape, fill_value=200.0))})

        # Add the OCB with E-scaled variables
        ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                  height_name="dummy1_alt",
                                  evar_names=[self.pysat_key],
                                  ocb=self.ocb, max_sdiff=self.del_time)

        self.set_new_keys(exclude_r_corr=False)
        self.assertIn('r_corr', self.pysat_keys)
        self.pysat_keys[self.pysat_keys.index("r_corr")] = None

        self.test_ocb_added()
        return

    def test_mismatched_vector_data(self):
        """Test that vector data with different dimensions fails."""
        # Load the data and boundaries
        self.load_instrument()

        with self.assertRaisesRegex(ValueError,
                                    'vector variables must all have the same'):
            ocb_pysat.add_ocb_to_data(self.test_inst, self.pysat_lat, "mlt",
                                      height_name=self.pysat_alt,
                                      evar_names=['vect_evar'],
                                      vector_names={
                                          'vect_evar':
                                          {'vect_n': self.pysat_key,
                                           'vect_e': self.pysat_lat,
                                           'dat_name': 'vect',
                                           'dat_units': 'm/s'}},
                                      ocb=self.ocb, max_sdiff=self.del_time)
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatCustMethods(PysatBase):
    """Integration tests for using ocbpy as a custom function with pysat pandas.

    """

    def setUp(self):
        """Initialize the unit tests for using the pysat.Custom methods."""
        PysatBase.setUp(self)

        # Set the custom defaults
        self.cust_kwargs = {'mlat_name': self.pysat_lat, 'mlt_name': 'mlt',
                            'height_name': self.pysat_alt,
                            'max_sdiff': self.del_time}

        return

    def tearDown(self):
        """Clean the test environment."""
        PysatBase.tearDown(self)

        del self.cust_kwargs
        return

    def test_load(self):
        """Test the pysat file loading without custom functions."""
        # Load the data and boundaries
        self.load_instrument()

        # Test the load for expected variables (with or without a custom func)
        self.assertFalse(self.test_inst.empty,
                         msg="No data loaded for {:}".format(self.test_inst))
        self.assertIn(self.cust_kwargs['mlat_name'], self.test_inst.variables)
        self.assertIn('mlt', self.test_inst.variables)
        return

    def test_cust_add_ocb_to_data(self):
        """Test adding ocb to pysat data using the loaded OCB object."""
        # Load the boundaries
        self.load_boundaries()

        # Set the second input set by type
        if self.ocb_name == "dualb":
            kw2 = ['ocbfile', 'instrument', 'hemisphere', 'ocb_filename',
                   'eab_filename']
            val2 = ['dual', 'image', 1, self.test_file,
                    self.ocb_kw['eab_filename']]
        else:
            kw2 = ['ocbfile', 'instrument', 'hemisphere']
            val2 = [self.test_file, 'image', 1]

        # Cycle through the different custom inputs
        for kw, val in [(['ocb'], [self.ocb]), (kw2, val2),
                        (['ocb', 'evar_names'], [self.ocb, [self.pysat_key]]),
                        (['ocb', 'curl_evar_names'],
                        [self.ocb, [self.pysat_var2]])]:
            # Ensure the record index is correct
            self.ocb.rec_ind = self.rec_ind

            # Update the Instrument to include a custom function
            self.pysat_kw['custom'] = [{'function': ocb_pysat.add_ocb_to_data,
                                        'kwargs': dict(self.cust_kwargs)}]

            # Update the custom function kwargs
            for i, kw_val in enumerate(kw):
                self.pysat_kw['custom'][0]['kwargs'][kw_val] = val[i]

            with self.subTest(cust_kwargs=self.pysat_kw['custom'][0]['kwargs']):
                # Load and test the defaults
                self.test_load()

                # Test the additional outputs
                self.set_new_keys(exclude_r_corr=False)
                self.assertIn('r_corr', self.pysat_keys,
                              msg="r_corr missing from {:}".format(
                                  self.test_inst.meta))
                self.pysat_keys[self.pysat_keys.index("r_corr")] = None
                self.test_ocb_added()
        return

    def test_cust_add_ocb_to_data_vect(self):
        """Test adding OCB vector data to pysat with load."""

        # Load the boundaries
        self.load_boundaries()

        # Update the custom variables
        self.cust_kwargs['ocb'] = self.ocb

        # Cycle through the different custom inputs
        for kw, val in [(['evar_names', 'vector_names'],
                         [['vect_evar'],
                          {'vect_evar': {'vect_n': self.pysat_var2,
                                         'vect_e': self.pysat_var2,
                                         'dat_name': 'vect',
                                         'dat_units': 'm/s'}}]),
                        (['curl_evar_names', 'vector_names'],
                         [['vect_cevar'],
                          {'vect_cevar': {'vect_n': self.pysat_var2,
                                          'vect_e': self.pysat_key,
                                          'dat_name': 'vect',
                                          'dat_units': 'm/s'}}]),
                        (['vector_names'],
                         [{'vect_cust': {'vect_n': self.pysat_key,
                                         'vect_e': self.pysat_var2,
                                         'dat_name': 'vect', 'dat_units': 'm/s',
                                         'scale_func': None}}]),
                        (['evar_names', 'curl_evar_names', 'vector_names'],
                         [[self.pysat_var2], [self.pysat_key],
                          {'vect_cust': {'vect_n': self.pysat_key,
                                         'vect_e': self.pysat_key,
                                         'dat_name': 'vect', 'dat_units': 'm/s',
                                         'scale_func': None}}])]:
            # Ensure the record index is correct
            self.ocb.rec_ind = self.rec_ind

            # Update the Instrument to include a custom function
            self.pysat_kw['custom'] = [{'function': ocb_pysat.add_ocb_to_data,
                                        'kwargs': dict(self.cust_kwargs)}]

            # Update the custom function kwargs
            for i, kw_val in enumerate(kw):
                self.pysat_kw['custom'][0]['kwargs'][kw_val] = val[i]

            with self.subTest(cust_kwargs=self.pysat_kw['custom'][0]['kwargs']):
                # Load and test the defaults
                self.test_load()

                # Test the additional data
                self.set_new_keys(exclude_r_corr=True)
                self.test_ocb_added()
        return

    def test_cust_add_ocb_to_data_no_file(self):
        """Test adding ocb to pysat with load using no OCB file or data."""
        # Update the custom kwargs
        self.cust_kwargs['ocbfile'] = None
        self.cust_kwargs['instrument'] = 'image'
        self.cust_kwargs['hemisphere'] = 1

        # Update the Instrument to include a custom function
        self.pysat_kw['custom'] = [{'function': ocb_pysat.add_ocb_to_data,
                                    'kwargs': self.cust_kwargs}]

        # Test the correct error is raised
        with self.assertRaisesRegex(ValueError,
                                    "can't determine desired boundary type"):
            self.test_load()

        return

    def test_cust_add_ocb_to_data_bad_inputs(self):
        """Test failure of unknown inputs in add_ocb_to_data custom func."""
        # Cycle through different bad inputs
        for kw, msg in [('mlt_name', 'unknown magnetic local time name bad'),
                        ('evar_names', 'at least one unknown E field name'),
                        ('curl_evar_names',
                         'at least one unknown E field name')]:
            # Update the input kwargs
            self.pysat_kw['custom'] = [{'function': ocb_pysat.add_ocb_to_data,
                                        'kwargs': dict(self.cust_kwargs)}]
            self.pysat_kw['custom'][0]['kwargs'][kw] = 'bad'

            with self.subTest(cust_kwargs=self.pysat_kw['custom'][0]['kwargs']):
                # Test the failure when loading data
                with self.assertRaisesRegex(ValueError, msg):
                    self.test_load()
        return

    def test_cust_add_ocb_to_data_bad_vector_scale(self):
        """Test failure of missing scaling function in custom func."""
        self.load_boundaries()
        self.cust_kwargs['vector_names'] = {'bad': {'vect_n': 'bad_n',
                                                    'vect_e': 'bad_e',
                                                    'dat_name': 'bad',
                                                    'dat_units': ''}}
        self.cust_kwargs['ocb'] = self.ocb
        self.pysat_kw['custom'] = [{'function': ocb_pysat.add_ocb_to_data,
                                    'kwargs': self.cust_kwargs}]

        with self.assertRaisesRegex(ValueError,
                                    'missing scaling function for bad'):
            self.test_load()
        return

    def test_cust_add_ocb_to_data_bad_vector_name(self):
        """Test failure of missing scaling function in custom func."""
        self.load_boundaries()
        self.cust_kwargs['ocb'] = self.ocb
        self.cust_kwargs['evar_names'] = ['bad']
        self.cust_kwargs['vector_names'] = {'bad':
                                            {'vect_n': 'bad_n',
                                             'vect_e': self.pysat_key,
                                             'dat_name': 'bad',
                                             'dat_units': ''}}
        self.pysat_kw['custom'] = [{'function': ocb_pysat.add_ocb_to_data,
                                    'kwargs': self.cust_kwargs}]

        with self.assertRaisesRegex(ValueError, 'unknown vector name bad_n'):
            self.test_load()
        return


@unittest.skipIf(no_pysat, "pysat not installed, cannot test routines")
class TestPysatCustMethodsEAB(TestPysatCustMethods):
    """Integration tests for pysat pandas through custom with EABs."""

    def setUp(self):
        """Initialize the unit tests for using the pysat.Custom methods."""
        # Initalize the defaults
        super().setUp()

        # Update the class defaults
        self.test_file = path.join(cc.test_dir, "test_north_eab")
        self.ocb_kw = {"filename": self.test_file,
                       "instrument": "image", "hemisphere": 1}
        self.ocb_class = ocbpy.EABoundary
        self.ocb_name = 'eab'
        self.ocb_key = "_".join([self.ocb_name, "test"])
        return


@unittest.skipIf(no_pysat, "pysat not installed, cannot test routines")
class TestPysatCustMethodsDual(TestPysatCustMethods):
    """Integration tests for pysat pandas through custom with dual boundaries.

    """

    def setUp(self):
        """Initialize the unit tests for using the pysat.Custom methods."""
        # Initalize the defaults
        super().setUp()

        # Update the class defaults
        self.ocb_class = ocbpy.DualBoundary
        self.ocb_name = 'dualb'
        self.ocb_key = "_".join([self.ocb_name, "test"])
        self.ocb_kw = {'ocb_filename': self.test_file,
                       'ocb_instrument': 'image',
                       'eab_filename': self.test_file.replace('north_ocb',
                                                              'north_eab'),
                       'eab_instrument': 'image', 'hemisphere': 1}
        self.rec_ind = 0
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatCustMethodsXarray(TestPysatCustMethods):
    """Integration tests for using ocbpy as a custom function with pysat Xarray.

    """

    def setUp(self):
        """Initialize the unit tests for using the pysat.Custom methods."""
        # Initalize the defaults
        super().setUp()

        # Update the class defaults
        # TODO(#130) remove version checking by updating minimum supported pysat
        if version.Version(pysat.__version__) < version.Version('3.1.0'):
            self.test_module = pysat.instruments.pysat_testing2d_xarray
        else:
            self.test_module = pysat.instruments.pysat_ndtesting
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatCustMethodsModel(TestPysatCustMethods):
    """Integration tests for using ocbpy with pysat model data."""

    def setUp(self):
        """Initialize the tests for using the pysat.Custom methods."""
        # Initalize the defaults
        super().setUp()

        # Update the class defaults
        self.test_module = pysat.instruments.pysat_testmodel
        self.pysat_alt = ''
        self.cust_kwargs['height_name'] = self.pysat_alt
        return


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatReshape(unittest.TestCase):
    """Unit tests for the `reshape_pad_mask_flatten` function."""

    def setUp(self):
        """Set up the test environment."""
        self.shape = [4, 5]
        self.data = xr.DataArray(data=np.ones(shape=self.shape),
                                 dims=['x', 'y'])
        self.flat = None
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.shape, self.data, self.flat
        return

    def test_reshape_pad_mask_flatten(self):
        """Test successful data padding, masking, and flattening."""

        for mask_shape, dims in [(self.shape, ['x', 'y']),
                                 (tuple(reversed(self.shape)), ['y', 'x']),
                                 (self.shape + [3], ['x', 'y', 'z'])]:
            mask = xr.DataArray(data=np.ones(shape=mask_shape, dtype=bool),
                                dims=dims)
            with self.subTest(mask_dims=dims):
                self.flat = ocb_pysat.reshape_pad_mask_flatten(
                    self.data, mask)

                self.assertEqual(self.flat.shape, np.prod(mask_shape))
                self.assertGreaterEqual(self.flat.shape, np.prod(self.shape))
        return

    def test_bad_reshape_pad_mask_flatten(self):
        """Test data padding, masking, and flattening failure."""
        # Change the mask shape, and add one extra dimension
        self.shape[0] += 2

        for mask_shape, dims, verr in [
                (self.shape, ['x', 'y'], "different shapes for the same dim"),
                (self.shape + [3], ['x', 'y', 'z'], "vector variables must "),
                (self.shape, ['z', 'y'], "vector variables must all have")]:
            mask = xr.DataArray(data=np.ones(shape=mask_shape, dtype=bool),
                                dims=dims)

            with self.subTest(mask_shape=mask_shape, dims=dims):
                with self.assertRaisesRegex(ValueError, verr):
                    ocb_pysat.reshape_pad_mask_flatten(self.data, mask)
        return
