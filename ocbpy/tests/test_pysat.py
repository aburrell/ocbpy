#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocb_scaling class and functions
"""

import unittest
import numpy as np

import logbook

import ocbpy
try:
    import pysat
    import ocbpy.instruments.pysat_instruments as ocb_pysat
    no_pysat = False
except ImportError:
    no_pysat = True

@unittest.skipIf(not no_pysat, "pysat installed, cannot test failure")
class TestPysatFailure(unittest.TestCase):
    def setUp(self):
        """ No initialization needed """
        pass

    def tearDown(self):
        """ No teardown needed"""
        pass

    def test_import_failure(self):
        """ Test pysat import failure"""

        with self.assertRaisesRegexp(ImportError, 'unable to load the pysat'):
            import ocbpy.instruments.pysat_instruments as ocb_pysat


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatStructure(unittest.TestCase):
    def setUp(self):
        """ No setup needed"""
        pass

    def tearDown(self):
        """ No teardown needed"""
        pass

    def test_add_ocb_to_data_defaults(self):
        """ test the add_ocb_to_data function defaults"""

        defaults = ocb_pysat.add_ocb_to_data.func_defaults

        for i in [0, 1, 3]:
            self.assertListEqual(defaults[i], list())

        for i in [4, 5]:
            self.assertIsNone(defaults[i])

        for i, val in enumerate([600, 7, 8.0, 23.0, 10.0, 0.15]):
            self.assertEqual(defaults[i+6], val)

        self.assertDictEqual(defaults[2], dict())

    def test_add_ocb_to_metadata_defaults(self):
        """ test the add_ocb_to_metadata function defaults"""

        defaults = ocb_pysat.add_ocb_to_metadata.func_defaults

        for i in [0, 2]:
            self.assertFalse(defaults[i])

        self.assertRegexpMatches(defaults[1], '')


@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        from os import path
        import pandas as pds
        
        ocb_dir = path.split(ocbpy.__file__)[0]
        self.test_file = path.join(ocb_dir, "tests", "test_data",
                                   "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb = ocbpy.OCBoundary(self.test_file)
        self.ocb.rec_ind = 27

        self.test_inst = pysat.Instrument('pysat', 'testing', tag='50400',
                                          clean_level='clean',
                                          update_files=True, \
                            file_date_range=pds.date_range(self.ocb.dtime[0],
                                                           self.ocb.dtime[-1],
                                                           freq='1D'))
        self.test_inst.load(date=self.ocb.dtime[self.ocb.rec_ind])
        self.test_inst['latitude'][np.sign(self.test_inst.data['latitude']) !=
                                   self.ocb.hemisphere] *= -1.0

        self.ocb_out = {'latitude': {'max': 86.86586, 'min': 1.26930},
                        'mlt': {'max': 18.47234, 'min': 14.60623},
                        'dummy1': {'max': 17.21520, 'min': 0.0},
                        'dummy2': {'max': 16.37831, 'min': 0.0}}

        self.log_handler = logbook.TestHandler()
        self.log_handler.push_thread()

    def tearDown(self):
        self.log_handler.pop_thread()
        del self.test_file, self.log_handler, self.test_inst, self.ocb
        del self.ocb_out

    def test_add_ocb_to_metadata(self):
        """ Test the metadata adding routine
        """
        ocb_pysat.add_ocb_to_metadata(self.test_inst, "ocb_test", "dummy1",
                                      notes="test notes")

        self.test_ocb_metadata(meta=self.test_inst.meta, notes='test notes')

    def test_add_ocb_to_metadata_vector(self):
        """ Test the metadata adding routine for vector data
        """
        ocb_pysat.add_ocb_to_metadata(self.test_inst, "ocb_test", "dummy1",
                                      notes="test notes", isvector=True)

        self.test_ocb_metadata(meta=self.test_inst.meta, notes='test notes',
                               isvector=True)

    def test_ocb_metadata(self, meta=None, ocb_key='ocb_test',
                          pysat_key='dummy1', notes=None, isvector=False):
        """ Test that metadata was added properly
        """

        if meta is None:
            self.assertTrue(True)
        else:
            self.assertTrue(ocb_key in meta.keys())
            if pysat_key is not None:
                self.assertTrue(pysat_key in meta.keys())

            if isvector:
                # Test the fill value
                self.assertIsNone(meta[ocb_key][meta.fill_label])
            elif pysat_key is not None:
                # Test the elements that are identical
                for ll in [meta.units_label, meta.scale_label, meta.min_label,
                           meta.max_label, meta.fill_label]:
                    try:
                        if np.isnan(meta[pysat_key][ll]):
                            self.assertTrue(np.isnan(meta[ocb_key][ll]))
                        else:
                            self.assertEqual(meta[ocb_key][ll],
                                             meta[pysat_key][ll])
                    except TypeError:
                        self.assertRegexpMatches(meta[ocb_key][ll],
                                                 meta[pysat_key][ll])

            # Test the elements that have "OCB" appended to the text
            for ll in [meta.name_label, meta.axis_label, meta.plot_label]:
                sline = meta[ocb_key][ll].split(" ")
                self.assertRegexpMatches(sline[0], "OCB")
                if not isvector and pysat_key is not None:
                    self.assertRegexpMatches(" ".join(sline[1:]),
                                             meta[pysat_key][ll])

            # Test the remaining elements
            self.assertEqual(meta[ocb_key][meta.desc_label].find("Open Closed"),
                             0)
            if notes is not None:
                self.assertRegexpMatches(meta[ocb_key][meta.notes_label], notes)

            del ll, sline

    def test_missing_metadata(self):
        """ Test the metadata adding routine when pysat object has no metadata
        """
        ocb_pysat.add_ocb_to_metadata(pysat.Instrument(), "ocb_test", "dummy1")

        self.assertEqual(len(self.log_handler.formatted_records), 1)
        self.assertTrue(self.log_handler.formatted_records[0].find( \
                                        'original data has no metadata') > 0)

    def test_no_overwrite_metadata(self):
        """ Test the overwrite block on metadata adding routine
        """
        self.test_add_ocb_to_metadata()
        ocb_pysat.add_ocb_to_metadata(self.test_inst, "ocb_test", "dummy1",
                                      notes="test notes two", overwrite=False)

        self.assertEqual(len(self.log_handler.formatted_records), 1)
        self.assertTrue(self.log_handler.formatted_records[0].find( \
                                        'OCB data already has metadata') > 0)

    def test_overwrite_metadata(self):
        """ Test the overwrite permission on metadata adding routine
        """
        self.test_add_ocb_to_metadata()
        ocb_pysat.add_ocb_to_metadata(self.test_inst, "ocb_test", "dummy1",
                                      notes="test notes two", overwrite=True)

        meta = self.test_inst.meta
        self.assertRegexpMatches(meta['ocb_test'][meta.notes_label],
                                 "test notes two")

        del meta

    def test_ocb_added(self, test_inst=None, added_keys=[], pysat_keys=[],
                       isvector=[], nkeys=0):
        """ Test if OCB data was added correctly
        """

        if test_inst is None:
            self.assertTrue(True)
        else:
            if len(isvector) < len(added_keys):
                isvector = [False for ocb_key in added_keys]
            
            self.assertEqual(len(added_keys), nkeys)

            for i, ocb_key in enumerate(added_keys):
                # Test to see that data was added
                self.assertTrue(ocb_key in test_inst.data.columns)
                                
                # Test the metadata
                self.test_ocb_metadata(meta=test_inst.meta, ocb_key=ocb_key,
                                       pysat_key=pysat_keys[i],
                                       isvector=isvector[i])

                # Test to see that data within 10 minutes of the test OCBs has
                # OCB locations and other data is NaN
                match_data = test_inst[ocb_key]
                if not isvector[i]:
                    match_data = match_data[np.isfinite(match_data)]
                else:
                    match_data = match_data[np.not_equal(match_data, None)]

                self.assertEqual(len(match_data), 1770)
                self.assertLessEqual(abs(match_data.index[0] -
                                         self.ocb.dtime[27]).total_seconds(),
                                     600.0)
                if isvector[i]:
                    self.assertTrue(isinstance(match_data[0],
                                               ocbpy.ocb_scaling.VectorData))
                elif pysat_keys[i] is not None:
                    self.assertAlmostEqual(match_data.max(),
                                           self.ocb_out[pysat_keys[i]]['max'],
                                           places=5)
                    self.assertAlmostEqual(match_data.min(),
                                           self.ocb_out[pysat_keys[i]]['min'],
                                           places=5)

            del ocb_key, match_data

    def test_add_ocb_to_data_ocb_obj(self):
        """ Test adding ocb to pysat data using the loaded OCB object
        """

        added = ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                          ocb=self.ocb)

        pysat_keys = [aa.split("_ocb")[0] for aa in added]
        self.assertTrue('r_corr' in pysat_keys)
        pysat_keys[pysat_keys.index("r_corr")] = None

        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, isvector=[False, False],
                            nkeys=3)
        del added, pysat_keys, aa

    def test_add_ocb_to_data_ocb_file(self):
        """ Test adding ocb to pysat data using the OCB file name
        """

        added = ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                          ocbfile=self.test_file)

        pysat_keys = [aa.split("_ocb")[0] for aa in added]
        self.assertTrue('r_corr' in pysat_keys)
        pysat_keys[pysat_keys.index("r_corr")] = None

        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, isvector=[False, False],
                            nkeys=3)
        del added, pysat_keys, aa

    def test_add_ocb_to_data_evar(self):
        """ Test adding ocb to pysat with E-field related variables
        """

        added = ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                          evar_names=['dummy1'], ocb=self.ocb)

        pysat_keys = [aa.split("_ocb")[0] for aa in added]
        self.assertTrue('r_corr' in pysat_keys)
        pysat_keys[pysat_keys.index("r_corr")] = None

        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys,
                            isvector=[False, False, False], nkeys=4)
        del added, pysat_keys, aa

    def test_add_ocb_to_data_curl_evar(self):
        """ Test adding ocb to pysat with Curl E-field related variables
        """

        added = ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                          curl_evar_names=['dummy2'],
                                          ocb=self.ocb)

        pysat_keys = [aa.split("_ocb")[0] for aa in added]
        self.assertTrue('r_corr' in pysat_keys)
        pysat_keys[pysat_keys.index("r_corr")] = None

        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys,
                            isvector=[False, False, False], nkeys=4)
        del added, pysat_keys, aa

    def test_add_ocb_to_data_evar_vect(self):
        """ Test adding ocb to pysat with Curl E-field related variables
        """

        added = ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                          evar_names=['vect_evar'],
                                          vector_names={'vect_evar':
                                                        {'aacgm_n': 'dummy1',
                                                         'aacgm_e': 'dummy2',
                                                         'dat_name': 'vect',
                                                         'dat_units': 'm/s'}},
                                          ocb=self.ocb)

        pysat_keys = list()
        isvector = list()
        for aa in added:
            pp = aa.split("_ocb")[0]
            if pp not in self.test_inst.data.columns:
                pp = 'dummy1'
                isvector.append(True)
            else:
                isvector.append(False)
            pysat_keys.append(pp if pp != "r_corr" else None)
        
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, isvector=isvector, nkeys=4)
        
        del added, pysat_keys, aa, pp, isvector

    def test_add_ocb_to_data_curl_evar_vect(self):
        """ Test adding ocb to pysat with Curl E-field related variables
        """

        added = ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                          curl_evar_names=['vect_cevar'],
                                          vector_names={'vect_cevar':
                                                        {'aacgm_n': 'dummy1',
                                                         'aacgm_e': 'dummy2',
                                                         'dat_name': 'vect',
                                                         'dat_units': 'm/s'}},
                                          ocb=self.ocb)

        pysat_keys = list()
        isvector = list()
        for aa in added:
            pp = aa.split("_ocb")[0]
            if pp not in self.test_inst.data.columns:
                pp = 'dummy1'
                isvector.append(True)
            else:
                isvector.append(False)
            pysat_keys.append(pp if pp != "r_corr" else None)
        
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, isvector=isvector, nkeys=4)
        
        del added, pysat_keys, aa, pp, isvector

    def test_add_ocb_to_data_custom_vect(self):
        """ Test adding ocb to pysat with custom scaled variables
        """

        added = ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                          vector_names={'vect_cust':
                                                        {'aacgm_n': 'dummy1',
                                                         'aacgm_e': 'dummy2',
                                                         'dat_name': 'vect',
                                                         'dat_units': 'm/s',
                                                         'scale_func': None}},
                                          ocb=self.ocb)

        pysat_keys = list()
        isvector = list()
        for aa in added:
            pp = aa.split("_ocb")[0]
            if pp not in self.test_inst.data.columns:
                pp = 'dummy1'
                isvector.append(True)
            else:
                isvector.append(False)
            pysat_keys.append(pp if pp != "r_corr" else None)
        
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, isvector=isvector, nkeys=4)
        
        del added, pysat_keys, aa, pp, isvector

    def test_add_ocb_to_data_all_types(self):
        """ Test adding ocb to pysat with E-field, Curl, and Vector data
        """

        added = ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                          evar_names=['dummy1'],
                                          curl_evar_names=['dummy2'],
                                          vector_names={'vect_cust':
                                                        {'aacgm_n': 'dummy1',
                                                         'aacgm_e': 'dummy2',
                                                         'dat_name': 'vect',
                                                         'dat_units': 'm/s',
                                                         'scale_func': None}},
                                          ocb=self.ocb)

        pysat_keys = list()
        isvector = list()
        for aa in added:
            pp = aa.split("_ocb")[0]
            if pp not in self.test_inst.data.columns:
                pp = 'dummy1'
                isvector.append(True)
            else:
                isvector.append(False)
            pysat_keys.append(pp if pp != "r_corr" else None)
        
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, isvector=isvector, nkeys=6)
        
        del added, pysat_keys, aa, pp, isvector

    def test_add_ocb_to_data_no_file(self):
        """ Test adding ocb to pydat data when no OCB file or data is provided
        """
        ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                  ocbfile=None)

        self.assertEqual(len(self.log_handler.formatted_records), 1)
        self.assertTrue(self.log_handler.formatted_records[0].find( \
                                        'no data in OCB file') > 0)

    def test_add_ocb_to_data_bad_mlat(self):
        """ Test failure of unknown mlat key in add_ocb_to_data
        """

        with self.assertRaisesRegexp(ValueError,
                                     'unknown magnetic latitude name: mlat'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "mlat", "mlt",
                                      ocb=self.ocb)

    def test_add_ocb_to_data_bad_mlt(self):
        """ Test failure of unknown mlt key in add_ocb_to_data
        """

        with self.assertRaisesRegexp(ValueError,
                                     'unknown magnetic local time name: bad'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "bad",
                                      ocb=self.ocb)

    def test_add_ocb_to_data_bad_evar(self):
        """ Test failure of unknown E field key in add_ocb_to_data
        """

        with self.assertRaisesRegexp(ValueError,
                                     'at least one unknown E field name'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                      evar_names=["bad"], ocb=self.ocb)

    def test_add_ocb_to_data_bad_curl(self):
        """ Test failure of unknown E field key in add_ocb_to_data
        """

        with self.assertRaisesRegexp(ValueError,
                                     'at least one unknown E field name'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                      curl_evar_names=["bad"], ocb=self.ocb)

    def test_add_ocb_to_data_bad_vector_scale(self):
        """ Test failure of missing scaling function in add_ocb_to_data
        """

        with self.assertRaisesRegexp(ValueError,
                                     'missing scaling function for: bad'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                      vector_names={'bad': {'aacgm_n': 'bad_n',
                                                            'aacgm_e': 'bad_e',
                                                            'dat_name': 'bad',
                                                            'dat_units':''}},
                                     ocb=self.ocb)

    def test_add_ocb_to_data_bad_vector_name(self):
        """ Test failure of missing scaling function in add_ocb_to_data
        """

        with self.assertRaisesRegexp(ValueError,
                                     'unknown vector name: bad_n'):
            ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                      evar_names=['bad'],
                                      vector_names={'bad': {'aacgm_n': 'bad_n',
                                                            'aacgm_e': 'dummy1',
                                                            'dat_name': 'bad',
                                                            'dat_units':''}},
                                     ocb=self.ocb)


if __name__ == '__main__':
    unittest.main()

