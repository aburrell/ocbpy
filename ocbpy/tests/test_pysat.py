#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocb_scaling class and functions
"""

import ocbpy.instruments.pysat as ocb_pysat
import unittest
import numpy as np
import logbook

try:
    import pysat
    no_pysat = False
except ImportError:
    no_pysat = True

@unittest.skipIf(no_pysat, "pysat not installed")
class TestPysatMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        from os import path
        import ocbpy
        
        ocb_dir = path.split(ocbpy.__file__)[0]
        self.test_file = path.join(ocb_dir, "tests", "test_data",
                                   "test_north_circle")
        self.assertTrue(path.isfile(self.test_file))
        self.ocb = ocbpy.OCBoundary(self.test_file)
        self.ocb.rec_ind = 27

        self.test_inst = pysat.Instrument('pysat', 'testing', tag='ocb',
                                          sat_id=self.ocb.instrument,
                                          clean_level='clean',
                                          update_files=True)
        self.test_inst.load(date=self.ocb.dtime[self.ocb.rec_ind])
        self.test_inst['latitude'][np.sign(self.test_inst.data['latitude']) !=
                                   self.ocb.hemisphere] *= -1.0

        self.ocb_out = {'latitude': {'max': 87.23394, 'min': 24.17184},
                        'mlt': {'max': 23.14262, 'min': 8.74717},
                        'dummy1': {'max': 16.74727, 'min': 8.20059},
                        'dummy2': {'max': 24.651598, 'min': 0.0}}

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
            self.assertTrue(pysat_key in meta.keys())

            if isvector:
                # Test the fill value
                self.assertIsNone(meta[ocb_key][meta.fill_label])
            else:
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
                if not isvector:
                    self.assertRegexpMatches(sline[1], meta[pysat_key][ll])

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
                       nkeys=0):
        """ Test if OCB data was added correctly
        """

        if test_inst is None:
            self.assertTrue(True)
        else:
            self.assertEqual(len(added_keys), nkeys)

            for i, ocb_key in enumerate(added_keys):
                # Test to see that data was added
                self.assertTrue(ocb_key in test_inst.data.columns)
                                
                # Test the metadata
                isvector = False if pysat_keys[i] in test_inst.data.columns \
                    else True
                    
                self.test_ocb_metadata(meta=test_inst.meta, ocb_key=ocb_key,
                                       pysat_key=pysat_keys[i],
                                       isvector=isvector)

                # Test to see that data within 10 minutes of the test OCBs has
                # OCB locations and other data is NaN
                match_data = test_inst[ocb_key][np.isfinite(test_inst[ocb_key])]
                self.assertEqual(len(match_data), 2040)
                self.assertLessEqual(abs(match_data.index[0] -
                                         self.ocb.dtime[27]).total_seconds(),
                                     600.0)
                if isvector:
                    self.assertTrue(isinstance(match_data[0],
                                               ocbpy.ocb_scaling.VectorData))
                else:
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
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, nkeys=2)
        del added, pysat_keys, aa

    def test_add_ocb_to_data_ocb_file(self):
        """ Test adding ocb to pysat data using the OCB file name
        """

        added = ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                          ocbfile=self.test_file)
        pysat_keys = [aa.split("_ocb")[0] for aa in added]
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, nkeys=2)
        del added, pysat_keys, aa

    def test_add_ocb_to_data_evar(self):
        """ Test adding ocb to pysat with E-field related variables
        """

        added = ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                          evar_names=['dummy1'], ocb=self.ocb)
        pysat_keys = [aa.split("_ocb")[0] for aa in added]
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, nkeys=3)
        del added, pysat_keys, aa

    def test_add_ocb_to_data_curl_evar(self):
        """ Test adding ocb to pysat with Curl E-field related variables
        """

        added = ocb_pysat.add_ocb_to_data(self.test_inst, "latitude", "mlt",
                                          curl_evar_names=['dummy2'],
                                          ocb=self.ocb)
        pysat_keys = [aa.split("_ocb")[0] for aa in added]
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, nkeys=3)
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
        for aa in added:
            pp = aa.split("_ocb")[0]
            if pp not in self.test_inst.data.columns:
                pp = 'dummy1'
            pysat_keys.append(pp)
        
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, nkeys=3)
        
        del added, pysat_keys, aa, pp

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
        for aa in added:
            pp = aa.split("_ocb")[0]
            if pp not in self.test_inst.data.columns:
                pp = 'dummy1'
            pysat_keys.append(pp)
        
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, nkeys=3)
        
        del added, pysat_keys, aa, pp

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
        for aa in added:
            pp = aa.split("_ocb")[0]
            if pp not in self.test_inst.data.columns:
                pp = 'dummy1'
            pysat_keys.append(pp)
        
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, nkeys=3)
        
        del added, pysat_keys, aa, pp

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
        for aa in added:
            pp = aa.split("_ocb")[0]
            if pp not in self.test_inst.data.columns:
                pp = 'dummy1'
            pysat_keys.append(pp)
        
        self.test_ocb_added(test_inst=self.test_inst, added_keys=added,
                            pysat_keys=pysat_keys, nkeys=5)
        
        del added, pysat_keys, aa, pp

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

