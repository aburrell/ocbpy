#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the boundary EABoundary class."""

import datetime
from io import StringIO
import logging
import numpy
from os import path
import platform
import unittest

import ocbpy
from . import test_boundary_ocb as test_ocb


class TestDualBoundaryLogFailure(unittest.TestCase):
    """Test the logging messages raised by the DualBoundary class."""

    def setUp(self):
        """Initialize the test class."""
        self.lwarn = ""
        self.lout = ""
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)
        return

    def tearDown(self):
        """Tear down the test case."""
        del self.lwarn, self.lout, self.log_capture
        return

    def test_bad_instrument_name(self):
        """Test OCB initialization with bad instrument name."""
        self.lwarn = "OCB instrument must be a string"

        # Initialize the DualBoundary class with bad instrument names
        for bad_inst in [1, None, True]:
            for btype in ["eab", "ocb"]:
                # Define the kwarg input
                val = {"_".join([btype, "instrument"]): bad_inst}
                with self.subTest(val=val):
                    # Initalize the DualBoundary class
                    bound = ocbpy.DualBoundary(**val)
                    subclass = getattr(bound, btype)

                    # Test the values for the sub-class
                    self.assertIsNone(subclass.filename)
                    self.assertIsNone(subclass.instrument)

                    self.lout = self.log_capture.getvalue()

                    # Test logging error message for each bad initialization
                    self.assertRegex(self.lout, self.lwarn)
        return

    def test_bad_filename(self):
        """Test initialization with a bad default file/instrument pairing."""
        self.lwarn = "name provided is not a file\ncannot open OCB file [hi]"

        # Try to load data with a non-existant file name
        for btype in ["eab", "ocb"]:
            # Define the kwarg input
            val = {"_".join([btype, "filename"]): "hi"}
            with self.subTest(val=val):
                # Initalize the DualBoundary class
                bound = ocbpy.DualBoundary(**val)
                subclass = getattr(bound, btype)

                # Test the values for the sub-class
                self.assertIsNone(subclass.filename)

                self.lout = self.log_capture.getvalue()

                # Test logging error message for each bad initialization
                self.assertTrue(
                    self.lout.find(self.lwarn) >= 0,
                    msg="logging output {:} != expected output {:}".format(
                        repr(self.lout), repr(self.lwarn)))

        return


class TestDualBoundaryInstruments(test_ocb.TestOCBoundaryInstruments):
    """Test the DualBoundary handling of different instruments."""

    def setUp(self):
        """Initialize the instrument information."""
        self.test_class = ocbpy.DualBoundary
        self.test_dir = path.join(path.dirname(ocbpy.__file__), "tests",
                                  "test_data")
        self.inst_attrs = {"image": ["year", "soy", "num_sectors", "a",
                                     "r_err", "fom"],
                           "ampere": ["date", "time", "x", "y", "fom"],
                           "dmsp-ssj": ["date", "time", "sc", "x", "y", "fom",
                                        "x_1", "x_2", "y_1", "y_2"]}
        self.not_attrs = {"image": ["date", "time", "x", "y", "x_1", "x_2",
                                    "y_1", "y_2", "sc"],
                          "ampere": ["year", "soy", "x_1", "y_1", "x_2",
                                     "y_2", "sc", "num_sectors", "a",
                                     "r_err"],
                          "dmsp-ssj": ["year", "soy", "num_sectors", "a",
                                       "r_err"]}
        self.inst_init = [{"eab_instrument": "dmsp-ssj", "hemisphere": 1,
                           "eab_filename": path.join(self.test_dir,
                                                     "dmsp-ssj_north_out.eab"),
                           "ocb_instrument": "image",
                           "ocb_filename": path.join(self.test_dir,
                                                     "test_north_circle")},
                          {"eab_instrument": "dmsp-ssj", "hemisphere": 1,
                           "eab_filename": path.join(self.test_dir,
                                                     "dmsp-ssj_north_out.eab"),
                           "ocb_instrument": "dmsp-ssj",
                           "ocb_filename": path.join(self.test_dir,
                                                     "dmsp-ssj_north_out.ocb")},
                          {"eab_instrument": "dmsp-ssj", "hemisphere": -1,
                           "eab_filename": path.join(self.test_dir,
                                                     "dmsp-ssj_south_out.eab"),
                           "ocb_instrument": "ampere",
                           "ocb_filename": path.join(self.test_dir,
                                                     "test_south_circle")}]
        self.ocb = None
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.test_dir, self.inst_attrs, self.inst_init, self.ocb
        del self.test_class
        return


class TestDualBoundaryMethodsGeneral(test_ocb.TestOCBoundaryMethodsGeneral):
    """Test the DualBoundary general methods."""

    def setUp(self):
        """Initialize the test environment."""
        self.test_class = ocbpy.DualBoundary
        test_dir = path.join(path.dirname(ocbpy.__file__), "tests", "test_data")
        self.set_empty = {"ocb_filename": path.join(test_dir, "test_empty"),
                          "eab_filename": path.join(test_dir, "test_empty"),
                          "ocb_instrument": "image", "eab_instrument": "image",
                          "hemisphere": 1}
        self.set_default = {"ocb_filename":
                            path.join(test_dir, "dmsp-ssj_north_out.ocb"),
                            "eab_filename":
                            path.join(test_dir, "dmsp-ssj_north_out.eab"),
                            "ocb_instrument": "dmsp-ssj",
                            "eab_instrument": "dmsp-ssj", "hemisphere": 1,
                            "max_delta": 600}
        self.ocb = None
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.set_empty, self.set_default, self.ocb
        return

    def test_repr_string(self):
        """Test __repr__ method string."""
        # Initalize the class object
        self.ocb = self.test_class(**self.set_default)

        # Get the representation of the class object and split by subclasses
        rocb = repr(self.ocb).split("ocb=")

        # Test the name of the repr object
        self.assertRegex(rocb[0], self.test_class.__name__)

        # Test each set kwarg has the expected value
        for val in self.set_default.keys():
            with self.subTest(val=val):
                i = 0 if val.find("eab") == 0 else 1

                # Construct the expected string
                if(val.find("filename") >= 0
                   and platform.system().lower().find("windows") >= 0):
                    # Windows has trouble recognizing the filename in Regex
                    test_str = "filename="
                elif i == 0 or val.find("ocb") == 0:
                    test_str = "=".join([val.split("_")[-1],
                                         repr(self.set_default[val])])
                else:
                    test_str = "=".join([val, repr(self.set_default[val])])

                # Test the correct part of the repr output.
                self.assertRegex(rocb[i], test_str)
        return

    def test_short_str(self):
        """Test the default class print output."""
        self.ocb = self.test_class(**self.set_default)
        self.ocb.records = 1

        self.assertRegex(self.ocb.__str__(), "1 good boundary pairs from")
        return

    def test_bad_hemisphere(self):
        """Test raises ValueError with conflicting boundary hemispheres."""

        # Add Boundary objects with conflicting hemispheres to input
        self.set_default['ocb'] = ocbpy.OCBoundary(
            filename=self.set_default["ocb_filename"],
            instrument=self.set_default["ocb_instrument"], hemisphere=1)
        self.set_default['eab'] = ocbpy.EABoundary(
            filename=self.set_default["eab_filename"].replace('north', 'south'),
            instrument=self.set_default["eab_instrument"], hemisphere=-1)

        with self.assertRaisesRegex(ValueError, "mismatched hemisphere"):
            self.test_class(**self.set_default)
        return

    def test_bad_rfunc_inst(self):
        """Test failure setting default rfunc for unknown instrument."""

        for bound in ['eab', 'ocb']:
            with self.subTest(bound=bound):
                self.set_empty["_".join([bound, "instrument"])] = "bad"

                with self.assertRaisesRegex(ValueError, "unknown instrument"):
                    self.ocb = self.test_class(**self.set_empty)
        return

    def test_no_file_str(self):
        """Test the unset class print output."""

        for bound in ['eab', 'ocb', 'both']:
            out_str = []
            # Update the kwarg input
            if bound in ['eab', 'both']:
                self.set_default['eab_filename'] = None
                out_str.append("No {:s} file specified".format(
                    ocbpy.EABoundary.__name__))
            if bound in ['ocb', 'both']:
                self.set_default['ocb_filename'] = None
                out_str.append("No {:s} file specified".format(
                    ocbpy.OCBoundary.__name__))

            # Initalise the object
            self.ocb = self.test_class(**self.set_default)

            for val in out_str:
                with self.subTest(val=val):
                    # Test the output string
                    self.assertRegex(self.ocb.__str__(), val)
        return

    def test_nofile_init(self):
        """Ensure that the class can be initialised without loading a file."""
        self.ocb = self.test_class(eab_filename=None, ocb_filename=None)

        self.assertIsNone(self.ocb.eab.filename)
        self.assertIsNone(self.ocb.ocb.filename)
        self.assertIsNone(self.ocb.dtime)
        self.assertEqual(self.ocb.records, 0)
        return

    def test_attrs(self):
        """Test the default attributes in a DualBoundary object."""
        self.ocb = self.test_class(**self.set_default)

        # Ensure standard attributes are present
        for attr in ['eab', 'ocb', 'max_delta', 'records', 'rec_ind',
                     'dtime', 'ocb_ind', 'eab_ind']:
            self.assertTrue(hasattr(self.ocb, attr),
                            msg="missing attr: {:}".format(attr))

        return

    def test_first_good(self):
        """Test to see that the first good point returns the expected index."""

        # Initalize the object
        self.ocb = self.test_class(**self.set_default)

        # Evaluate the record index
        self.assertEqual(self.ocb.rec_ind, 0)
        return

    def test_assign_rec_ind_valid(self):
        """Test to see that the record indices are all updated together."""

        # Initalize the object
        self.ocb = self.test_class(**self.set_default)

        # Cycle through all valid indices
        for i in range(self.ocb.records):
            with self.subTest(i=i):
                self.ocb.rec_ind = i

                # Ensure the sub-class indices are correct
                self.assertEqual(self.ocb.ocb.rec_ind, self.ocb.ocb_ind[i])
                self.assertEqual(self.ocb.eab.rec_ind, self.ocb.eab_ind[i])
        return

    def test_assign_rec_ind_max(self):
        """Test to see that setting `rec_ind` above the max is consistent."""

        # Initialize the object
        self.ocb = self.test_class(**self.set_default)

        # Set the index to above the allowed range
        self.ocb.rec_ind = self.ocb.records

        # Ensure the sub-class indices are also above the allowed range
        self.assertEqual(self.ocb.ocb.rec_ind, self.ocb.ocb.records)
        self.assertEqual(self.ocb.eab.rec_ind, self.ocb.eab.records)
        return

    def test_assign_rec_ind_unset(self):
        """Test to see that unsetting `rec_ind` above is consistent."""

        # Initialize the object
        self.ocb = self.test_class(**self.set_default)

        # Set the index to below the allowed range
        self.ocb.rec_ind = -1

        # Ensure the sub-class indices are also above the allowed range
        self.assertLessEqual(self.ocb.ocb.rec_ind, -1)
        self.assertLessEqual(self.ocb.eab.rec_ind, -1)
        return

    def test_custom_ind_selection(self):
        """Test use of custom boundary selection criteria."""
        # Initialize the boundary object and the comparison value
        self.ocb = self.test_class(**self.set_default)
        ocb_val = self.ocb.ocb.phi_cent[self.ocb.ocb.rec_ind]
        eab_val = self.ocb.eab.phi_cent[self.ocb.eab.rec_ind]
        ocb_ind = self.ocb.ocb_ind[self.ocb.rec_ind]
        eab_ind = self.ocb.eab_ind[self.ocb.rec_ind]

        # Cycle over each comparison method
        for method in ['max', 'min', 'maxeq', 'mineq', 'equal']:
            # Set the selection input and re-set the boundary index
            kwargs = {"ocb_kwargs": {'phi_cent': (method, ocb_val)},
                      "eab_kwargs": {'phi_cent': (method, eab_val)}}

            # Test the custom selection
            with self.subTest(kwargs=kwargs):
                # Reset the good indices and the first good index
                self.ocb.set_good_ind(**kwargs)
                self.ocb.rec_ind = -1
                self.ocb.get_next_good_ind()

                # Evaluate the current first good index
                if method.find('eq') >= 0:
                    self.assertEqual(self.ocb.ocb.rec_ind, ocb_ind)
                    self.assertEqual(self.ocb.eab.rec_ind, eab_ind)
                else:
                    self.assertGreater(self.ocb.eab.rec_ind, eab_ind)
                    self.assertGreater(self.ocb.ocb.rec_ind, ocb_ind)
        return


class TestDualBoundaryMethodsLocation(unittest.TestCase):
    """Test the DualBoundary location methods."""

    def setUp(self):
        """Initialize the test environment."""
        self.test_dir = path.join(path.dirname(ocbpy.__file__),
                                  "tests", "test_data")
        self.set_default = {"ocb_instrument": "dmsp-ssj",
                            "eab_instrument": "dmsp-ssj",
                            "max_delta": 60}
        self.dual = None

        # Set data test values
        self.bounds = [numpy.array([75.70330362, 72.32137562, 66.56506657,
                                    72.32137562, 61.31754838, 60.49909062,
                                    64.04636507, 65.51113019]),
                       numpy.array([17.49152542, 16.6779661, 8.94915254,
                                    16.6779661, 10.57627119, 10.98305085,
                                    14.6440678, 15.05084746]),
                       numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan,
                                    numpy.nan, numpy.nan, 73.85195423,
                                    numpy.nan]),
                       numpy.array([17.49152542, 16.6779661, 8.94915254,
                                    16.6779661, 10.57627119, 10.98305085,
                                    14.6440678, 15.05084746])]
        self.lat = {1: [50.0, 65.0, 66.0, 66.0, 75.0, 75.0, 75.0, 75.0],
                    -1: [-75.0, -75.0, -73.1, -73.5, -65.0, -65.0, -50.0]}
        self.mlt = {1: [17.49152542, 16.6779661, 8.94915254, 16.6779661,
                        10.57627119, 10.98305085, 14.6440678, 15.05084746],
                    -1: [8.54237288, 17.08474576, 20.8, 20.9, 7.72881356,
                         17.89830508, 18.30508475]}

        self.nlat = {1: [42.2702821, 57.52102977, 63.45670812, 58.40596869,
                         68.549302, 71.92618313, 75.36450117, 71.9639519],
                     -1: [-75.20091699603744, -77.43241520800147, -65.63470973,
                          -66.20944239, -63.02562069687926, -60.95171308470224,
                          -46.4345919807581]}
        self.nmlt = {1: [18.80644991, 18.85841629, 6.67391185, 18.96355496,
                         6.39130306, 6.7183392, 18.93115591, 19.3161857],
                     -1: [6.94756463, 18.65574753, 21.93370602, 22.03753856,
                          6.79635684, 18.83239647, 18.88673298]}
        self.olat = {1: [2.77770149, 41.62137725, 40.15557199, 43.59274117,
                         68.549302, 71.92618313, 75.36450117, 71.9639519],
                     -1: [-75.200917, -77.43241521, -69.91446894, -70.17230283,
                          -64.26664886, -66.49974046, -50.923274]}
        self.rcorr = 0.0
        self.scaled_r = {1: [64.0, 64.0, 64.0, 64.0, 10.0, 10.0, 16.0, 10.0],
                         -1: [16.0, 16.0, 10.0, 10.0, 64.0, 64.0, 64.0]}
        self.unscaled_r = {1: [75.70330362, 72.32137562, 66.56506657,
                               72.32137562, numpy.nan, numpy.nan, 6.769,
                               numpy.nan],
                           -1: [15.785, 15.785, 5.10033853, 5.12896403,
                                66.00490331, 68.25074784, 68.91414059]}
        self.out = []
        return

    def tearDown(self):
        """Clean up the test environment."""
        del self.test_dir, self.set_default, self.dual, self.mlt, self.lat
        del self.out, self.bounds
        return

    def update_default_kwargs(self, hemisphere=1):
        """Update the default kwargs to include filenames.

        Parameters
        ----------
        hemisphere : int
            Flag indicating hemisphere, 1 is North and -1 is South (default=1)

        Notes
        -----
        only updates filenames if the instrument is DMSP-SSJ

        """

        hemi_name = {1: "north", -1: "south"}

        # Set the hemisphere
        self.set_default['hemisphere'] = hemisphere

        if hemisphere in hemi_name.keys():
            # Set the filenames
            for bname in ['eab', 'ocb']:
                ikey = '_'.join([bname, 'instrument'])
                fkey = '_'.join([bname, 'filename'])
                if self.set_default[ikey] == 'dmsp-ssj':
                    self.set_default[fkey] = path.join(
                        self.test_dir, 'dmsp-ssj_{:s}_out.{:s}'.format(
                            hemi_name[hemisphere], bname))

        return

    def eval_coords(self, hemisphere=1, tol=1.0e-7, ind=None, revert=False,
                    radius=False):
        """Evaluate the results of the `normal_coord` method.

        Parameters
        ----------
        hemisphere : int
            Flag indicating hemisphere, 1 is North and -1 is South (default=1)
        tol : float
            Tolerance for error in floating point results (default=1.0e-7)
        ind : int or NoneType
            Evaluate a single value instead of an array
        revert : bool
            False if calculating normalized coordinates, True if going
            from normalized to geo/magnetic coordinates (default=False)
        radius : bool
            False if calculating coordinates, True if calculating radius
            (default=False)

        """

        # Get the expected lat/lt output
        lat_str = "normalized latitude"
        mlt_str = "normalized MLT"
        if revert:
            rlat = self.lat[hemisphere]
            rmlt = self.mlt[hemisphere]
            rshape = 2
        else:
            if radius:
                rlat = self.scaled_r[hemisphere]
                rmlt = self.unscaled_r[hemisphere]
                lat_str = "scaled radius"
                mlt_str = "unscaled radius"
                rshape = 2
            else:
                rlat = self.nlat[hemisphere]
                rmlt = self.nmlt[hemisphere]
                rshape = 4

        # Evalute the output shape
        self.assertEqual(len(self.out), rshape,
                         msg="unexpected number of output values")

        # Evaluate based on array or float output
        if ind is None:
            self.assertEqual(len(numpy.isnan(self.out[0])),
                             len(numpy.isnan(rlat)))
            if not numpy.isnan(rlat).all():
                self.assertTrue(
                    numpy.less(abs(self.out[0] - rlat), tol,
                               where=~numpy.isnan(rlat)).all(),
                    msg="unequal {:s}: {:} != {:}".format(lat_str, self.out[0],
                                                          rlat))
            self.assertEqual(len(numpy.isnan(self.out[1])),
                             len(numpy.isnan(rmlt)))
            if not numpy.isnan(rmlt).all():
                self.assertTrue(
                    numpy.less(abs(self.out[1] - rmlt), tol,
                               where=~numpy.isnan(rmlt)).all(),
                    msg="unequal {:s}: {:} != {:}".format(mlt_str, self.out[1],
                                                          rmlt))

            if rshape > 2:
                self.assertEqual(len(numpy.isnan(self.out[2])),
                                 len(numpy.isnan(self.olat[hemisphere])))
                if not numpy.isnan(self.out[2]).all():
                    self.assertTrue(
                        numpy.less(abs(self.out[2] - self.olat[hemisphere]),
                                   tol, where=~numpy.isnan(self.out[2])).all(),
                        msg="unequal OCB latitude: {:} != {:}".format(
                            self.out[2], self.olat[hemisphere]))

                if numpy.isnan(self.rcorr):
                    self.assertTrue(numpy.isnan(self.out[3]).all(),
                                    msg="{:} is not NaN".format(self.out[3]))
                else:
                    self.assertTrue(
                        numpy.less(abs(self.out[3] - self.rcorr), tol,
                                   where=~numpy.isnan(self.out[3])).all(),
                        msg="unequal radial correction: {:} != {:}".format(
                            self.out[3], self.rcorr))
        else:
            if numpy.isnan(self.out[0]):
                self.assertTrue(numpy.isnan(rlat[ind]))
            else:
                self.assertAlmostEqual(
                    self.out[0], rlat[ind],
                    msg="unequal {:s}: {:} != {:}".format(lat_str, self.out[0],
                                                          rlat[ind]))
            if numpy.isnan(self.out[1]):
                self.assertTrue(numpy.isnan(rmlt[ind]))
            else:
                self.assertAlmostEqual(
                    self.out[1], rmlt[ind],
                    msg="unequal {:s}: {:} != {:}".format(mlt_str, self.out[1],
                                                          rmlt[ind]))

            if rshape > 2:
                if numpy.isnan(self.out[2]):
                    self.assertTrue(numpy.isnan(self.olat[hemisphere][ind]))
                else:
                    self.assertAlmostEqual(
                        self.out[2], self.olat[hemisphere][ind],
                        msg="unequal OCB latitude: {:} != {:}".format(
                            self.out[2], self.olat[hemisphere][ind]))

                if numpy.isnan(self.out[3]):
                    self.assertTrue(numpy.isnan(self.rcorr))
                else:
                    self.assertAlmostEqual(
                        self.out[3], self.rcorr,
                        msg="unequal radial correction: {:} != {:}".format(
                            self.out[3], self.rcorr))
        return

    def test_normal_coord_float_nan(self):
        """Test the normalisation calculation with NaN float input."""

        ind = 0
        self.rcorr = numpy.nan

        for hemi in [-1, 1]:
            # Initalize the object
            self.update_default_kwargs(hemisphere=hemi)
            self.dual = ocbpy.DualBoundary(**self.set_default)

            # Update the expected output
            self.nlat[hemi][ind] = numpy.nan
            self.nmlt[hemi][ind] = numpy.nan
            self.olat[hemi][ind] = numpy.nan

            # Evaluate the calculation
            with self.subTest(hemi=hemi):
                self.out = self.dual.normal_coord(numpy.nan, 0.0)
                self.eval_coords(hemisphere=hemi, ind=ind, revert=False)
        return

    def test_normal_coord_array_nan(self):
        """Test the normalisation calculation with all NaN array input."""

        for hemi in [-1, 1]:
            # Initalize the object
            self.update_default_kwargs(hemisphere=hemi)
            self.dual = ocbpy.DualBoundary(**self.set_default)

            # Update the input
            self.mlt[hemi] = numpy.full(shape=len(self.lat[hemi]),
                                        fill_value=numpy.nan)

            # Update the expected output
            self.nlat[hemi] = numpy.full(shape=len(self.lat[hemi]),
                                         fill_value=numpy.nan)
            self.nmlt[hemi] = numpy.full(shape=len(self.lat[hemi]),
                                         fill_value=numpy.nan)
            self.olat[hemi] = numpy.full(shape=len(self.lat[hemi]),
                                         fill_value=numpy.nan)

            # Evaluate the calculation
            with self.subTest(hemi=hemi):
                self.out = self.dual.normal_coord(self.lat[hemi],
                                                  self.mlt[hemi])
                self.eval_coords(hemisphere=hemi, revert=False)
        return

    def test_normal_coord_float(self):
        """Test the normalisation calculation with float input."""

        for hemi in [-1, 1]:
            # Initalize the object
            self.update_default_kwargs(hemisphere=hemi)
            self.dual = ocbpy.DualBoundary(**self.set_default)

            for i, in_lat in enumerate(self.lat[hemi]):
                in_mlt = self.mlt[hemi][i]

                # Evaluate the calculation
                with self.subTest(hemi=hemi, in_lat=in_lat, in_mlt=in_mlt):
                    self.out = self.dual.normal_coord(in_lat, in_mlt)
                    self.eval_coords(hemisphere=hemi, ind=i, revert=False)
        return

    def test_normal_coord_array(self):
        """Test the normalisation calculation with array input."""

        for hemi in [-1, 1]:
            # Initalize the object
            self.update_default_kwargs(hemisphere=hemi)
            self.dual = ocbpy.DualBoundary(**self.set_default)

            # Evaluate the calculation
            with self.subTest(hemi=hemi):
                self.out = self.dual.normal_coord(self.lat[hemi],
                                                  self.mlt[hemi])
                self.eval_coords(hemisphere=hemi, revert=False)
        return

    def test_normal_coord_mag_label(self):
        """Test the normalisation calculation with good mag labels."""

        # Initalize the object
        hemi = 1
        self.update_default_kwargs(hemisphere=hemi)
        self.dual = ocbpy.DualBoundary(**self.set_default)

        for coords in ["magnetic", "Mag"]:
            # Evaluate the calculation
            with self.subTest(coords=coords):
                self.out = self.dual.normal_coord(
                    self.lat[hemi], self.mlt[hemi], coords=coords)
                self.eval_coords(hemisphere=hemi, revert=False)

        return

    def test_normal_coord_geodetic_label(self):
        """Test the normalisation calculation with geodetic data."""

        # Initalize the object
        hemi = 1
        self.update_default_kwargs(hemisphere=hemi)
        self.dual = ocbpy.DualBoundary(**self.set_default)

        # Set the expected output
        self.nlat[hemi][0] = 38.50196955983855
        self.nmlt[hemi][0] = 18.541058394108877
        self.olat[hemi][0] = -6.87103469312359

        # Evaluate the calculation
        self.out = self.dual.normal_coord(self.lat[hemi][0], self.mlt[hemi][0],
                                          coords="geodetic")
        self.eval_coords(hemisphere=hemi, ind=0, revert=False)

        return

    def test_normal_coord_geocentric_label(self):
        """Test the normalisation calculation with geocentric data."""

        # Initalize the object
        hemi = 1
        self.update_default_kwargs(hemisphere=hemi)
        self.dual = ocbpy.DualBoundary(**self.set_default)

        # Set the expected output
        self.nlat[hemi][0] = 38.67674104478799
        self.nmlt[hemi][0] = 18.544698246714468
        self.olat[hemi][0] = -6.402309989057102

        # Evaluate the calculation
        self.out = self.dual.normal_coord(self.lat[hemi][0], self.mlt[hemi][0],
                                          coords="geocentric")
        self.eval_coords(hemisphere=hemi, ind=0, revert=False)

        return

    def test_normal_coord_low_rec_ind(self):
        """Test the normalized location failure with a low record index."""
        # Initalize the object
        hemi = 1
        self.update_default_kwargs(hemisphere=hemi)
        self.dual = ocbpy.DualBoundary(**self.set_default)

        # Unset the indices
        self.dual.ocb.rec_ind = -1
        self.dual.eab.rec_ind = -1
        self.dual.rec_ind = -1

        # Run the calculation
        self.out = self.dual.normal_coord(self.lat[hemi][0], self.mlt[hemi][0])

        # Evaluate the output
        self.assertEqual(len(self.out), 4)
        self.assertTrue(numpy.all(numpy.isnan(self.out)))
        return

    def test_normal_coord_high_rec_ind(self):
        """Test the normalized location failure with a high record index."""
        # Initalize the object
        hemi = 1
        self.update_default_kwargs(hemisphere=hemi)
        self.dual = ocbpy.DualBoundary(**self.set_default)

        # Unset the indices
        self.dual.ocb.rec_ind = self.dual.ocb.records + 1
        self.dual.eab.rec_ind = self.dual.eab.records + 1
        self.dual.rec_ind = self.dual.records + 1

        # Run the calculation
        self.out = self.dual.normal_coord(self.lat[hemi][0], self.mlt[hemi][0])

        # Evaluate the output
        self.assertEqual(len(self.out), 4)
        self.assertTrue(numpy.all(numpy.isnan(self.out)))
        return

    def test_revert_coord_float(self):
        """Test the reversion calculation with float input."""

        for hemi in [-1, 1]:
            # Initalize the object
            self.update_default_kwargs(hemisphere=hemi)
            self.dual = ocbpy.DualBoundary(**self.set_default)

            for i, in_lat in enumerate(self.olat[hemi]):
                in_mlt = self.nmlt[hemi][i]

                # Evaluate the calculation
                with self.subTest(hemi=hemi, in_lat=in_lat, in_mlt=in_mlt):
                    self.out = self.dual.revert_coord(in_lat, in_mlt,
                                                      self.rcorr)
                    self.eval_coords(hemisphere=hemi, ind=i, revert=True)
        return

    def test_revert_coord_array(self):
        """Test the reversion calculation with array input."""

        for hemi in [-1, 1]:
            # Initalize the object
            self.update_default_kwargs(hemisphere=hemi)
            self.dual = ocbpy.DualBoundary(**self.set_default)

            # Evaluate the calculation
            with self.subTest(hemi=hemi):
                self.out = self.dual.revert_coord(self.olat[hemi],
                                                  self.nmlt[hemi], self.rcorr)
                self.eval_coords(hemisphere=hemi, revert=True)
        return

    def test_get_current_aacgm_boundary_unset(self):
        """Test retrieval of the unset EAB/OCB AACGM boundary locations."""

        # Initalize the object
        hemi = 1
        self.update_default_kwargs(hemisphere=hemi)
        self.dual = ocbpy.DualBoundary(**self.set_default)
        self.out = self.dual._get_current_aacgm_boundary()

        self.assertEqual(len(self.out), 4)
        for abound in self.out:
            with self.subTest(abound=abound):
                self.assertIsNone(abound)
        return

    def test_get_current_aacgm_boundary_set(self):
        """Test retrieval of the set EAB/OCB AACGM boundary locations."""

        # Initalize the object
        hemi = 1
        self.update_default_kwargs(hemisphere=hemi)
        self.dual = ocbpy.DualBoundary(**self.set_default)

        # Set the EAB and OCB AACGM boundaries
        self.dual.get_aacgm_boundary_lats(self.mlt[hemi])

        # Get the current boundaries
        self.out = self.dual._get_current_aacgm_boundary()

        # Evaluate the output
        self.assertEqual(len(self.out), len(self.bounds))
        for i, abound in enumerate(self.out):
            with self.subTest(abound=abound):
                # Ensure the expected number of boundary points are present
                self.assertEqual(len(abound), len(self.mlt[hemi]))

                # Ensure the expected values and fill values are returned
                self.assertTrue(
                    numpy.less(abs(abound - self.bounds[i]), 1e-7,
                               where=~numpy.isnan(abound)).all(),
                    msg="unexpected boundary: {:} != {:}".format(
                        abound, self.bounds[i]))
                self.assertEqual(len(numpy.isnan(abound)),
                                 len(numpy.isnan(self.bounds[i])))
        return

    def test_calc_r_bad_ind(self):
        """Test the scaled/unscaled radius calculation with a bad index."""
        # Initialize the data
        hemi = 1
        self.update_default_kwargs(hemisphere=hemi)
        self.dual = ocbpy.DualBoundary(**self.set_default)

        # Set the expected output
        self.scaled_r[hemi] = numpy.full(shape=len(self.lat[hemi]),
                                         fill_value=numpy.nan)
        self.unscaled_r[hemi] = numpy.full(shape=len(self.lat[hemi]),
                                           fill_value=numpy.nan)

        # Cycle through indices above and below the desired range
        for ind in [-1, self.dual.records]:
            with self.subTest(ind=ind):
                self.dual.rec_ind = ind
                self.out = self.dual.calc_r(self.nlat[hemi], self.nmlt[hemi],
                                            self.mlt[hemi], self.rcorr)
                self.eval_coords(hemisphere=hemi, radius=True)

        return

    def test_calc_r(self):
        """Test the scaled/unscaled radius calculation."""

        for hemi in [1, -1]:
            self.update_default_kwargs(hemisphere=hemi)
            self.dual = ocbpy.DualBoundary(**self.set_default)

            with self.subTest(hemi=hemi):
                self.out = self.dual.calc_r(self.nlat[hemi], self.nmlt[hemi],
                                            self.mlt[hemi], self.rcorr)
                self.eval_coords(hemisphere=hemi, radius=True)

        return

    def test_calc_r_float(self):
        """Test the scaled/unscaled radius calculation for float input."""

        for hemi in [1, -1]:
            self.update_default_kwargs(hemisphere=hemi)
            self.dual = ocbpy.DualBoundary(**self.set_default)

            for i, amlt in enumerate(self.mlt[hemi]):
                blat = self.nlat[hemi][i]
                bmlt = self.nmlt[hemi][i]

                with self.subTest(hemi=hemi, amlt=amlt, bmlt=bmlt, blat=blat):
                    self.out = self.dual.calc_r(blat, bmlt, amlt, self.rcorr)
                    self.eval_coords(hemisphere=hemi, ind=i, radius=True)
        return


class TestDualBoundaryFailure(unittest.TestCase):
    """Test that DualBoundary class failures raise appropriate errors."""

    def setUp(self):
        """Set up the test environment."""
        test_dir = path.join(path.dirname(ocbpy.__file__),
                             "tests", "test_data")
        self.set_default = {"ocb_instrument": "dmsp-ssj",
                            "eab_instrument": "dmsp-ssj",
                            "hemisphere": 1,
                            "ocb_filename": path.join(
                                test_dir, 'dmsp-ssj_north_out.ocb'),
                            "eab_filename": path.join(
                                test_dir, 'dmsp-ssj_north_out.eab')}
        self.dual = None

    def tearDown(self):
        """Clean up the test environment."""
        del self.set_default, self.dual
        return

    def test_bad_boundaries(self):
        """Test ValueError raised with bad OCB/EAB boundary combo."""

        self.set_default["eab_lat"] = 80.0

        with self.assertRaisesRegex(ValueError, "OCB must be poleward of the"):
            self.dual = ocbpy.DualBoundary(**self.set_default)

        return

    def test_bad_max_delta(self):
        """Test ValueError raised with bad `max_delta` kwarg value."""

        self.set_default["max_delta"] = -1.0

        with self.assertRaisesRegex(ValueError, "must be positive or zero"):
            self.dual = ocbpy.DualBoundary(**self.set_default)

        return
