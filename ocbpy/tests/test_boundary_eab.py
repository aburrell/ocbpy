#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Tests the boundary EABoundary class."""

import numpy
from os import path

import ocbpy
import ocbpy.tests.class_common as cc
import ocbpy.tests.test_boundary_ocb as test_ocb


class TestEABoundaryLogFailure(test_ocb.TestOCBoundaryLogFailure):
    """Test the logging messages raised by the EABoundary class."""

    def setUp(self):
        """Initialize the test class."""
        super().setUp()

        # Update the setup
        self.test_class = ocbpy.EABoundary
        self.inst_init = {"instrument": "image", "hemisphere": 1,
                          "filename": path.join(cc.test_dir,
                                                "test_north_ocb")}
        return


class TestEABoundaryInstruments(test_ocb.TestOCBoundaryInstruments):
    """Test the EABoundary handling of different instruments."""

    def setUp(self):
        """Initialize the instrument information."""
        super().setUp()

        # Update the attributes
        self.test_class = ocbpy.EABoundary
        self.inst_attrs = {"image": ["year", "soy", "num_sectors", "a",
                                     "r_err", "fom"],
                           "dmsp-ssj": ["date", "time", "sc", "x", "y", "fom",
                                        "x_1", "x_2", "y_1", "y_2"]}
        self.not_attrs = {"image": ["date", "time", "x", "y", "x_1", "x_2",
                                    "y_1", "y_2", "sc"],
                          "dmsp-ssj": ["year", "soy", "num_sectors", "a",
                                       "r_err"]}
        self.inst_init = [{"instrument": "image", "hemisphere": 1,
                           "filename": path.join(cc.test_dir,
                                                 "test_north_ocb")},
                          {"instrument": "dmsp-ssj", "hemisphere": 1,
                           "filename": path.join(cc.test_dir,
                                                 "dmsp-ssj_north_out.eab")},
                          {"instrument": "dmsp-ssj", "hemisphere": -1,
                           "filename": path.join(cc.test_dir,
                                                 "dmsp-ssj_south_out.eab")}]
        return


class TestEABoundaryMethodsGeneral(test_ocb.TestOCBoundaryMethodsGeneral):
    """Test the OCBoundary general methods."""

    def setUp(self):
        """Initialize the test environment."""
        super().setUp()

        self.test_class = ocbpy.EABoundary
        return


class TestEABoundaryMethodsNorth(test_ocb.TestOCBoundaryMethodsNorth):
    """Unit tests for the EABoundary class in the northern hemisphere."""

    def setUp(self):
        """Initialize the test environment."""
        super().setUp()

        self.test_class = ocbpy.EABoundary
        self.ref_boundary = 64.0

        self.ocb_lat = [numpy.nan, -37.95918548, -6.92874899, 20.28409774,
                        51.9732, 84.90702626]
        self.ocb_mlt = [numpy.nan, 4.75942194, 9.76745427, 14.61843964,
                        19.02060793, 17.832]
        return

    def test_normal_coord_north_geodetic(self):
        """Test the geodetic normalisation calculation in the north."""
        self.set_ocb()
        self.out = self.ocb.normal_coord(self.lat[-1], self.mlt[-1],
                                         coords='geodetic')
        self.assertAlmostEqual(self.out[0][0], 72.5526, places=3)
        self.assertAlmostEqual(self.out[1][0], 19.3839, places=3)
        self.assertEqual(self.out[2][0], self.r_corr)
        return

    def test_normal_coord_north_geocentric(self):
        """Test the geocentric normalisation calculation in the north."""
        self.set_ocb()
        self.out = self.ocb.normal_coord(self.lat[-1], self.mlt[-1],
                                         coords='geocentric')
        self.assertAlmostEqual(self.out[0][0], 72.5564, places=3)
        self.assertAlmostEqual(self.out[1][0], 19.3852, places=3)
        self.assertEqual(self.out[2][0], self.r_corr)
        return

    def test_revert_coord_north_geodetic(self):
        """Test the reversion to geodetic coordinates in the north."""
        self.set_ocb()
        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr, coords='geodetic')
        self.assertAlmostEqual(self.out[0][0], 77.13321838, places=3)
        self.assertAlmostEqual(self.out[1][0], 19.18124285, places=3)
        return

    def test_revert_coord_north_geocentric(self):
        """Test the reversion to geocentric coordinates in the north."""
        self.set_ocb()
        self.out = self.ocb.revert_coord(self.ocb_lat[-2], self.ocb_mlt[-2],
                                         self.r_corr, coords='geocentric')
        self.assertAlmostEqual(self.out[0][0], 77.05394766, places=3)
        self.assertAlmostEqual(self.out[1][0], 19.18124285, places=3)
        return


class TestEABoundaryMethodsSouth(test_ocb.TestOCBoundaryMethodsSouth):
    """Unit tests for the EABoundary methods in the southern hemisphere."""

    def setUp(self):
        """Initialize the test environment."""
        super().setUp()

        self.test_class = ocbpy.EABoundary
        self.ref_boundary = -64.0
        self.set_south = {"filename": path.join(cc.test_dir,
                                                "dmsp-ssj_south_out.eab"),
                          "instrument": "dmsp-ssj",
                          "hemisphere": -1,
                          "rfunc": ocbpy.ocb_correction.circular}
        self.ocb = self.test_class(**self.set_south)
        self.ocb.rec_ind = 0

        self.mlt = numpy.array([2.44067797, 2.84745763, 22.37288136,
                                22.77966102])
        self.lat = numpy.full(shape=self.mlt.shape, fill_value=-75.0)
        self.ocb_lat = [-84.51747815, -82.89664467, -76.7993294, -78.4535674]
        self.ocb_mlt = [6.02245268, 6.69193021, 18.56617597, 18.88837759]
        return

    def test_dmspssj_attrs(self):
        """Test that DMSP-SSJ attributes are available in the south."""

        for self.out in ["sc", "x_1", "x_2", "y_1", "y_2"]:
            self.assertTrue(hasattr(self.ocb, self.out),
                            msg="missing attr: {:s}".format(self.out))
        return

    def test_ampere_attrs(self):
        """Test that AMPERE attributes are not available in the DMSP data."""

        for self.out in ['date', 'time', 'x', 'y', 'fom']:
            self.assertTrue(hasattr(self.ocb, self.out),
                            msg="missing attr present: {:s}".format(
                                self.out))
        return

    def test_normal_coord_south_geocentric(self):
        """Test the geocentric normalisation calculation in the south."""
        self.out = self.ocb.normal_coord(self.lat[0], self.mlt[0], height=830,
                                         coords='geocentric')

        self.assertAlmostEqual(self.out[0][0], -76.3801206, places=3)
        self.assertAlmostEqual(self.out[1][0], 20.494021798, places=3)
        self.assertEqual(self.out[2][0], self.r_corr)
        return

    def test_normal_coord_south_geodetic(self):
        """Test the geodetic normalisation calculation in the south."""
        self.out = self.ocb.normal_coord(self.lat[0], self.mlt[0], height=830,
                                         coords='geodetic')

        self.assertAlmostEqual(self.out[0][0], -76.34575670, places=3)
        self.assertAlmostEqual(self.out[1][0], 20.524524159, places=3)
        self.assertEqual(self.out[2][0], self.r_corr)
        return

    def test_revert_coord_south_geodetic(self):
        """Test the reversion to geodetic coordinates in the south."""
        self.out = self.ocb.revert_coord(self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr, coords='geodetic')
        self.assertAlmostEqual(self.out[0][0], -66.317542, places=3)
        self.assertAlmostEqual(self.out[1][0], 5.90843131, places=3)
        return

    def test_revert_coord_south_geocentric(self):
        """Test the reversion to geocentric coordinates in the south."""
        self.out = self.ocb.revert_coord(self.ocb_lat[1], self.ocb_mlt[1],
                                         self.r_corr, coords='geocentric')
        self.assertAlmostEqual(self.out[0][0], -66.1832772, places=3)
        self.assertAlmostEqual(self.out[1][0], 5.90843131, places=3)
        return

    def test_normal_coord_south_corrected(self):
        """Test normalisation calculation in the south with a corrected OCB."""
        self.r_corr = 1.0
        self.ocb.rfunc_kwargs[self.ocb.rec_ind]['r_add'] = self.r_corr
        self.out = self.ocb.normal_coord(self.lat[0], self.mlt[0])

        self.assertAlmostEqual(self.out[0], -84.764005494, places=3)
        self.assertAlmostEqual(self.out[1], 6.02245269240, places=3)
        self.assertEqual(self.out[2], self.r_corr)
        return

    def test_aacgm_boundary_location_good_south(self):
        """Test finding the EAB in AACGM coordinates in the south."""

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=self.ocb.rec_ind)

        # Test value of latitude attribute
        self.assertTrue(
            numpy.all(self.ocb.aacgm_boundary_lat[self.ocb.rec_ind] < 0.0))
        self.assertAlmostEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].min(),
            -61.64470517204886)
        self.assertEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].argmin(), 2)
        self.assertAlmostEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].max(),
            -56.71986304055978)
        self.assertEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].argmax(), 0)
        return

    def test_aacgm_boundary_location_good_south_corrected_func_arr(self):
        """Test func array init with good, southern, corrected EAB."""
        self.set_south['rfunc'] = numpy.full(
            shape=self.ocb.r.shape, fill_value=ocbpy.ocb_correction.circular)
        self.set_south['rfunc_kwargs'] = numpy.full(shape=self.ocb.r.shape,
                                                    fill_value={"r_add": 1.0})
        self.ocb = self.test_class(**self.set_south)

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=self.ocb.rec_ind)

        # Test value of latitude attribute
        self.assertTrue(
            numpy.all(self.ocb.aacgm_boundary_lat[self.ocb.rec_ind] < 0.0))
        self.assertAlmostEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].min(),
            -59.90673085116123)
        self.assertEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].argmin(), 1)
        self.assertAlmostEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].max(),
            -56.588628664055825)
        self.assertEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].argmax(), 3)
        return

    def test_aacgm_boundary_location_good_south_corrected_kwarg_arr(self):
        """Test kwarg array init with good, southern, corrected OCB."""
        self.set_south['rfunc_kwargs'] = numpy.full(shape=self.ocb.r.shape,
                                                    fill_value={"r_add": 1.0})
        self.ocb = self.test_class(**self.set_south)

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=self.ocb.rec_ind)

        # Test value of latitude attribute
        self.assertTrue(
            numpy.all(self.ocb.aacgm_boundary_lat[self.ocb.rec_ind] < 0.0))
        self.assertAlmostEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].min(),
            -59.90673085116123)
        self.assertEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].argmin(), 1)
        self.assertAlmostEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].max(),
            -56.588628664055825)
        self.assertEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].argmax(), 3)
        return

    def test_aacgm_boundary_location_good_south_corrected_dict(self):
        """Test dict init with good, southern, corrected OCB."""
        self.set_south['rfunc_kwargs'] = {"r_add": 1.0}
        self.ocb = self.test_class(**self.set_south)

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=self.ocb.rec_ind)

        # Test value of latitude attribute
        self.assertTrue(
            numpy.all(self.ocb.aacgm_boundary_lat[self.ocb.rec_ind] < 0.0))
        self.assertAlmostEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].min(),
            -59.90673085116123)
        self.assertEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].argmin(), 1)
        self.assertAlmostEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].max(),
            -56.588628664055825)
        self.assertEqual(
            self.ocb.aacgm_boundary_lat[self.ocb.rec_ind].argmax(), 3)
        return

    def test_aacgm_boundary_location_good_south_corrected(self):
        """Test finding the corrected OCB in AACGM coordinates in the south."""
        self.out = self.ocb.rec_ind
        self.ocb.rfunc_kwargs[self.ocb.rec_ind]['r_add'] = 1.0

        # Add the attribute at the good location
        self.ocb.get_aacgm_boundary_lat(self.mlt, rec_ind=self.out)

        # Test value of latitude attribute
        self.assertTrue(numpy.all(self.ocb.aacgm_boundary_lat[self.out] < 0.0))
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[self.out].min(),
                               -60.54469198966263)
        self.assertEqual(self.ocb.aacgm_boundary_lat[self.out].argmin(), 2)
        self.assertAlmostEqual(self.ocb.aacgm_boundary_lat[self.out].max(),
                               -55.70577000484318)
        self.assertEqual(self.ocb.aacgm_boundary_lat[self.out].argmax(), 0)
        return


class TestEABoundaryFailure(test_ocb.TestOCBoundaryFailure):
    """Test the EABoundary class failures raise appropriate errors."""

    def setUp(self):
        """Set up the test environment."""
        self.test_class = ocbpy.EABoundary
        return
