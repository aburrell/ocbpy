#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
""" Tests the ocb_scaling class and functions
"""
import ocbpy
import unittest
import numpy as np

class TestOCBScalingMethods(unittest.TestCase):

    def setUp(self):
        """ Initialize the OCBoundary object using the test file, as well as
        the VectorData object
        """
        from os.path import isfile

        ocb_dir = ocbpy.__file__.split("/")
        test_file = "{:s}/{:s}".format("/".join(ocb_dir[:-1]),
                                       "tests/test_data/test_north_circle")
        self.assertTrue(isfile(test_file))
        self.ocb = ocbpy.ocboundary.OCBoundary(filename=test_file)
        self.ocb.rec_ind = 27
        self.vdata = ocbpy.ocb_scaling.VectorData(0, self.ocb.rec_ind, 75.0,
                                                  22.0, aacgm_n=50.0,
                                                  aacgm_e=86.5, aacgm_z=5.0,
                                                  dat_name="Test",
                                                  dat_units="$m s^{-1}$")

    def tearDown(self):
        del self.ocb, self.vdata

    def test_init(self):
        """ Test the initialisation of the VectorData object
        """
        self.assertAlmostEqual(self.vdata.aacgm_mag, 100.036243432)

    def test_calc_vec_pole_angle(self):
        """ Test to see that the polar angle calculation is performed properly
        """
        self.vdata.ocb_aacgm_mlt = self.ocb.phi_cent[self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.r_cent[self.vdata.ocb_ind]
        (self.vdata.ocb_lat,
         self.vdata.ocb_mlt) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                     self.vdata.aacgm_mlt)

        # Test the calculation of the test pole angle
        self.vdata.calc_vec_pole_angle()
        self.assertAlmostEqual(self.vdata.pole_angle, 7.52899382202)
        
        # If the measurement has the same MLT as the OCB pole, the angle is
        # zero.  This includes the AACGM pole and the OCB pole.
        self.vdata.aacgm_mlt = self.vdata.ocb_aacgm_mlt
        self.vdata.calc_vec_pole_angle()
        self.assertEqual(self.vdata.pole_angle, 0.0)

        # If the measurement is on the opposite side of the AACGM pole as the
        # OCB pole, then the angle will be 180 degrees
        self.vdata.aacgm_mlt = self.vdata.ocb_aacgm_mlt + 12.0
        self.vdata.calc_vec_pole_angle()
        self.assertEqual(self.vdata.pole_angle, 180.0)

        # Set the distance between the data point and the OCB is equal to the
        # distance between the AACGM pole and the OCB so that the triangles
        # we're examining are isosceles triangles
        
        # When the MLT is 00:00 the calculated angle is equal to the OCB MLT
        self.vdata.aacgm_mlt = 0.0
        self.vdata.ocb_lat = self.vdata.ocb_aacgm_lat
        self.vdata.calc_vec_pole_angle()
        self.assertEqual(self.vdata.pole_angle, self.vdata.ocb_aacgm_mlt * 15.0)

        # When the MLT is 06:00 the calculated angle equals 90 - the OCB MLT
        self.vdata.aacgm_mlt = 6.0
        self.vdata.calc_vec_pole_angle()
        self.assertEqual(self.vdata.pole_angle + self.vdata.ocb_aacgm_mlt
                         * 15.0, 90.0)

        # Return to the default AACGM MLT value
        self.vdata.aacgm_mlt = 22.0

    def test_define_quadrants(self):
        """ Test the assignment of quadrants
        """
        # Set the initial values
        self.vdata.ocb_aacgm_mlt = self.ocb.phi_cent[self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.r_cent[self.vdata.ocb_ind]
        (self.vdata.ocb_lat,
         self.vdata.ocb_mlt) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                     self.vdata.aacgm_mlt)
        self.vdata.calc_vec_pole_angle()
        
        # Get the test quadrants
        self.vdata.define_quadrants()
        self.assertEqual(self.vdata.ocb_quad, 1)
        self.assertEqual(self.vdata.vec_quad, 1)

    def test_calc_ocb_vec_sign(self):
        """ Test the calculation of the OCB vector signs
        """

        # Set the initial values
        self.vdata.ocb_aacgm_mlt = self.ocb.phi_cent[self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.r_cent[self.vdata.ocb_ind]
        (self.vdata.ocb_lat,
         self.vdata.ocb_mlt) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                     self.vdata.aacgm_mlt)
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()

        vmag = np.sqrt(self.vdata.aacgm_n**2 + self.vdata.aacgm_e**2)
        self.vdata.aacgm_naz = np.degrees(np.arccos(self.vdata.aacgm_n / vmag))

        # Calculate the vector data signs
        vsigns = self.vdata.calc_ocb_vec_sign(north=True, east=True)
        self.assertTrue(vsigns['north'])
        self.assertTrue(vsigns['east'])
        
    def test_scale_vec(self):
        """ Test the calculation of the OCB vector signs
        """

        # Set the initial values
        self.vdata.ocb_aacgm_mlt = self.ocb.phi_cent[self.vdata.ocb_ind] / 15.0
        self.vdata.ocb_aacgm_lat = 90.0 - self.ocb.r_cent[self.vdata.ocb_ind]
        (self.vdata.ocb_lat,
         self.vdata.ocb_mlt) = self.ocb.normal_coord(self.vdata.aacgm_lat,
                                                     self.vdata.aacgm_mlt)
        self.vdata.calc_vec_pole_angle()
        self.vdata.define_quadrants()

        vmag = np.sqrt(self.vdata.aacgm_n**2 + self.vdata.aacgm_e**2)
        self.vdata.aacgm_naz = np.degrees(np.arccos(self.vdata.aacgm_n / vmag))

        # Scale the data vector
        self.vdata.scale_vector()

        # Test the North and East components
        self.assertAlmostEqual(self.vdata.ocb_n, 60.9028461225)
        self.assertAlmostEqual(self.vdata.ocb_e, 79.2028619065)
        
        # Test to see that the magnitudes and z-components are the same
        self.assertEqual(self.vdata.aacgm_mag, self.vdata.ocb_mag)
        self.assertEqual(self.vdata.ocb_z, self.vdata.aacgm_z)

    def test_set_ocb(self):
        """ Test setting of OCB values for the VectorData object
        """

        # Set the OCB values without any E-field scaling, test to see that the
        # AACGM and OCB vector magnitudes are the same
        self.vdata.set_ocb(self.ocb)
        self.assertEqual(self.vdata.aacgm_mag, self.vdata.ocb_mag)

        # Set the OCB values with scaling for a variable proportional to
        # the electric field
        self.vdata.set_ocb(self.ocb, scale_func=ocbpy.ocb_scaling.normal_evar)
        self.assertAlmostEqual(self.vdata.ocb_mag, 80.3516453163)

        # Set the OCB values with scaling for a variable proportional to
        # the curl of the electric field
        self.vdata.set_ocb(self.ocb,
                           scale_func=ocbpy.ocb_scaling.normal_curl_evar)
        self.assertAlmostEqual(self.vdata.ocb_mag, 80.3516453163)
        
if __name__ == '__main__':
    unittest.main()

