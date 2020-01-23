Load a test AMPERE OCB file
------------------------------------------
A mock AMPERE file is available in the test directory, containing data for the
southern hemisphere.  Load this data using the following commands.
::
   ocb = ocbpy.ocboundary.OCBoundary(filename="~/ocbpy/ocbpy/tests/test_data/test_south_circle", instrument="ampere", hemisphere=-1)
   print ocb

   Open-Closed Boundary file: tests/test_data/test_south_circle
   Source instrument: AMPERE
   Open-Closed Boundary reference latitude: -72.0 degrees

   14 records from 2010-01-01 00:00:00 to 2010-01-01 00:26:00

   YYYY-MM-DD HH:MM:SS Phi_Centre R_Centre R
   -----------------------------------------------------------------------------
   2010-01-01 00:00:00 296.57 2.24 10.00
   2010-01-01 00:02:00 315.00 2.83 12.00
   2010-01-01 00:24:00 270.00 2.00 10.00
   2010-01-01 00:26:00 270.00 2.00 10.00

Note that the OCB reference latitude is now -72 instead of +74 degrees.  The
sign is specified by the hemisphere keyword and the magnitude of the reference
latitude was set based on the differences in the boundaries measured by
AMPERE and IMAGE FUV.

If you compare the test files for IMAGE FUV and AMPERE, there are more
differences.  The AMPERE data has stored the OCB size and location in Cartesian
coordinates (where the origin lies at the AACGM pole, the x-axis lies along the
dusk-dawn meridian, and the y-axis lies along the midnight-noon meridian), while
the IMAGE data has stored this information in polar coordinates.  The
differences in the two data sets also means that the conditions for evaluating
good OCBs differ.  AMPERE data uses the relative difference in magnitude of the
upward/downward current systems, rather than the number of MLT sectors with
useable information (as IMAGE FUV does).

Any other data file that contains the OCB data in one of the two coordinate
sets can be loaded without any alteration by setting the *instrument* keyword
appropriately.  However, if good boundaries require alternate quantities to be
evaluated (look at the **ocbpy.ocbounary.OCBoundary.get_next_good_ocb_ind**
`routine <ocb_gridding.html#module-ocbpy.ocboundary>`__ for more information),
then modifications will need to be made, or inappropriate boundaries removed
from the input file.
