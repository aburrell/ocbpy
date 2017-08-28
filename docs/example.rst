Example
============

Here is a simple example that will show how to initialise an OCBoundary object,
find a trustworthy open-closed boundary, and convert from AACGM coordinates to
OCB coordinates.

Initialise an OCBoundary object
--------------------------------
Start a python or iPython session, and begin by importing ocbpy, numpy,
matplotlib, and datetime.
::
   import numpy as np
   import datetime as dt
   import matplotlib.pyplot as plt
   import ocbpy
  
Next, initialise an OCB class object.  This uses the default IMAGE FUV file and
will take a few minutes to load.
::
   ocb = ocbpy.ocboundary.OCBoundary()
   print ocb
  
   Open-Closed Boundary file: ~/ocbpy/ocbpy/boundaries/si13_north_circle
   Source instrument: IMAGE
   Open-Closed Boundary reference latitude: 74.0 degrees
  
   219927 records from 2000-05-05 11:35:27 to 2002-08-22 00:01:28
  
   YYYY-MM-DD HH:MM:SS Phi_Centre R_Centre R
   -----------------------------------------------------------------------------
   2000-05-05 11:35:27 356.93 8.74 9.69
   2000-05-05 11:37:23 202.97 13.23 22.23
   2002-08-21 23:55:20 322.60 5.49 15.36
   2002-08-22 00:01:28 179.02 2.32 19.52

Retrieve a good OCB record
--------------------------
Get the first good OCB record, which will be record index 27.
::
   ocb.get_next_good_ocb_ind()
   print ocb.rec_ind

To get the OCB record closest to a specified time, use **ocbpy.match_data_ocb**
::
   first_good_time = ocb.dtime[ocb.rec_ind]
   test_times = [first_good_time + dt.timedelta(minutes=5*(i+1)) for i in range(5)]
   itest = ocbpy.match_data_ocb(ocb, test_times, idat=0)
   print itest, ocb.rec_ind, test_times[itest], ocb.dtime[ocb.rec_ind]
  
   0 31 2000-05-05 13:45:30 2000-05-05 13:50:29

Convert between AACGM and OCB coordinates
------------------------------------------
We'll start by visualising the location of the OCB using the first good OCB
in the default IMAGE FUV file.
::
   f = plt.figure()
   ax = f.add_subplot(111, projection="polar")
   ax.set_theta_zero_location("S")
   ax.xaxis.set_ticks([0, 0.5*np.pi, np.pi, 1.5*np.pi])
   ax.xaxis.set_ticklabels(["00:00", "06:00", "12:00 MLT", "18:00"])
   ax.set_rlim(0,25)
   ax.set_rticks([5,10,15,20])
   ax.yaxis.set_ticklabels(["85$^\circ$", "80$^\circ$", "75$^\circ$", "70$^\circ$"]

Mark the location of the reference OCB, set at 74 degrees AACGM latitude
::
   lon = np.arange(0.0, 2.0 * np.pi + 0.1, 0.1)
   ref_lat = np.ones(shape=lon.shape) * 16.0
   ax.plot(lon, ref_lat, "--", linewidth=2, color="0.6", label="Reference OCB")

Mark the location of the circle centre in AACGM coordinates
::
   ocb.rec_ind = 27
   phi_cent_rad = np.radians(ocb.phi_cent[ocb.rec_ind])
   ax.plot([phi_cent_rad], [ocb.r_cent[ocb.rec_ind]], "mx", ms=10, label="OCB Pole")

Calculate at plot the location of the OCB in AACGM coordinates
::
   del_lon = lon - phi_cent_rad
   ax.plot(lon, lat, "m-", linewidth=2, label="OCB")

Now add the location of a point in AACGM coordinates
::
   aacgm_lat = 85.0
   aacgm_lon = np.pi

   ax.plot([aacgm_lon], [90.0-aacgm_lat], "ko", ms=5, label="AACGM Point")

Find the location relative to the current OCB.  Note that the AACGM coordinates
must be in degrees latitude and hours of magnetic local time (MLT).
::
   ocb_lat, ocb_mlt = ocb.normal_coord(aacgm_lat, aacgm_lon * 12.0 / np.pi)
   ax.plot([ocb_mlt * np.pi / 12.0], [90.0 - ocb_lat], "mo", label="OCB Point")

Add a legend to finish the figure.
::
   ax.legend(loc=2, fontsize="medium", bbox_to_anchor=(-0.4,1.15), title="{:}".format(ocb.dtime[ocb.rec_ind]))

.. image:: example_ocb_location.png

Scaling of values dependent on the electric potential can be found in the
**ocbpy.ocb_scaling** `module <ocb_gridding.html#ocb-scaling>`__.


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
