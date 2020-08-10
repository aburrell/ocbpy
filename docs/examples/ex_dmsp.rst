Loading DMSP SSJ boundary files
-------------------------------
Unlike the IMAGE and AMPERE boundaries, the DMSP SSJ boundaries are not included
with the package.  However, routines to obtain them are.  To use them, you need
to install the
`ssj_auroral_boundary <https://github.com/lkilcommons/ssj_auroral_boundary>`__
package.  Once installed, you can download DMSP SSJ data and obtain a boundary
file for a specified time period.  For this example, we'll use a single day.
You can download the files into any directory, but this example will put them
in the same directory as the other OCB files.

::
   
   import datetime as dt
   import ocbpy
   import os

   stime = dt.datetime(2010, 12, 31)
   etime = stime + dt.datetime(days=1)
   out_dir = os.path.join(os.path.split(ocbpy.__file__)[0], "boundaries")

   bfiles = ocbpy.boundaries.dmsp_ssj_files.fetch_format_ssj_boundary_files(stime, etime, out_dir=out_dir, rm_temp=False)


By setting `rm_temp=False`, all of the different DMSP files will be kept in the
specified output directory.  You should have three CDF files (the data
downloaded from each DMSP spacecraft), the CSV files (the boundaries calculated
for each DMSP spacecraft) and four boundary files.  The boundary files have
an extention of `.eab` for the Equatorial Auroral Boundary and `.ocb` for the
Open-Closed field line Boundary.  The files are separated by hemisphere, and
also specify the date range.  Because only one day was obtained, the start and
end dates in the filename are identical.  When `rm_temp=True`, the CDF and CSV
files are removed.

You can now load the DMSP SSJ boundaries by specifying the desired filename,
instrument, and hemisphere or merely the instrument and hemisphere.


::
   
   # Load with filename, instrument, and hemisphere
   south_file = os.path.join(out_dir, "dmsp-ssj_south_20101231_20101231_v1.1.2.ocb")
   ocb_south = ocbpy.ocboundary.OCBoundary(filename=south_file, instrument='dmsp-ssj', hemisphere=-1)
   print(ocb_south)

   Open-Closed Boundary file: ~/ocbpy/ocbpy/boundaries/dmsp-ssj_south_20101231_20101231_v1.1.2.ocb
   Source instrument: DMSP-SSJ
   Open-Closed Boundary reference latitude: -74.0 degrees

   21 records from 2010-12-31 00:27:23 to 2010-12-31 22:11:38

   YYYY-MM-DD HH:MM:SS Phi_Centre R_Centre R
   -----------------------------------------------------------------------------
   2010-12-31 00:27:23 356.72 14.02 1.70
   2010-12-31 12:27:56 324.82 0.86 0.65
   2010-12-31 18:49:58 233.68 6.12 2.48
   2010-12-31 22:11:38 318.60 4.64 4.26

   Uses scaling function(s):
   circular(**{})

   # Load with date, instrument, and hemisphere
   ocb_north = ocbpy.ocboundary.OCBoundary(stime=stime, instrument='dmsp-ssj', hemisphere=1)
   print(ocb_north)

   Open-Closed Boundary file: ~/ocbpy/ocbpy/boundaries/dmsp-ssj_north_20101231_20101231_v1.1.2.ocb
   Source instrument: DMSP-SSJ
   Open-Closed Boundary reference latitude: 74.0 degrees

   27 records from 2010-12-31 01:19:13 to 2010-12-31 23:02:48

   YYYY-MM-DD HH:MM:SS Phi_Centre R_Centre R
   -----------------------------------------------------------------------------
   2010-12-31 01:19:13 191.07 10.69 0.54
   2010-12-31 06:27:18 195.29 13.52 0.35
   2010-12-31 21:21:32 259.27 2.73 2.03
   2010-12-31 23:02:48 234.73 3.94 1.38

   Uses scaling function(s):
   circular(**{})

The circular scaling function with no input adds zero the the boundaries, and
so performs no scaling.  At this point in time, the EAO boundaries are not
used, but future versions of this package will grid data relative to both the
OCB and EAO boundary.
