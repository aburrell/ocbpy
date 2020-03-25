
Load a general data file (DMSP)
---------------------------------------------
DMSP SSIES provides commonly used polar data, which can be accessed from the
University of Texas at Dallas `Center for Space Science <http://cindispace.utdallas.edu/DMSP/dmsp_data_at_utdallas.html>`_.  To run this example, follow the
previous link and download the ASCII file for F15 on 23 June 2000 14:08 UT.
This will provide you with a file named **f15_rl001751408.txt**.  To load this
file, use the following commands.
::
   hh = ["YYDDD    SOD  R I   Alt    GLAT   GLONG    MLAT     MLT     Vx     Vy      Vz     RMSx  SigmaY  SigmaZ    Ni_(cm^-3)    Frac_O  Frac_He   Frac_H   Ti     Te      pts"]
   dmsp_filename = "f15_rl001751408.txt"
   dmsp_head, dmsp_data = ocbpy.instruments.general.load_ascii_data(dmsp_filename, 3, datetime_cols=[0,1], datetime_fmt="YYDDD SOD", header=hh, int_cols=[2, 3, 21])

   print dmsp_data['Ti'].shape, dmsp_data.keys()
   
   (1517,) ['GLONG', 'Ti', 'datetime', 'MLAT', 'SigmaY', 'SigmaZ', 'RMSx', 'Te', 'pts', 'SOD', 'Ni_(cm^-3)', 'Frac_H', 'Frac_O', 'Frac_He', 'I', 'GLAT', 'R', 'MLT', 'Vz', 'YYDDD', 'Vx', 'Vy', 'Alt']

In the call to ocbpy.instruments.general.load_ascii_data, quality flags and
number of points are saved as integers by specifying int_cols.  The header
needs to be specified using **header** because the header in the data file,
even though it specifies the data columns in the last line, does not use white
space to only seperate different data column names.

Before calculating the OCB coordinates, add space in the data dictionary for the
OCB coordinates and find out which data have a good quality flag.
::

    dens_key = 'Ni_(cm^-3)'
    dmsp_data['OCB_MLT'] = np.zeros(shape=dmsp_data['Vx'].shape, dtype=float) * np.nan
    dmsp_data['OCB_LAT'] = np.zeros(shape=dmsp_data['Vx'].shape, dtype=float) * np.nan
    igood = [i for i,r in enumerate(dmsp_data['R']) if r < 3 and dmsp_data['I'][i] < 3]
    print len(igood), dmsp_data[dens_key][igood].max(), dmsp_data[dens_key][igood].min()

    702 153742.02 4692.9639

   
Now get the OCB coordinates for each location.  This will not be possible
everywhere, since IMAGE doesn't provide Southern Hemisphere data and only times
with a good OCB established within the last 5 minutes will be used.
::
   idmsp = 0
   ndmsp = len(igood)
   ocb = ocbpy.ocboundary.OCBoundary()
   ocb.get_next_good_ocb_ind()

   while idmsp < ndmsp and ocb.rec_ind < ocb.records:
       idmsp = ocbpy.match_data_ocb(ocb, dmsp_data['datetime'][igood], idat=idmsp, max_tol=600)
       if idmsp < ndmsp and ocb.rec_ind < ocb.records:
           print idmsp, igood[idmsp], ocb.rec_ind
           nlat, nmlt = ocb.normal_coord(dmsp_data['MLAT'][igood[idmsp]], dmsp_data['MLT'][igood[idmsp]])
           dmsp_data['OCB_LAT'][igood[idmsp]] = nlat
           dmsp_data['OCB_MLT'][igood[idmsp]] = nmlt
           idmsp += 1

    igood = [i for i,m in enumerate(dmsp_data['OCB_LAT']) if not np.isnan(m)]
    print len(igood), dmsp_data['OCB_LAT'][igood].max()

    334 78.8453722883

Now, let's plot the satellite track over the pole, relative to the OCB, with
the location accouting for changes in the OCB at a 5 minute resolution.  Note
how the resolution results in apparent jumps in the satellite location.  We
aren't going to plot the ion velocity here, because it is provided in spacecraft
coordinates rather than magnetic coordinates, adding an additional
(and not intensive) level of processing.
::
   f = plt.figure()
   f.suptitle("DMSP F15 in OCB Coordinates")
   ax = f.add_subplot(111, projection="polar")
   ax.set_theta_zero_location("S")
   ax.xaxis.set_ticks([0, 0.5*np.pi, np.pi, 1.5*np.pi])
   ax.xaxis.set_ticklabels(["00:00", "06:00", "12:00 MLT", "18:00"])
   ax.set_rlim(0,40)
   ax.set_rticks([10,20,30,40])
   ax.yaxis.set_ticklabels(["80$^\circ$", "70$^\circ$", "60$^\circ$", "50$^\circ$"])

   lon = np.arange(0.0, 2.0 * np.pi + 0.1, 0.1)
   lat = np.ones(shape=lon.shape) * (90.0 - ocb.boundary_lat)
   ax.plot(lon, lat, "m-", linewidth=2, label="OCB")

   dmsp_lon = dmsp_data['OCB_MLT'][igood] * np.pi / 12.0
   dmsp_lat = 90.0 - dmsp_data['OCB_LAT'][igood]
   dmsp_time = mpl.dates.date2num(dmsp_data['datetime'][igood])
   ax.scatter(dmsp_lon, dmsp_lat, c=dmsp_time, cmap=mpl.cm.get_cmap("Blues"), marker="o", s=10)
   ax.text(10 * np.pi / 12.0, 41, "Start of satellite track")
