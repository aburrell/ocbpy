This directory contains files with Open Closed field line Boundaries obtained
from different instruments

IMAGE (image/si12/si13/wic) File Format
---------------------------------------
YR, SOY, NB, PHICENT, RCENT, R, A, R_ERR, R_MERIT
 
YR      : Year
SOY     : Seconds of year (be careful in leap years)
NB      : Number of MLT sectors that a boundary existed for that fit
PHICENT : Angle from midnight of the line from the AACGM pole to the
          ‘fitted pole’  (effectively MLT*15)
RCENT   : AACGM Co-latitude of the location of the ‘fitted pole’
R       : radius of the circle (co-latitude) remembering that the circle centre
          is the ‘fitted pole’ in degrees
A       : Psuedo area of the circle (as in AACGM, not geographic)
R_ERR   : Root-mean squared error in the radius in degrees
R_MERIT : Radial distance from the most typical pole location in degrees
 
Previosly, there are certain ranges for NB, RCENT, and R that were not
recommended for use (see Chisham (2017), doi:10.1002/2016JA023235).  Currently,
R_MERIT is used instead, with recommended merit values documented in Chisham
et al. (2022), doi:10.1029/2022JA030622.  These ranges are the defaults in
OCBoundary.get_next_good_ocb_ind.  When using these boundaries, remember to cite
Chisham (2017) or Chisham et al. (2022). From ocbpy version 0.3.0 onward,
the SI12, SI13, and WIC files contain uncorrected auroral boundary fits, while
the IMAGE file contains DMSP corrected average boundaries for the OCB and EAB.

AMPERE (amp) File Format
------------------------
DATE, TIME, RADIUS, X, Y, J_MAG

DATE   : 4-digit year, 2-digit month, and 2-digit day of month (YYYYMMDD)
TIME   : 2-digit hours of day and 2-digit minutes of hour, separated by a colon
RADIUS : Radius of the R1/R2 boundary circle in degrees
X      : Location of the R1/R2 boundary circle centre in degrees along the
         dawn/dusk axis
Y      : Location of the R1/R2 boundary circle centre in degrees along the
         noon/midnight axis
J_MAG  : Largest positive-to-negative changed in the summed currents (micro amps
         per meter)

There are certain ranges for J_MAG that shouldn't be trusted.  This limitation
is explained in Milan et al. (2015), doi:10.1002/2015JA021680.  This range is
the default in OCBoundary.get_next_good_ocb_ind.  When using these boundaries,
remember to cite Milan et al. (2015).  If the OCB correction was applied, also
remember to cite Burrell at al. (2019).

Files
-----
amp_north_radii.txt    : Active Magnetosphere and Planetary Electrodynamics
                         Response Experiment
		         [Waters, et al. (2001), Geophys. Res. Lett., 28,
		          2165-2168]
amp_south_radii.txt    : Active Magnetosphere and Planetary Electrodynamics
                         Response Experiment
		         [Waters, et al. (2001), Geophys. Res. Lett., 28,
		          2165-2168]
si12_north_circle.xxx  : Spectrographic Imager SI12
       		         [Mende, et al. (2000), Space Sci. Rev., 91, 287–318]
si13_north_circle.xxx  : Spectrographic Imager SI13
                         [Mende, et al. (2000), Space Sci. Rev., 91, 287–318]
wic_north_circle.xxx   : Wideband Imaging Camera (WIC)
                         [Mende, et al. (2000b), Space Sci. Rev., 91, 271–285]
image_north_circle.xxx : Combined data from all IMAGE instruments
