This directory contains files with Open Closed field line Boundaries obtained
from different instruments

File Format
----------------
YR, SOY, NB, PHICENT, RCENT, R, A, R_ERR
 
YR      : Year
SOY     : Seconds of year (be careful in leap years)
NB      : Number of MLT sectors that a boundary existed for that fit
PHICENT : Angle from midnight of the line from the AACGM pole to the
          ‘fitted pole’  (effectively MLT*15)
RCENT   : AACGM Co-latitude of the location of the ‘fitted pole’
R       : radius of the circle (co-latitude) remembering that the circle centre
          is the ‘fitted pole’
A       : Area of the circle
R_ERR   : Error in the radius
 
There are certain ranges for NB, RCENT, and R that you shouldn’t use that can
be found (and explained) in Chisham (2017), doi:10.1002/2016JA023235.  These
ranges are the defaults in OCBoundary.get_next_good_ocb_ind.

Files
---------
si12_north_circle : Spectrographic Imager SI12
       		    [Mende, et al. (2000), Space Sci. Rev., 91, 287–318]
si13_north_circle : Spectrographic Imager SI13
                    [Mende, et al. (2000), Space Sci. Rev., 91, 287–318]
wic_north_circle  : Wideband Imaging Camera (WIC)
                    [Mende, et al. (2000b), Space Sci. Rev., 91, 271–285]
