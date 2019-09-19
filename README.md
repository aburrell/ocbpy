[![Build Status](https://www.travis-ci.org/aburrell/ocbpy.svg?branch=master)](https://www.travis-ci.org/aburrell/ocbpy)	[![Coverage Status](https://coveralls.io/repos/github/aburrell/ocbpy/badge.svg?branch=master)](https://coveralls.io/github/aburrell/ocbpy?branch=master)	[![Documentation Status](https://readthedocs.org/projects/ocbpy/badge/?version=latest)](http://ocbpy.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/96153180.svg)](https://zenodo.org/badge/latestdoi/96153180)

# Overview

ocbpy is a Python module that converts between AACGM coordinates and a magnetic
coordinate system that adjusts latitude and local time relative to the Open
Closed field line Boundary (OCB).  This is particulary useful for statistical
studies of the poles, where gridding relative to a fixed magnetic coordinate
system would cause averaging of different physical regions, such as auroral
and polar cap measurements.  This coordinate system is described in:

  * Chisham, G. (2017), A new methodology for the development of high‐latitude
    ionospheric climatologies and empirical models, Journal of Geophysical
    Research: Space Physics,
    [doi:10.1002/2016JA023235.](https://doi.org/10.1002/2016JA023235)

  * Full [documentation](http://ocbpy.rtfd.io/)

OCBs must be obtained from observations for this coordinate transformation.
In the British Antarctic Survey's [IMAGE Auroral Boundary data project](https://www.bas.ac.uk/project/image-auroral-boundary-data/)
from three auroral instruments provide northern hemisphere OCB locations
for 3 May 2000 03:01:42 UT - 22 Aug 2002 00:01:28, though not all of the times
included in these files contain high-quality estimations of the OCB.
Recommended selection criteria are included as defaults in the OCBoundary class.
OCBpy also supports boundaries provided by AMPERE, as discussed in:

  * Burrell, A. G. et al.: AMPERE Polar Cap Boundaries, Ann. Geophys. Discuss.,
    [doi:10.5194/angeo-2019-113](https://doi.org/10.5194/angeo-2019-113),
    in review, 2019.

Currently, support is included for files from the following datasets:

  * SuperMAG (available at http://supermag.jhuapl.edu)
  * SuperDARN Vorticity (contact GC at gchi@bas.ac.uk)
  * Any pysat Instrument (available at https://github.com/rstoneback/pysat)

These routines may be used as a guide to write routines for other datasets.

# Python versions

This module has been tested on python version 2.7, 3.5 - 3.7.  Support for 2.7
will be dropped in 2020.

# Dependencies

The listed dependecies were tested with the following versions:
  * numpy
  * logbook
  * aacgmv2
  * pysat (2.0.0)

Testing is performed using the python module, unittest.  To limit dependency
issues, pysat (>=2.0.0) is an optional dependency.

# Installation

Installation is now available through pypi [![PyPI version](https://badge.fury.io/py/ocbpy.svg)](https://badge.fury.io/py/ocbpy)

```
    $ pip install ocbpy
```

You may also checkout the repository and install it yourself:

```
    $ git clone git://github.com/aburrell/ocbpy.git;
```

Change directories into the repository folder and run the setup.py file.  For
a local install use the "--user" flag after "install".

```
    $ cd ocbpy/
    $ python setup.py install
```

To run the unit tests,

```
    $ python setup.py test
```

# Example

In iPython, run:

```
import numpy as np
import ocbpy
```

Then initialise an OCB class object.  This uses the default IMAGE FUV file and
will take a few minutes to load.

```
ocb = ocbpy.ocboundary.OCBoundary()
print ocb
```

The output should be as follows:

```
Open-Closed Boundary file: ~/ocbpy/ocbpy/boundaries/si13_north_circle
Source instrument: IMAGE
Open-Closed Boundary reference latitude: 74.0 degrees

219927 records from 2000-05-05 11:35:27 to 2002-08-22 00:01:28

YYYY-MM-DD HH:MM:SS NumSectors Phi_Centre R_Centre R  R_Err Area
-----------------------------------------------------------------------------
2000-05-05 11:35:27 4 356.93 8.74 9.69 0.14 3.642e+06
2000-05-05 11:37:23 5 202.97 13.23 22.23 0.77 1.896e+07
2002-08-21 23:55:20 8 322.60 5.49 15.36 0.61 9.107e+06
2002-08-22 00:01:28 7 179.02 2.32 19.52 0.89 1.466e+07
```

Get the first good OCB record, which will be record index 27.

```
ocb.get_next_good_ocb_ind()
print ocb.rec_ind

27
```

Now plot the location of the OCB

First initialise the figure
```
import matplotlib.pyplot as plt
f = plt.figure()
ax = f.add_subplot(111, projection="polar")
ax.set_theta_zero_location("S")
ax.xaxis.set_ticks([0, 0.5*np.pi, np.pi, 1.5*np.pi])
ax.xaxis.set_ticklabels(["00:00", "06:00", "12:00 MLT", "18:00"])
ax.set_rlim(0,25)
ax.set_rticks([5,10,15,20])
ax.yaxis.set_ticklabels(["85$^\circ$","80$^\circ$","75$^\circ$","70$^\circ$"])
```

Mark the location of the circle centre in AACGM coordinates
```
ax.plot(np.radians(ocb.phi_cent[ocb.rec_ind]), ocb.r_cent[ocb.rec_ind], "mx", ms=10, label="OCB Pole")
```

Calculate at plot the location of the OCB in AACGM coordinates
```
lon = np.linspace(0.0, 2.0 * np.pi, num=64)
ocb.get_aacgm_boundary_lat(aacgm_lon=np.degrees(lon), rec_ind=ocb.rec_ind)
ax.plot(lon, 90.0-ocb.aacgm_boundary_lat[ocb.rec_ind], "m-", linewidth=2, label="OCB")
ax.text(lon[35], lat[35]+1.5, "74$^\circ$", fontsize="medium", color="m")
```

Add reference labels for OCB coordinates
```
lon_clock = list()
lat_clock = list()

for ocb_mlt in np.arange(0.0, 24.0, 6.0):
    aa,oo = ocb.revert_coord(74.0, ocb_mlt)
    lon_clock.append(oo * np.pi / 12.0)
    lat_clock.append(90.0 - aa)

ax.plot(lon_clock, lat_clock, "m+")
ax.plot([lon_clock[0], lon_clock[2]], [lat_clock[0], lat_clock[2]], "-", color="lightpink", zorder=1)
ax.plot([lon_clock[1], lon_clock[3]], [lat_clock[1], lat_clock[3]], "-", color="lightpink", zorder=1)
ax.text(lon_clock[2]+.2, lat_clock[2]+1.0, "12:00",fontsize="medium",color="m")
ax.text(lon[35], olat[35]+1.5, "82$^\circ$", fontsize="medium", color="m")

```

Now add the location of a point in AACGM coordinates, calculate the
location relative to the OCB, and output both coordinates in the legend
```
aacgm_lat = 85.0
aacgm_lon = np.pi
ocb_lat, ocb_mlt = ocb.normal_coord(aacgm_lat, aacgm_lon * 12.0 / np.pi)
plabel = "Point (MLT, lat)\nAACGM (12:00,85.0$^\circ$)\nOCB ({:.0f}:{:.0f},{:.1f}$^\circ$)".format(np.floor(ocb_mlt), (ocb_mlt - np.floor(ocb_mlt))*60.0, ocb_lat)

ax.plot([aacgm_lon], [90.0-aacgm_lat], "ko", ms=5, label=plabel)

ax.legend(loc=2, fontsize="small", title="{:}".format(ocb.dtime[ocb.rec_ind]), bbox_to_anchor=(-0.4,1.15))
```


The figure should now look like:
<div align="center">
        <img height="0" width="0px">
        <img width="80%" src="/docs/example_ocb_location.png" alt="OCB Example" title="OCB Example"</img>
</div>
