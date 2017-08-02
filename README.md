[![Build Status](https://www.travis-ci.org/aburrell/ocbpy.svg?branch=master)](https://www.travis-ci.org/aburrell/ocbpy)	[![Coverage Status](https://coveralls.io/repos/github/aburrell/ocbpy/badge.svg)](https://coveralls.io/github/aburrell/ocbpy)	[![Documentation Status](https://readthedocs.org/projects/ocbpy/badge/?version=latest)](http://ocbpy.readthedocs.io/en/latest/?badge=latest)

# Overview

ocbpy is a Python module that converts between AACGM coordinates and a magnetic
coordinate system that adjusts latitude and local time relative to the Open
Closed field line Boundary (OCB).  This is particulary useful for statistical
studies of the poles, where gridding relative to a fixed magnetic coordinate
system would cause averaging of different physical regions, such as auroral
and polar cap measurements.  This coordinate system is described in:

  * Chisham, G. (2017), A new methodology for the development of high‐latitude
    ionospheric climatologies and empirical models, Journal of Geophysical
    Research: Space Physics, doi:10.1002/2016JA023235.

  * Full [documentation](http://ocbpy.rtfd.io/)

OCBs must be obtained from observations for this coordinate transformation.
Data from three auroral instruments provide northern hemisphere OCB locations
for 3 May 2000 03:01:42 UT - 22 Aug 2002 00:01:28, though not all of the times
included in these files contain high-quality estimations of the OCB.
Recommended selection criteria are included as defaults in the OCBoundary class.

Currently, support is included for files from the following datasets:

  * SuperMAG (available at http://supermag.jhuapl.edu)
  * SuperDARN Vorticity (contact GC at gchi@bas.ac.uk)

These routines may be used as a guide to write routines for other datasets.

# Dependencies

The listed dependecies were tested with the following versions:
  * datetime 
  * numpy (1.12.1)
  * logging (0.5.1.2)
  * os 
  * setuptools (36.0.1)

These additional packages are needed to perform unit tests
  * unittest
  * filecmp

# Installation

First, checkout the repository:

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
Open-Closed Boundary file: /Users/ab763/Programs/Git/ocbpy/ocbpy/boundaries/si13_north_circle

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
ax.yaxis.set_ticklabels(["85$^\circ$", "80$^\circ$", "75$^\circ$", "70$^\circ$"]
```

Mark the location of the reference OCB, set at 74 degrees
```
lon = np.arange(0.0, 2.0 * np.pi + 0.1, 0.1)
ref_lat = np.ones(shape=lon.shape) * 16.0
ax.plot(lon, ref_lat, "--", linewidth=2, color="0.6", label="Reference OCB")
```

Mark the location of the circle centre in AACGM coordinates
```
phi_cent_rad = np.radians(ocb.phi_cent[ocb.rec_ind])
ax.plot([phi_cent_rad], [ocb.r_cent[ocb.rec_ind]], "mx", ms=10, label="OCB Pole")
```

Calculate at plot the location of the OCB in AACGM coordinates
```
del_lon = lon - phi_cent_rad
ax.plot(lon, lat, "m-", linewidth=2, label="OCB")
```

Now add the location of a point in AACGM coordinates
```
aacgm_lat = 85.0
aacgm_lon = np.pi

ax.plot([aacgm_lon], [90.0-aacgm_lat], "ko", ms=5, label="AACGM Point")
```

Find the location relative to the current OCB and add a legend
```
ocb_lat, ocb_mlt = ocb.normal_coord(aacgm_lat, aacgm_lon * 12.0 / np.pi)
ax.plot([ocb_mlt * np.pi / 12.0], [90.0 - ocb_lat], "mo", label="OCB Point")
ax.legend(loc=2, fontsize="medium", bbox_to_anchor=(-0.4,1.15), title="{:}".format(ocb.dtime[ocb.rec_ind]))
```

The figure should now look like:
<div align="center">
        <img height="0" width="0px">
        <img width="80%" src="/docs/example_ocb_location.png" alt="OCB Example" title="OCB Example"</img>
</div>


# Uninstallation 

1. The install directory for pyglow is outputted when you run the
   `python ./setup.py install` command.  For example, for macs this is usually
    `/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages`.
2.  Remove the `ocbpy` folder from this directory.
