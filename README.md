[![Linux Status](https://www.travis-ci.org/aburrell/ocbpy.svg)](https://www.travis-ci.org/aburrell/ocbpy)
[![Windows Status](https://ci.appveyor.com/api/projects/status/741n3cv8n68s280v?svg=true)](https://ci.appveyor.com/project/aburrell/ocbpy)
[![Coverage Status](https://coveralls.io/repos/github/aburrell/ocbpy/badge.svg)](https://coveralls.io/github/aburrell/ocbpy)
[![Documentation Status](https://readthedocs.org/projects/ocbpy/badge/?version=latest)](http://ocbpy.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/96153180.svg)](https://zenodo.org/badge/latestdoi/96153180)
[![PyPI version](https://badge.fury.io/py/ocbpy.svg)](https://badge.fury.io/py/ocbpy)

<h1> <img width="128" height="128" src="/docs/figures/ocbpy_logo.gif" alt="Planet with auroral oval and two pythons representing closed and open magnetic field lines" title="OCBpy Logo" style="float:left;">
Overview </h1>

OCBpy is a Python module that converts between AACGM coordinates and a magnetic
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
OCBpy also supports boundaries provided by AMPERE and DMSP:

  * Burrell, A. G. et al. (2020): AMPERE Polar Cap Boundaries, Ann. Geophys.,
    38, 481-490,
    [doi:10.5194/angeo-38-481-2020](https://doi.org/10.5194/angeo-38-481-2020)
  * [ssj_auroral_boundary](https://github.com/lkilcommons/ssj_auroral_boundary)

Currently, support is included for files from the following datasets:

  * SuperMAG (available at http://supermag.jhuapl.edu)
  * SuperDARN Vorticity (contact GC at gchi@bas.ac.uk)
  * Any pysat Instrument (available at https://github.com/pysat/pysat)

These routines may be used as a guide to write routines for other datasets.

# Python versions

This module has been tested on python version 2.7, 3.5 - 3.8.  Support for 2.7
will be dropped in 2020.

# Dependencies

The listed dependecies were tested with the following versions:
  * numpy
  * aacgmv2
  * pysat (2.0.0+)
  * ssj_auroral_boundary

Testing is performed using the python module, unittest.  To limit dependency
issues, pysat (>=2.0.0) and ssj_auroral_boundary are optional dependencies.

# Installation

Installation is now available through pypi

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
print(ocb)
```

The output should be as follows:

```
Open-Closed Boundary file: ~/ocbpy/ocbpy/boundaries/si13_north_circle
Source instrument: IMAGE
Open-Closed Boundary reference latitude: 74.0 degrees

219927 records from 2000-05-05 11:35:27 to 2002-08-22 00:01:28

YYYY-MM-DD HH:MM:SS Phi_Centre R_Centre R
-----------------------------------------
2000-05-05 11:35:27 356.93 8.74 9.69
2000-05-05 11:37:23 202.97 13.23 22.23
2002-08-21 23:55:20 322.60 5.49 15.36
2002-08-22 00:01:28 179.02 2.32 19.52
```

Get the first good OCB record, which will be record index 27.

```
ocb.get_next_good_ocb_ind()
print(ocb.rec_ind)

27
```

To get the OCB record closest to a specified time, use **ocbpy.match_data_ocb**

```
first_good_time = ocb.dtime[ocb.rec_ind]
test_times = [first_good_time + dt.timedelta(minutes=5*(i+1)) for i in range(5)]
itest = ocbpy.match_data_ocb(ocb, test_times, idat=0)
print(itest, ocb.rec_ind, test_times[itest], ocb.dtime[ocb.rec_ind])
  
0 31 2000-05-05 13:45:30 2000-05-05 13:50:29
```

More examples are available in the documentation.