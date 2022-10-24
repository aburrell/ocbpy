[![Documentation Status](https://readthedocs.org/projects/ocbpy/badge/?version=latest)](http://ocbpy.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/96153180.svg)](https://zenodo.org/badge/latestdoi/96153180)
[![PyPI version](https://badge.fury.io/py/ocbpy.svg)](https://badge.fury.io/py/ocbpy)

[![Test Status](https://github.com/aburrell/ocbpy/actions/workflows/main.yml/badge.svg)](https://github.com/aburrell/ocbpy/actions/workflows/main.yml)
[![Coverage Status](https://coveralls.io/repos/github/aburrell/ocbpy/badge.svg)](https://coveralls.io/github/aburrell/ocbpy)

<h1> <img width="128" height="128" src="/docs/figures/ocbpy_logo.gif" alt="Planet with auroral oval and two pythons representing closed and open magnetic field lines" title="OCBpy Logo" style="float:left;">
Overview </h1>

OCBpy is a Python module that converts between AACGM coordinates and a magnetic
coordinate system that adjusts latitude and local time relative to the Open
Closed field line Boundary (OCB), Equatorial Auroral Boundary (EAB), or both.
This is particulary useful for statistical studies of the poles, where gridding
relative to a fixed magnetic coordinate system would cause averaging of
different physical regions, such as auroral and polar cap measurements.  This
coordinate system is described in:

  * Chisham, G. (2017), A new methodology for the development of high‐latitude
    ionospheric climatologies and empirical models, Journal of Geophysical
    Research: Space Physics,
    [doi:10.1002/2016JA023235.](https://doi.org/10.1002/2016JA023235)

  * Full [documentation](http://ocbpy.rtfd.io/)

Boundaries must be obtained from observations or models for this coordinate
transformation. Several boundary data sets are included within this package.
These include northern hemisphere boundaries from the IMAGE satellite,
northern and southern hemisphere OCBs from AMPERE, and single-point boundary
locations from DMSP.

  * [IMAGE Auroral Boundary data](https://www.bas.ac.uk/project/image-auroral-boundary-data/)
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

This module currently supports Python version 3.7 - 3.10.

# Dependencies

The listed dependecies were tested with the following versions:
  * numpy
  * aacgmv2
  * pysat (3.0.1+)
  * ssj_auroral_boundary

Testing is performed using the python module, unittest.  To limit dependency
issues, pysat (>=3.0.1) and ssj_auroral_boundary are optional dependencies.

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
    $ python -m unittest discover
```

# Example

In iPython, run:

```
import datetime as dt
import ocbpy
```

Then initialise an OCB class object.  This uses the default IMAGE FUV file and
will take a few minutes to load.

```
ocb = ocbpy.OCBoundary()
print(ocb)
```

The output should be as follows:

```
Open-Closed Boundary file: ~/ocbpy/ocbpy/boundaries/image_north_circle.ocb
Source instrument: IMAGE
Boundary reference latitude: 74.0 degrees

305805 records from 2000-05-04 03:03:20 to 2002-10-31 20:05:16

YYYY-MM-DD HH:MM:SS Phi_Centre R_Centre R
-----------------------------------------------------------------------------
2000-05-04 03:03:20 4.64 2.70 21.00
2000-05-04 03:07:15 147.24 2.63 7.09
2002-10-31 20:03:16 207.11 5.94 22.86
2002-10-31 20:05:16 335.47 6.76 11.97

Uses scaling function(s):
ocbpy.ocb_correction.circular(**{})
```

Get the first good OCB record, which will be record index 0.

```
ocb.get_next_good_ocb_ind()
print(ocb.rec_ind)
```

To get the good OCB record closest to a specified time (with a maximum of a
60 sec time difference, as a default), use **ocbpy.match_data_ocb**

```
test_times = [dt.datetime(otime.year, otime.month, otime.day, otime.hour,
                          otime.minute, 0) for otime in ocb.dtime[1:10]]
itest = ocbpy.match_data_ocb(ocb, test_times, idat=0)
print(itest, ocb.rec_ind, test_times[itest], ocb.dtime[ocb.rec_ind])

4 5 2000-05-05 11:39:00 2000-05-05 11:39:20
```

More examples are available in the documentation.
