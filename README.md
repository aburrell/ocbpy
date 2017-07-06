[![Documentation Status](https://readthedocs.org/projects/ocbpy/badge/?version=latest)](http://ocbpy.readthedocs.io/en/latest/?badge=latest)

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

* In Python, run:

```
import ocbpy


```

* The output should be as follows:

```

```

# Uninstallation 

1. The install directory for pyglow is outputted when you run the
   `python ./setup.py install` command.  For example, for macs this is usually
    `/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages`.
2.  Remove the `ocbpy` folder from this directory.
