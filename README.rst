 |travis| |docs| |coveralls| |doi|

Overview
=========

ocbpy is a Python module that converts between AACGM coordinates and a magnetic
coordinate system that adjusts latitude and local time relative to the Open
Closed field line Boundary (OCB).  This is particulary useful for statistical
studies of the poles, where gridding relative to a fixed magnetic coordinate
system would cause averaging of different physical regions, such as auroral
and polar cap measurements.  This coordinate system is described in:

* Chisham, G. (2017), A new methodology for the development of high‐latitude
  ionospheric climatologies and empirical models, Journal of Geophysical
  Research: Space Physics, doi:10.1002/2016JA023235.
* `Full documentation <http://ocbpy.rtfd.io/>`_

OCBs must be obtained from observations for this coordinate transformation.
Data from three auroral instruments provide northern hemisphere OCB locations
for 3 May 2000 03:01:42 UT - 22 Aug 2002 00:01:28, though not all of the times
included in these files contain high-quality estimations of the OCB.
Recommended selection criteria are included as defaults in the OCBoundary class.

Currently, support is included for files from the following datasets:

* SuperMAG (available at http://supermag.jhuapl.edu)
* SuperDARN Vorticity (contact GC at gchi@bas.ac.uk)

These routines may be used as a guide to write routines for other datasets.

Python versions
===============

This module has been tested on python version 2.7, 3.4 - 3.6.  Local testing on
3.3 was also performed, but may not be supported in the next version.

Dependencies
============

The listed dependecies were tested with the following versions:

* datetime 
* numpy (1.11.3, 1.12.1, 1.14.1)
* logbook
* setuptools (36.0.1)

Testing is performed using the python module, unittest

Installation
============

Installation is now available through pypi |version|

::
   pip install ocbpy

   
First, checkout the repository:

::
   git clone git://github.com/aburrell/ocbpy.git
   
   
Change directories into the repository folder and run the setup.py file.  For
a local install use the "--user" flag after "install".

::
    cd ocbpy/
    python setup.py install

    
To run the unit tests,

::
    python setup.py test


Example
=======

In iPython, run:

::
   import numpy as np
   import ocbpy


Then initialise an OCB class object.  This uses the default IMAGE FUV file and
will take a few minutes to load.

::
   ocb = ocbpy.ocboundary.OCBoundary()
   print ocb
   
   Open-Closed Boundary file: ~/ocbpy/ocbpy/boundaries/si13_north_circle
   Source instrument: IMAGE
   Open-Closed Boundary reference latitude: 74.0 degrees

   219927 records from 2000-05-05 11:35:27 to 2002-08-22 00:01:28
 
   YYYY-MM-DD HH:MM:SS NumSectors Phi_Centre R_Centre R  R_Err Area
   ----------------------------------------------------------------------------
   2000-05-05 11:35:27 4 356.93 8.74 9.69 0.14 3.642e+06
   2000-05-05 11:37:23 5 202.97 13.23 22.23 0.77 1.896e+07
   2002-08-21 23:55:20 8 322.60 5.49 15.36 0.61 9.107e+06
   2002-08-22 00:01:28 7 179.02 2.32 19.52 0.89 1.466e+07

    
Get the first good OCB record, which will be record index 27.

::
   ocb.get_next_good_ocb_ind()
   print ocb.rec_ind
   27

If this works, your download should be working.  More detailed examples are
available in the full documentation.

.. |travis| image:: https://travis-ci.org/aburrell/ocbpy.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/aburrell/ocbpy

.. |coveralls| image:: https://coveralls.io/repos/github/aburrell/ocbpy/badge.svg?branch=master
    :alt: Coveralls Testing Coverage
    :target: https://coveralls.io/github/aburrell/ocbpy?branch=master

.. |doi| image:: https://zenodo.org/badge/96153180.svg
    :alt: Reference DOI
    :target: https://zenodo.org/badge/latestdoi/96153180

.. |docs| image:: https://readthedocs.org/projects/ocbpy/badge/?version=latest
    :target: http://ocbpy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |version| image:: https://img.shields.io/pypi/pyversions/ocbpy.svg?style=flat
    :alt: PyPi Package download
    :target: https://pypi.python.org/pypi/ocbpy  
