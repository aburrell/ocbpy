.. _exinit:

Getting Started
===============

:py:mod:`ocbpy` is centred around Boundary class objects.  The default class
for most functions and examples, is the Open-Closed field line Boundary class,
:py:class:`ocbpy.OCBoundary`.


Initialise an OCBoundary object
-------------------------------
Start a python or iPython session, and begin by importing ocbpy, numpy,
matplotlib, and datetime.

::

   
   import numpy as np
   import datetime as dt
   import matplotlib as mpl
   import matplotlib.pyplot as plt
   import ocbpy
  
Next, initialise an OCB class object.  This uses the default IMAGE FUV file and
will take a few minutes to load.

::

   
   ocb = ocbpy.OCBoundary()
   print(ocb)
  
   OCBoundary file: ~/ocbpy/ocbpy/boundaries/si13_north_circle
   Source instrument: IMAGE
   Boundary reference latitude: 74.0 degrees
  
   219927 records from 2000-05-05 11:35:27 to 2002-08-22 00:01:28
  
   YYYY-MM-DD HH:MM:SS Phi_Centre R_Centre R
   -----------------------------------------
   2000-05-05 11:35:27 356.93 8.74 9.69
   2000-05-05 11:37:23 202.97 13.23 22.23
   2002-08-21 23:55:20 322.60 5.49 15.36
   2002-08-22 00:01:28 179.02 2.32 19.52

   Uses scaling function(s):
   ocbpy.ocb_correction.circular(**{})


Other Boundary classes
----------------------

The other Boundary classes, :py:class:`ocbpy.EABoundary` and
:py:class:`ocbpy.DualBoundary`, build upon the `ocbpy.OCBoundary` class.
Initialising these classes is done bascially the same way.  To make this example
run more quickly, we will limit the period of time over which boundaries are
loaded.  The same temporal selection procedure works with the other Boundary
classes.


::


   stime = dt.datetime(2000, 5, 5)
   etime = dt.datetime(2000, 5, 8)

   dual = ocbpy.DualBoundary(stime=stime, etime=etime)
   print(dual)
  
   Dual Boundary data
   13 good boundary pairs from 2000-05-05 11:58:48 to 2000-05-05 15:30:23
   Maximum  boundary difference of 60.0 s

   EABoundary file: ~/ocbpy/ocbpy/boundaries/image_north_circle.eab
   Source instrument: IMAGE
   Boundary reference latitude: 64.0 degrees

   81 records from 2000-05-05 11:35:27 to 2000-05-07 23:32:14

   YYYY-MM-DD HH:MM:SS Phi_Centre R_Centre R
   -----------------------------------------------------------------------------
   2000-05-05 11:35:27 111.80 2.34 25.12
   2000-05-05 11:37:23 296.23 1.42 26.57
   2000-05-07 21:50:20 220.84 7.50 16.89
   2000-05-07 23:32:14 141.27 10.33 18.32

   Uses scaling function(s):
   ocbpy.ocb_correction.circular(**{})

   OCBoundary file: ~/ocbpy/ocbpy/boundaries/image_north_circle.ocb
   Source instrument: IMAGE
   Boundary reference latitude: 74.0 degrees

   76 records from 2000-05-05 11:19:52 to 2000-05-07 23:32:14

   YYYY-MM-DD HH:MM:SS Phi_Centre R_Centre R
   -----------------------------------------------------------------------------
   2000-05-05 11:19:52 218.54 9.44 11.48
   2000-05-05 11:35:27 304.51 8.69 15.69
   2000-05-07 23:24:14 199.84 10.91 12.69
   2000-05-07 23:32:14 141.53 9.24 13.03
   

   Uses scaling function(s):
   ocbpy.ocb_correction.circular(**{})
