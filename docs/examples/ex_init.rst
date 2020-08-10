Initialise an OCBoundary object
--------------------------------
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

   
   ocb = ocbpy.ocboundary.OCBoundary()
   print(ocb)
  
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
