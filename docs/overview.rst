Overview
============

One of the challenges of working in the polar
Magnetosphere-Ionosphere-Thermosphere (MIT) system is choosing an appropriate
coordinate system.  The ocbpy module converts between the Altitude Adjusted
Corrected GeoMagnetic
`(AACGM) coordinates <http://superdarn.thayer.dartmouth.edu/aacgm.html>`_ and a
grid that is constructed relative to the Open Closed field line Boundary (OCB).
This is particulary useful for statistical studies of the poles, where gridding
relative to a fixed magnetic coordinate system would cause averaging of
different physical regions, such as auroral and polar cap measurements.  This
coordinate system is described in
`Chisham (2017) <http://onlinelibrary.wiley.com/doi/10.1002/2016JA023235/pdf>`_.
