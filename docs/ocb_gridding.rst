.. _ocbgrid:

OCB Gridding
============

OCB gridding is performed by matching observations and OCBs in Universal Time
(UT) and then normalising the AACGM magnetic coordinates of the observation
to OCB coordinates.  This is done by determining the observation's location
relative to the current OCB and placing it in the same location relative to
a typical OCB that has a magnetic latitude of 74 degrees.  Data matching is
performed by :py:func:`ocbpy.ocboundary.match_data_ocb`.  Coordinate
normalisation, as well as OCB loading and data cycling is done within
:py:class:`ocbpy.ocboundary.OCBoundary`. These classes and functions make up
the :py:mod:`ocbpy.ocboundary` module.

For observations that depend on the cross polar cap potential, it is also
important to scale the magnitude.  This ensures that the magnitudes from
different sized polar caps compare to the *typical* polar cap the OCB gridding
produces.  For vector data, the local polar north and east components may also
change.  Magnitude scaling is performed by
:py:func:`ocbpy.ocb_scaling.normal_evar` or
:py:func:`ocbpy.ocb_scaling.normal_curl_evar`. Vector scaling, re-orientation,
and OCB coordinate normalisation are performed within the class
:py:class:`~ocbpy.ocb_scaling.VectorData`.  These classes and functions make up
the :py:mod:`ocbpy.ocb_scaling` module.


.. _ocbgrid-ocb:

OCBoundary Module
-----------------
.. automodule:: ocbpy.ocboundary
    :members:


.. _ocbgrid-scale:

OCB Scaling Module
------------------
.. automodule:: ocbpy.ocb_scaling
    :members:
