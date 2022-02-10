.. _ocbgrid:

OCB Gridding
============

OCB and dual-boundary gridding is performed by matching observations and OCBs
and/or EABs in Universal Time (UT) and then normalising the AACGM magnetic
coordinates of the observation to boundary coordinates.  This is done by
determining the observation's location relative to the current boundary and
placing it in the same location relative to a typical OCB and/or EAB.  For the
OCB, this defaults to 74 degrees, while for the EAB, this defaults to 64
degrees.  Data matching is performed by
:py:func:`ocbpy.cycle_boundary.match_data_ocb`.  Coordinate normalisation, as
well as boundary loading and data cycling is done within the appropriate
boundary classes: :py:class:`ocbpy.OCBoundary`, :py:class:`ocbpy.EABoundary`,
and :py:class:`ocbpy.DualBoundary`.

For observations that depend on the cross polar cap potential, it is also
important to scale the magnitude.  This ensures that the magnitudes from
different sized polar caps compare to the *typical* polar cap the OCB gridding
produces.  For vector data, the local polar north and east components may also
change.  Magnitude scaling is performed by
:py:func:`ocbpy.ocb_scaling.normal_evar` or
:py:func:`ocbpy.ocb_scaling.normal_curl_evar`. Vector scaling, re-orientation,
and boundary coordinate normalisation are performed within the class
:py:class:`~ocbpy.ocb_scaling.VectorData`.  These classes and functions make up
the :py:mod:`ocbpy.ocb_scaling` module.


.. _ocbgrid-ocb:

Boundary Classes
----------------
.. automodule:: ocbpy.OCboundary
    :members:

.. automodule:: ocbpy.EAboundary
    :members:

.. automodule:: ocbpy.Dualboundary
    :members:


.. _ocbgrid-cycle:

Cycle Boundary Module
---------------------
.. automodule:: ocbpy.cycle_boundaries
    :members:


.. _ocbgrid-scale:

OCB Scaling Module
------------------
.. automodule:: ocbpy.ocb_scaling
    :members:
