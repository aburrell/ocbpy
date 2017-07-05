OCB Gridding
=============

OCB gridding is performed by matching observations and OCBs in Universal Time
(UT) and then normalising the AACGM magnetic coordinates of the observation
to OCB coordinates.  This is done by determining the observation's location
relative to the current OCB and placing it in the same location relative to
a typical OCB that has a magnetic latitude of 74 degrees.  Data matching is
performed by `ocbpy.ocboundary.match_data_ocb <ocbpy.ocboundary.match_data_ocb_>`_.  Coordinate normalisation,
as well as OCB loading and data cycling is done within
ocbpy.ocboundary.OCBoundary.
These classes and functions make up the **ocbpy.ocbounary**
`module <OCBoundary_>`_.

For observations that depend on the cross polar cap potential, it is also
important to scale the magnitude.  This ensures that the magnitudes from
different sized polar caps compare to the *typical* polar cap the OCB gridding
produces.  For vector data, the local polar north and east components may also
change.  Magnitude scaling is performed by
`ocbpy.ocb_scaling.normal_evar <ocbpy.ocb_scaling.normal_evar_>`_ or
`ocbpy.ocb_scaling.normal_curl_evar <ocbpy.ocb_scaling.normal_curl_evar_>`_.
Vector scaling, re-orientation, and OCB coordinate normalisation are performed
within the class VectorData.  These classes
and functions make up the **ocbpy.ocb_scaling** `module <OCB Scaling_>`_.

OCBoundary
-----------
.. automodule:: ocbpy.ocboundary
    :members:

OCB Scaling
-----------
.. automodule:: ocbpy.ocb_scaling
    :members:
