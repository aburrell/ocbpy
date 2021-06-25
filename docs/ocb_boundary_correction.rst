OCB Boundary Correction
=======================

Many high-latitude boundaries are related to each other.  Both the poleward
edge of the auroral oval and the R1/R2 current boundary have been successfully
related to the OCB.  If you have a data set of boundaries that can be related
to the OCB, OCBpy is capable of applying this correction as a function of MLT.
These corrections are applied using the :py:data:`rfunc` and
:py:data:`rfunc_kwargs` keyword arguments in :py:class:`ocbpy.OCBoundary`
object.  Several correction functions are provided as a part of
:py:mod:`ocbpy.ocb_correction` module.

OCB Correction Module
---------------------
.. automodule:: ocbpy.ocb_correction
    :members:
