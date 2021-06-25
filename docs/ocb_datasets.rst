OCB Data Sets
=============

OCBs must be obtained from observations for this coordinate transformation.
The standard OCB data sets can be found in ``ocbpy/boundaries``, though this
location may also found using
:py:func:`ocbpy.boundaries.files.get_boundary_directory`.  Currently,
three different boundary data set types are available.  Some data sets also
include the locations of the Equatorward Auroral oval Boundary (EAB). Routines
to retrieve boundary filenames from specific instruments, time periods,
hemispheres, and boundary types are provided in the
:py:mod:`ocbpy.boundaries.files` sub-module.

IMAGE
-----

Data from three auroral instruments provide northern hemisphere OCB locations
for 3 May 2000 03:01:42 UT - 22 Aug 2002 00:01:28, though not all of the times
included in these files contain high-quality estimations of the OCB.
Recommended selection criteria are included as defaults in the
:py:class:`~ocbpy.ocboundary.OCBoundary` class. You can read more about the OCB
determination and this selection criteria, as well as the three auroral
instruments (IMAGE Wideband Imaging Camera (WIC) and FUV Spectrographic Imagers
SI12 and SI13) in the articles listed in :ref:`cite-image`.

AMPERE
------

OCB data sets can also be obtained from AMPERE (Active Magnetosphere and
Planetary Electrodynamics Response Experiment) R1/R2 Field-Aligned Current (FAC)
boundary data.  This data is provided for both hemispheres between 2010-2016,
inclusive. Because there is an offset between the R1/R2 FAC boundary and the
OCB, a correction is required.  This correction can be implemented using the
routines in :py:mod:`ocbpy.ocb_correction`.  More information about the method
behind the identification of these boundaries and their offset can be found in
the articles listed in :ref:`cite-ampere`. Recommended selection criteria are
included as defaults in the :py:class:`~ocbpy.OCBoundary` class. 

DMSP SSJ
--------

DMSP particle precipitation instruments make it possible to identify the
poleward and equatorward boundaries of the auroral oval along the satellite
orbit.  Details about this identification process can be found in the references
listed in :ref:`cite-dmsp`.  Routines to download and proceess the DMSP boundary
files are provided in the :py:mod:`ocbpy.boundaries.dmsp_ssj_files` sub-module.

Boundary File Module
--------------------

.. automodule:: ocbpy.boundaries.files
    :members:

DMSP SSJ File Module
--------------------

.. automodule:: ocbpy.boundaries.dmsp_ssj_files
    :members:
