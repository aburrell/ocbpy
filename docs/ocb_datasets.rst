.. _bound-data:


Boundary Data Sets
==================

Poleward (OCB) and Equatorward Auroral Boundaries (EABs) must be obtained from
observations or a model for this coordinate transformation. The standard OCB
and EAB data sets can be found in ``ocbpy/boundaries``, though this location may
also found using :py:func:`ocbpy.boundaries.files.get_boundary_directory`.
Currently, three different boundary data set types are available.  Not all data
sets include the locations of the EAB. Routines to retrieve boundary filenames
from specific instruments, time periods, hemispheres, and boundary types are
provided in the :py:mod:`ocbpy.boundaries.files` sub-module.


.. _bound-data-image:

IMAGE
-----

Data from three auroral instruments provide northern hemisphere OCB and EAB
locations for 3 May 2000 02:41:43 UT - 31 Oct 2002 20:05:16, though not all of
the times included in these files contain high-quality estimations of the
boundary locations. Recommended selection criteria are included as defaults in
the :py:class:`~ocbpy.ocboundary.OCBoundary` class. There are also boundary
files that combine the information from all instruments to obtain the OCB and
EAB. You can read more about the OCB determination, EAB determination, this
selection criteria, and the three auroral instruments (IMAGE Wideband Imaging
Camera (WIC) and FUV Spectrographic Imagers SI12 and SI13) in the articles
listed in :ref:`cite-image`.


.. _bound-data-ampere:

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


.. _bound-data-dmsp-ssj:

DMSP SSJ
--------

DMSP particle precipitation instruments make it possible to identify the
poleward and equatorward boundaries of the auroral oval along the satellite
orbit.  Details about this identification process can be found in the references
listed in :ref:`cite-dmsp`.  Routines to download and proceess the DMSP boundary
files are provided in the :py:mod:`ocbpy.boundaries.dmsp_ssj_files` sub-module.


.. _bound-data-module:

Boundaries Module
-----------------
.. automodule:: ocbpy.boundaries


.. _bound-data-module-files:

Boundary Files
^^^^^^^^^^^^^^

.. automodule:: ocbpy.boundaries.files
    :members:


.. _bound-data-module-dsfiles:

DMSP SSJ Files
^^^^^^^^^^^^^^

.. automodule:: ocbpy.boundaries.dmsp_ssj_files
    :members:
