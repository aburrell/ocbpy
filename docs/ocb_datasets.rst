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

Data from three auroral instruments provide northern hemisphere poleward auroral
boundary (PAB) and EAB locations for 3 May 2000 02:41:43 UT - 31 Oct 2002
20:05:16, though not all of the times included in these files contain
high-quality estimations of the boundary locations. Recommended selection
criteria are included as defaults in the :py:class:`~ocbpy.OCBoundary` class.
There are also boundary files that combine the information from all instruments
to obtain the OCB and EAB. These combined files are the default boundaries for
the IMAGE time period.  You can read more about the OCB determination, EAB
determination, this selection criteria, and the three auroral instruments
(IMAGE Wideband Imaging Camera (WIC) and FUV Spectrographic Imagers SI12 and
SI13) in the articles listed in :ref:`cite-image`.

The most recent corrects for each instrument that add the DMSP particle
precipitation corrections to the PAB and EAB locations are included in
:py:mod:`ocbpy.ocb_correction`.  These corrections should be applied to the
data used to obtain the circle fits included in the instrument files, not the
circle fits themselves. These data sets may be obtained from the British
Antarctic Survey.


.. _bound-data-ampere:

AMPERE
------

OCB data sets can also be obtained from AMPERE (Active Magnetosphere and
Planetary Electrodynamics Response Experiment) R1/R2 Field-Aligned Current (FAC)
boundary data.  This data is provided for both hemispheres between 2010-2021,
inclusive. Because there is an offset between the R1/R2 FAC boundary and the
OCB, a correction is required.  This correction can be implemented using the
routines in :py:mod:`ocbpy.ocb_correction`.

In the most recent version
(`V2 <https://figshare.le.ac.uk/articles/dataset/AMPERE_R1_R2_FAC_radii_v2/22241338/1>`_)
of these AMPERE boundary fits, which uses the newly-processed AMPERE data, a fit
for the Heppner-Maynard Boundary (HMB) were also made available. The HMB has
been shown to be related to the equatorward boundary of the auroral oval near
magnetic midnight, and so is provided as an EAB. This may or may not be an
appropriate boundary for your auroral gridding purposes. As the HMB has not
been related to the equatorward particle precipitation boundary, there is no
correction function needed.

More information about the method behind the identification of these boundaries
and their offset can be found in the articles listed in :ref:`cite-ampere`.
Recommended selection criteria are included as defaults in the
:py:class:`~ocbpy.OCBoundary` and :py:class:`~ocbpy.EABoundary` classes. We
gratefully acknowledge the use of AMPERE data provided by JHU/APL, PIs Brian
Anderson / Sarah Vines. This analysis uses the newly-processed AMPERE data.


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
