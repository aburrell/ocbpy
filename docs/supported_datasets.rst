Supported Datasets
===================

Currently, support is included for files from the following datasets:

  * `SuperMAG <SuperMag_>`_
  * `SuperDARN Vorticity <SuperDARN Vorticity_>`_
  * `Madrigal TEC <Madrigal TEC_>`_

These routines may be used as a guide to write routines for other datasets.
A `general data <General_>`_ loading routine is also provided for ASCII files.
These routines are part of the **ocbpy.instruments** module.


General
--------------------
.. automodule:: ocbpy.instruments.general
    :members: 

SuperMAG
---------------------
.. automodule:: ocbpy.instruments.supermag
    :members:

SuperDARN Vorticity
-----------------------------
.. automodule:: ocbpy.instruments.vort
    :members:

Madrigal TEC
-----------------------------
This dataset requires the additional installation of davitpy to access AACGM,
since the TEC are provided in geographic coordinates.

.. automodule:: ocbpy.instruments.tec
    :members:
