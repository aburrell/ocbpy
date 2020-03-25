Supported Data Sets
===================

Currently, support is included for files from the following sources:

  * `SuperMAG <SuperMag_>`_
  * `SuperDARN Vorticity <SuperDARN Vorticity_>`_
  * `pysat <pysat>`_

These routines may be used as a guide to write routines for other data sets.
A `general data <General_>`_ loading routine is also provided for ASCII files.
These routines are part of the **ocbpy.instruments** module.  Supporting
time-handling routines that may be useful for specific data sets are provided
in `ocbpy.ocb_time <Time Handling_>`_.


General
-------
.. automodule:: ocbpy.instruments.general
    :members: 

SuperMAG
--------

.. automodule:: ocbpy.instruments.supermag
    :members:

SuperDARN Vorticity
-------------------

.. automodule:: ocbpy.instruments.vort
    :members:

pysat
-----

.. automodule:: ocbpy.instruments.pysat_instruments
    :members:

Time Handling
-------------

.. automodule:: ocbpy.ocb_time
    :members:
