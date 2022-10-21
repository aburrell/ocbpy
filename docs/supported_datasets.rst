Supported Instrument Data Sets
==============================

Currently, support is included for files from the following sources:

#. SuperMAG (:py:mod:`ocbpy.instruments.supermag`)
#. SuperDARN Vorticity (:py:mod:`ocbpy.instruments.vort`)
#. pysat (:py:mod:`ocbpy.instruments.pysat_instruments`)

These routines may be used as a guide to write routines for other data sets.
A :py:mod:`ocbpy.instruments.general` loading sub-module is also provided for
ASCII files. All the non-boundary data routines are part of the
:py:mod:`ocbpy.instruments` module. Support for time-handling that may be useful
for specific data sets are provided in :py:mod:`ocbpy.ocb_time`.


General Instrument Module
-------------------------

.. automodule:: ocbpy.instruments.general
    :members: 

SuperMAG Instrument Module
--------------------------

.. automodule:: ocbpy.instruments.supermag
    :members:

SuperDARN Vorticity Instrument Module
-------------------------------------

.. automodule:: ocbpy.instruments.vort
    :members:

pysat Instrument Module
-----------------------

.. automodule:: ocbpy.instruments.pysat_instruments
    :members:

Time Handling Module
--------------------

.. automodule:: ocbpy.ocb_time
    :members:
