.. _ex-rec:

Selecting Boundaries
====================

Each of the :ref:`bound-data` has a figure of merit that can be used
to select quality records. The various Boundary classes also contain methods
to cycle to the next good record.  Unlike standard Python indices, the Boundary
:py:attr:`rec_ind` must be positive.  This allows the user to know whether or
not quality Boundaries have been identified.

Retrieve a good OCB record
--------------------------
Continuing from the previous example, our next step is to get the first good
OCB record.  The :py:class:`ocbpy.OCBoundary` and :py:class:`ocbpy.EABoundary`
objects start without any good indices chosen to allow you to specify your
desired selection criteria.  This section uses the ``ocb`` from the previous
section (with default keyword arguements).

::

   
   ocb.get_next_good_ocb_ind()
   print(ocb.rec_ind)
   0

To get the OCB record closest to a specified time, use
:py:func:`~ocbpy.cycle_boundary.match_data_ocb`.  In this example the maximum
time tolerance is changed from the default of 60 seconds to 30 seconds to ensure
only one of the test times (that have a 60 second resolution) is returned.

::

   
   test_times = [ocb.dtime[ocb.rec_ind] + dt.timedelta(minutes=(i - 2))
                 for i in range(5)]
   itest = ocbpy.match_data_ocb(ocb, test_times, idat=0, max_tol=30)
   print(itest, ocb.rec_ind, test_times[itest], ocb.dtime[ocb.rec_ind])

   2 0 2000-05-04 03:03:20 2000-05-04 03:03:20

Retrive a good DualBoundary record
----------------------------------
The :py:class:`ocbpy.DualBoundary` class starts with good Boundaries selected
using the default criteria for the given instrument.  You can change that at
any time, as shown below.  This example uses the ``dual`` variable set in the
first example.

::

   # Before resetting, check that this is: 0 13 12 12
   print(dual.rec_ind, dual.records, dual.ocb.rec_ind, dual.eab.rec_ind)

   # Reset the good boundary pairs using more restrictive FOM values
   dual.set_good_ind(ocb_max_merit=dual.ocb.max_fom - 1.0,
                     eab_max_merit=dual.eab.max_fom - 1.0)
   print(dual.rec_ind, dual.records, dual.ocb.rec_ind, dual.eab.rec_ind)

   0 9 14 14


Cycling to the next record will increment :py:attr:`~ocbpy.DualBoundary.rec_ind`
by one and updates the sub-class record indices to their next good paired
values.

::

   dual.get_next_good_ind()
   print(dual.rec_ind, dual.records, dual.ocb.rec_ind, dual.eab.rec_ind)

   1 9 17 18
