.. _ex-rec:

Selecting Boundaries
====================

Each Boundary [data set]:ref:`bound-data` has a figure of merit that can be used
to select quality records. The various Boundary classes also contain methods
to cycle to the next good record.  Unlike standard Python indices, the Boundary
:py:attr:`rec_ind` must be positive.  This allows the user to know that no
quality Boundaries have been identified.

Retrieve a good OCB record
--------------------------
Continuing from the previous example, our next step is to get the first good
OCB record.  The :py:class:`ocbpy.OCBoundary` and :py:class:`ocbpy.EABoundary`
objects start without any good indices chosen to allow you to specify your
desired selection criteria.  Using the default selection criteria for ``ocb``
from the previous example should yeild ``ocb.rec_ind == 27``.

::

   
   ocb.get_next_good_ocb_ind()
   print(ocb.rec_ind)

To get the OCB record closest to a specified time, use
:py:func:`~ocbpy.cycle_boundary.match_data_ocb`

::

   
   first_good_time = ocb.dtime[ocb.rec_ind]
   test_times = [first_good_time + dt.timedelta(minutes=5 * (i + 1))
                 for i in range(5)]
   itest = ocbpy.match_data_ocb(ocb, test_times, idat=0)
   print(itest, ocb.rec_ind, test_times[itest], ocb.dtime[ocb.rec_ind])
  
   0 31 2000-05-05 13:45:30 2000-05-05 13:50:29

Retrive a good DualBoundary record
----------------------------------
The :py:class:`ocbpy.DualBoundary` class starts with good Boundaries selected
using the default criteria for the given instrument.  You can change that at
any time, as shown below.

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
