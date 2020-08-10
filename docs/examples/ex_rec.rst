Retrieve a good OCB record
--------------------------
Get the first good OCB record, which will be record index 27.

::

   
   ocb.get_next_good_ocb_ind()
   print(ocb.rec_ind)

To get the OCB record closest to a specified time, use **ocbpy.match_data_ocb**

::

   
   first_good_time = ocb.dtime[ocb.rec_ind]
   test_times = [first_good_time + dt.timedelta(minutes=5*(i+1)) for i in range(5)]
   itest = ocbpy.match_data_ocb(ocb, test_times, idat=0)
   print(itest, ocb.rec_ind, test_times[itest], ocb.dtime[ocb.rec_ind])
  
   0 31 2000-05-05 13:45:30 2000-05-05 13:50:29
