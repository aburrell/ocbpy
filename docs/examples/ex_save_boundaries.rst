.. _exsave:

Export and Save Boundaries
==========================

You may wish to save the boundary data contained in the various Boundary classes
to a file or use it in another data format. All the Boundary classes have a
:py:func:`to_dict` method that exports the data into two dictionaries, one
that contains the boundary information and one that contains the class metadata.
The dictionary may be formatted to simply hold the data using similarly-named
keys or may be formatted to allow the creation of an :py:class:`xarray.Dataset`.

::

   ocb_data, ocb_info = ocb.to_dict()
   print(ocb_info.keys(), '\n', ocb_data.keys())


This will show the following keys.  Note that all of the information keys are
single-value quantities that are useful for reproducibility, and the data keys
directly correspond to the class attributes in
:py:class:`~ocbpy._boundary.OCBoundary` or
:py:class:`~ocbpy._boundary.EABoundary` that are obtained directly or
calculated from the instrument boundary files.

::
   
   dict_keys(['instrument', 'filename', 'hemisphere', 'min_fom', 'max_fom', 'boundary_lat']) 
   dict_keys(['dtime', 'phi_cent', 'r_cent', 'r', 'fom', 'year', 'soy', 'num_sectors', 'a', 'r_err'])

The :py:class:`~ocbpy._boundary.DualBoundary` output is slightly different, as
it will contain the OCB and EAB pairs.

::

   dual_data, dual_info = dual.to_dict()
   print(dual_info.keys(), '\n', dual_data.keys())


Note that attributes pulled from each boundary have the boundary prepended to
the key.

::

   dict_keys(['eab_instrument', 'eab_filename', 'eab_min_fom', 'eab_max_fom', 'eab_boundary_lat', 'ocb_instrument', 'ocb_filename', 'ocb_min_fom', 'ocb_max_fom', 'ocb_boundary_lat', 'hemisphere', 'max_delta']) 
   dict_keys(['dtime', 'eab_phi_cent', 'eab_r_cent', 'eab_r', 'eab_fom', 'eab_year', 'eab_soy', 'eab_num_sectors', 'eab_a', 'eab_r_err', 'ocb_phi_cent', 'ocb_r_cent', 'ocb_r', 'ocb_fom', 'ocb_year', 'ocb_soy', 'ocb_num_sectors', 'ocb_a', 'ocb_r_err'])

You can create a netCDF file with this data using :py:mod:`xarray`.

::

   import xarray as xr

   dual_data, dual_info = dual.to_dict(xarray_style=True)
   dual_dataset = xr.Dataset(dual_data, attr=dual_info)
   dual_dataset.to_netcdf('/path/to/dual_boundary.nc')
