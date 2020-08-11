#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB & GC
# Full license can be found in License.md
# -----------------------------------------------------------------------------
""" Download and format DMSP SSJ boundary files

Functions
---------
fetch_ssj_files(stime, etime, [out_dir, sat_nums])
    Download DMSP SSJ files
create_ssj_boundary_files(cdf_files, [out_dir, out_cols, make_plots, plot_dir])
    Obtain pass boundaries from DMSP SSJ files
format_ssj_boundary_files(csv_files, [ref_alt, method])
    Format the individual pass boundaries, grouping data by boundary and
    hemisphere
fetch_format_ssj_boundary_files(stime, etime, [out_dir, rm_temp, ref_alt,
                                               method])
    Download and format the individual pass boundaries

Moduleauthor
------------
Angeline G. Burrell (AGB), 25 September 2019, Naval Research Laboratory (NRL)

References
----------
Kilcommons, L.M., et al. (2017), A new DMSP magnetometer and auroral boundary
 data set and estimates of field-aligned currents in dynamic auroral boundary
 coordinates, J. Geophys. Res.: Space Phys., 122, pp 9068-9079,
 doi:10.1002/2016ja023342,

"""

from __future__ import absolute_import, unicode_literals

import datetime as dt
import numpy as np
import os
import sys

import ocbpy

if sys.version_info.major == 2:
    import warnings
    warnings.simplefilter('default')

# AGB: The TypeError exception below is necessary due to a bug in
# ssj_auroral_boundary that was introduced by a change in matplotlib
# behaviour.  Once this bug is fixed (issue #11), it will be removed
err = ''.join(['unable to load the DMSP SSJ module; ssj_auroral_boundary ',
               'is available at: ',
               'https://github.com/lkilcommons/ssj_auroral_boundary'])
try:
    from spacepy import pycdf
    import aacgmv2
    import ssj_auroral_boundary as ssj
except (ImportError, TypeError) as ierr:
    raise ImportError("{:s}\n{:}".format(err, ierr))


def fetch_ssj_files(stime, etime, out_dir=None, sat_nums=None):
    """ Download DMSP SSJ files and place them in a specified directory

    Parameters
    ----------
    stime : (dt.datetime)
        Start time
    etime : (dt.datetime)
        End time
    out_dir : (str/NoneType)
        Output directory or None to download to ocbpy boundary directory
        (default=None)
    sat_nums : (list/NoneType)
        Satellite numbers or None for all satellites (default=None)

    Returns
    -------
    out_files : list
        List of filenames corresponding to downloaded files

    Notes
    -----
    If a file already exists, the routine will add the file to the output list
    without downloading it again

    """

    # Get and test the output directory
    if out_dir is None:
        out_dir = ocbpy.boundaries.files.get_boundary_directory()

    if not os.path.isdir(out_dir):
        raise ValueError("can't find the output directory")

    # SSJ5 was carried on F16 onwards. F19 was short lived, F20 was not
    # launched. Ref: https://space.skyrocket.de/doc_sdat/dmsp-5d3.htm
    sat_launch = {16: dt.datetime(2003, 10, 18),
                  17: dt.datetime(2006, 11, 4),
                  18: dt.datetime(2009, 10, 18)}
    sat_reentry = {16: dt.datetime(3000, 1, 1),
                   17: dt.datetime(3000, 1, 1),
                   18: dt.datetime(3000, 1, 1)}

    # Ensure the input parameters are appropriate
    if sat_nums is None:
        sat_nums = list(sat_launch.keys())

    if not np.all([snum in list(sat_launch.keys()) for snum in sat_nums]):
        raise ValueError("".join(["unknown satellite ID in ",
                                  "{:} use {:}".format(sat_nums,
                                                       sat_launch.keys())]))

    # Initialize the output
    out_files = list()

    # Cycle through the satellite IDs, downloading each day's file
    for snum in sat_nums:
        ctime = stime if stime >= sat_launch[snum] else sat_launch[snum]
        ltime = etime if etime <= sat_reentry[snum] else sat_reentry[snum]

        while ctime < ltime:
            # Construct the remote and local filenames
            remote, fname = ssj.files.cdf_url_and_filename(snum, ctime.year,
                                                           ctime.month,
                                                           ctime.day)
            local = os.path.join(out_dir, fname)

            # Download the remote file if it doesn't exist locally
            if os.path.isfile(local):
                out_files.append(local)
            else:
                try:
                    ssj.files.download_cdf_from_noaa(remote, local)
                    out_files.append(local)
                except RuntimeError as err:
                    ocbpy.logger.info(err)

            # Cycle by one day
            ctime += dt.timedelta(days=1)

    # Return list of available files for these satellites and times
    return out_files


def create_ssj_boundary_files(cdf_files, out_dir=None,
                              out_cols=['glat', 'glon'],
                              make_plots=False, plot_dir=None):
    """ Create SSJ boundary files for a list of DMSP SSJ daily satellite files

    Parameters
    ----------
    cdf_files : (array-like)
        List of daily satellite files
    out_dir : (str/NoneType)
        Output directory for the boundary files or None to use the ocbpy
        boundary directory (default=None)
    out_cols : list
        List of output columns.  Permitted are CDF file variable names or any
        of mlat, mlt, glat, glon, diff_flux (default=['glat', 'glon'])
    make_plots : bool
        Make plots for the boundary passes (default=False)
    plot_dir : string or NoneType
        If plots are made, this is their output directory.  If None, will be
        set to the same value as out_dir.  (default=None)

    Returns
    -------
    out_files : list
        List of output .csv boundary files

    """
    # Test the directory inputs
    if out_dir is None:
        out_dir = ocbpy.boundaries.files.get_boundary_directory()

    if not os.path.isdir(out_dir):
        raise ValueError("unknown output directory: {:}".format(out_dir))

    if plot_dir is None:
        plot_dir = out_dir

    if make_plots and not os.path.isdir(plot_dir):
        raise ValueError("unknown plot directory: {:}".format(plot_dir))

    # Error catch for input being a filename
    cdf_files = np.asarray(cdf_files)
    if len(cdf_files.shape) == 0:
        cdf_files = np.asarray([cdf_files])

    # Geographic lat and lon are currently provided through the CDF column name
    if 'glat' in out_cols:
        out_cols[out_cols.index('glat')] = 'SC_GEOCENTRIC_LAT'
    if 'glon' in out_cols:
        out_cols[out_cols.index('glon')] = 'SC_GEOCENTRIC_LON'

    # Cycle through all the CDF files, creating the desired CSV files
    out_files = list()
    for cdffn in np.asarray(cdf_files):
        if os.path.isfile(cdffn):
            try:
                with np.errstate(invalid='ignore', divide='ignore',
                                 over='ignore', under='ignore'):
                    absd = ssj.absatday.absatday(cdffn, csvdir=out_dir,
                                                 imgdir=plot_dir,
                                                 make_plot=make_plots,
                                                 csvvars=out_cols)
                    out_files.append(absd.csv.csvfn)
            except pycdf.CDFError as err:
                ocbpy.logger.warning("{:}".format(err))
            except Warning as war:
                ocbpy.logger.warning("{:}".format(war))
        else:
            ocbpy.logger.warning("bad input file {:}".format(cdffn))

    return out_files


def format_ssj_boundary_files(csv_files, ref_alt=830.0,
                              method='GEOCENTRIC|ALLOWTRACE'):
    """ Create SSJ boundary files for a list of DMSP SSJ daily satellite files

    Parameters
    ----------
    csv_files : (list)
        List of SSJ CSV boundary files with directory structure
    ref_alt : (float)
        Reference altitude for boundary locations in km (default=830.0)
    method : (str)
        AACGMV2 method, may use 'TRACE', 'ALLOWTRACE', 'BADIDEA', 'GEOCENTRIC'
        (default='GEOCENTRIC|ALLOWTRACE')

    Returns
    -------
    bound_files : (list)
        List of successfully updated .csv boundary files

    Notes
    -----
    Output format is 'sc date time r x y fom x_1 y_1 x_2 y_2'
    where:

    sc    = Spacecraft number
    date  = YYYY-MM-DD
    time  = HH:MM:SS of midpoint between the two measurements for this pass
    r     = Half the distance between the two pass boundaries
    x     = Distance between the midpoint of the two pass boundaries
            and the AACGMV2 pole in degrees along the dusk-dawn meridian
    y     = distance between the midpoint of the two pass boundaries and the
            AACGMV2 pole in degrees along the midnight-noon meridian
    fom   = FOM for the boundaries found along this pass
    x_1   = x coordinate of the first boundary
    y_1   = y coordinate of the first boundary
    x_2   = x coordinate of the second boundary
    y_2   = y coordinate of the second boundary

    Because two points are not enough to define the OCB or EAB across all local
    times, a circle that intersects the two boundary pass points is defined and
    the boundary location saved.  The DMSP SSJ boundary correction function
    will use this information to only return values within a small distance of
    the boundary locations.

    Separate files are created for each boundary and hemisphere, dates and
    spacecraft are combined.

    """

    # Error catch for input being a filename
    csv_files = np.asarray(csv_files)
    if len(csv_files.shape) == 0:
        csv_files = np.asarray([csv_files])

    # Remove any bad files
    good_files = list()
    for i, infile in enumerate(csv_files):
        if not os.path.isfile(infile):
            ocbpy.logger.warning("bad input file: {:}".format(infile))
        else:
            good_files.append(i)
    csv_files = csv_files[good_files]

    if len(csv_files) == 0:
        raise ValueError("empty list of input CSV files")

    # Set the hemisphere suffix and boundary prefix
    hemi_suffix = {1: "north", -1: "south"}
    bound_prefix = {'PO': '.ocb', 'EQ': '.eab'}

    # Initialize the file lists
    bad_files = list()

    # Initialize the output header
    out_head = "#sc date time r x y fom x_1 y_1 x_2 y_2\n"

    # Specify the output file information
    outfile_prefix = os.path.commonprefix(list(csv_files)).split('-f')[0]
    filename_sec = os.path.split(
        csv_files[0])[-1].split('dmsp-f')[-1].split('_')
    sdate = filename_sec[3]
    filename_sec = os.path.split(
        csv_files[-1])[-1].split('dmsp-f')[-1].split('_')
    edate = filename_sec[3]

    bound_files = {hh: {bb: "".join([outfile_prefix, "-", filename_sec[1], "_",
                                     hemi_suffix[hh], "_", sdate, "_", edate,
                                     "_", filename_sec[4], bound_prefix[bb]])
                        for bb in bound_prefix.keys()}
                   for hh in hemi_suffix.keys()}
    fpout = {hh: {bb: None for bb in bound_prefix.keys()}
             for hh in hemi_suffix.keys()}

    with open(bound_files[1]['PO'], 'w') as fpout[1]['PO'], \
         open(bound_files[-1]['PO'], 'w') as fpout[-1]['PO'], \
         open(bound_files[1]['EQ'], 'w') as fpout[1]['EQ'], \
         open(bound_files[-1]['EQ'], 'w') as fpout[-1]['EQ']:
        # Output the header in each file
        for hh in hemi_suffix.keys():
            for bb in bound_prefix.keys():
                fpout[hh][bb].write(out_head)

        # Cycle through all the SSJ CSV files, outputing appropriate data into
        # the desired boundary and hemisphere file
        for infile in csv_files:
            # Get spacecraft number and date from filename
            filename_sec = os.path.split(infile)[-1].split(
                'dmsp-f')[-1].split('_')
            sc = int(filename_sec[0])
            file_date = dt.datetime.strptime(filename_sec[3], '%Y%m%d')

            # Get the header line for the data and determine the number of
            # comment lines preceeding the header
            skiprows = 1
            with open(infile, 'r') as fpin:
                head_line = fpin.readline()
                while head_line.find("#") == 0:
                    skiprows += 1
                    head_line = fpin.readline()

            header_list = head_line.split("\n")[0].split(",")

            # Load the file data
            data = np.loadtxt(infile, skiprows=skiprows, delimiter=',')
            if data.shape[1] != len(header_list):
                bad_files.append(infile)
            else:
                # Establish the desired data indices
                time_ind = {bb: [header_list.index('UTSec {:s}{:d}'.format(
                    bb, i)) for i in [1, 2]]
                            for bb in bound_prefix.keys()}
                lat_ind = {bb: [header_list.index(
                    'SC_GEOCENTRIC_LAT {:s}{:d}'.format(bb, i))
                                for i in [1, 2]] for bb in bound_prefix.keys()}
                lon_ind = {bb: [header_list.index(
                    'SC_GEOCENTRIC_LON {:s}{:d}'.format(bb, i))
                                for i in [1, 2]] for bb in bound_prefix.keys()}

                # Calculate the midpoint seconds of day
                mid_utsec = {bb: 0.5 * (data[:, time_ind[bb][1]]
                                        + data[:, time_ind[bb][0]])
                             for bb in time_ind.keys()}

                # Select the hemisphere and FOM
                hemi = data[:, header_list.index('hemisphere')]
                fom = data[:, header_list.index('FOM')]

                # Cycle through each line of data, calculating the
                # necessary information
                for iline, data_line in enumerate(data):
                    hh = hemi[iline]

                    # Get the boundary locations using the midpoint time
                    # (sufficiently accurate at current sec. var.) for each
                    # boundary
                    for bb in bound_prefix.keys():
                        mid_time = file_date + dt.timedelta(
                            seconds=mid_utsec[bb][iline])
                        mloc = aacgmv2.get_aacgm_coord_arr(
                            data_line[lat_ind[bb]], data_line[lon_ind[bb]],
                            ref_alt, mid_time, method=method)

                        # Determine the circle radius in degrees
                        rad = 0.5 * abs(mloc[0][0]-mloc[0][1])

                        # Get the X-Y coordinates of each pass where X is
                        # positive towards dawn and Y is positive towards noon
                        theta = np.radians(mloc[2] * 15.0 - 90.0)
                        x = (90.0 - abs(mloc[0])) * np.cos(theta)
                        y = (90.0 - abs(mloc[0])) * np.sin(theta)

                        # The midpoint is the center of this circle
                        mid_x = 0.5 * sum(x)
                        mid_y = 0.5 * sum(y)

                        # Prepare the output line, which has the format:
                        # sc bound hemi date time r x y fom x_1 y_1
                        # x_2 y_2
                        out_line = " ".join(
                            ["{:d}".format(sc),
                             mid_time.strftime('%Y-%m-%d %H:%M:%S'),
                             " ".join(["{:.3f}".format(val)
                                       for val in [rad, mid_x, mid_y,
                                                   fom[iline], x[0], y[0],
                                                   x[1], y[1]]]), "\n"])

                        # Write the output line to the file
                        fpout[hh][bb].write(out_line)

    # If some input files were not processed, inform the user
    if len(bad_files) > 0:
        ocbpy.logger.warning("unable to format {:d} input files: {:}".format(
            len(bad_files), bad_files))

    # Recast the output file dictionary as a flat list
    bound_files = np.array([[fname for fname in ff.values()]
                            for ff in bound_files.values()])

    return list(bound_files.flatten())


def fetch_format_ssj_boundary_files(stime, etime, out_dir=None, rm_temp=True,
                                    ref_alt=830.0,
                                    method='GEOCENTRIC|ALLOWTRACE'):
    """ Download DMSP SSJ data and create boundary files for each hemisphere

    Parameters
    ----------
    stime : (dt.datetime)
        Start time
    etime : (dt.datetime)
        End time
    out_dir : (str/NoneType)
        Output directory or None to download to ocbpy boundary directory
        (default=None)
    rm_temp : (boolean)
        Remove all files that are not the final boundary files (default=True)
    ref_alt : (float)
        Reference altitude for boundary locations in km (default=830.0)
    method : (str)
        AACGMV2 method, may use 'TRACE', 'ALLOWTRACE', 'BADIDEA', 'GEOCENTRIC'
        (default='GEOCENTRIC|ALLOWTRACE')

    Returns
    -------
    bound_files : (list)
        List of the boundary file names

    """

    # Fetch the DMSP SSJ files for all available DMSP satellites
    dmsp_files = fetch_ssj_files(stime, etime, out_dir=out_dir)

    if len(dmsp_files) == 0:
        raise ValueError("".join(["unable to download any DMSP SSJ files ",
                                  "between {:} and {:}".format(stime, etime)]))

    # Create CSV files with geographic coordinates for the boundary locations
    csv_files = create_ssj_boundary_files(dmsp_files, out_dir=out_dir)

    # Test to see if there are any DMSP processed files
    if len(csv_files) == 0:
        raise ValueError("".join(["unable to process the downloaded SSJ files",
                                  " {:}".format(dmsp_files)]))

    # Remove the DMSP files, as their data has been processed
    if rm_temp:
        for tmp_file in dmsp_files:
            os.remove(tmp_file)

    # Create the boundary files
    bound_files = format_ssj_boundary_files(csv_files, ref_alt=ref_alt,
                                            method=method)

    # Remove the CSV files, as their data has been processed
    if rm_temp:
        for tmp_file in csv_files:
            os.remove(tmp_file)

    return bound_files
