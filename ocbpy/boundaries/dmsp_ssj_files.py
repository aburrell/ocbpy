#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DOI: 10.5281/zenodo.1179230
# Full license can be found in License.md
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# -----------------------------------------------------------------------------
"""Download and format DMSP SSJ boundary files.

References
----------
.. [2] Angeline Burrell, Christer van der Meeren, & Karl M. Laundal. (2020).
   aburrell/aacgmv2 (All Versions). Zenodo. doi:10.5281/zenodo.1212694.

.. [5] Kilcommons, L.M., et al. (2017), A new DMSP magnetometer and auroral
   boundary data set and estimates of field-aligned currents in dynamic auroral
   boundary coordinates, J. Geophys. Res.: Space Phys., 122, pp 9068-9079,
   doi:10.1002/2016ja023342.

.. [7] Kilcommons, L., Redmon, R., & Knipp, D. (2019). Defense Meteorology
   Satellite Program (DMSP) Electron Precipitation (SSJ) Auroral Boundaries,
   2010-2014 (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3373812

"""

import datetime as dt
from io import StringIO
import numpy as np
import os
import sys
import zipfile

import aacgmv2

import ocbpy

try:
    import zenodo_get
except ImportError as ierr:
    raise ImportError(''.join(['unable to load `zenodo_get` module; avalable ',
                               'from PyPi.\n', str(ierr)]))


def fetch_ssj_boundary_files(stime=None, etime=None, out_dir=None,
                             sat_nums=None, doi='10.5281/zenodo.3373811',
                             rm_temp=True):
    """Download DMSP SSJ files and place them in a specified directory.

    Parameters
    ----------
    stime : dt.datetime or NoneType
        Start time or None to get all boundaries (default=None)
    etime : dt.datetime or NoneType
        End time or None to get all boundaries (default=None)
    out_dir : str or NoneType
        Output directory or None to download to ocbpy boundary directory
        (default=None)
    sat_nums : list or NoneType
        Satellite numbers or None for all satellites (default=None)
    doi : str
        DOI for the DMSP SSJ boundary file Zenodo archive [7]_
        (default='10.5281/zenodo.3373811')
    rm_temp : bool
        Remove all files that are not the final boundary files (default=True)

    Returns
    -------
    out_files : list
        List of filenames corresponding to downloaded boundary files

    Raises
    ------
    ValueError
        If an unknown satellite ID is provided.
    IOError
        If unable to donwload the target archive and identify the zip file
    ImportError
        If called and zenodo_get is not available

    Notes
    -----
    If a file already exists, the routine will add the file to the output list
    without downloading it again.

    """
    # Test the requested satellite outputs. SSJ5 was carried on F16 onwards.
    # F19 was short lived, F20 was not launched. Ref:
    # https://space.skyrocket.de/doc_sdat/dmsp-5d3.htm
    known_sats = [16, 17, 18]

    # Ensure the input parameters are appropriate
    get_all = False
    if sat_nums is None:
        sat_nums = known_sats

        if stime is None and etime is None:
            get_all = True

    if not np.all([snum in known_sats for snum in sat_nums]):
        raise ValueError("".join(["unknown satellite ID in ",
                                  "{:} use {:}".format(sat_nums, known_sats)]))

    # Get and test the output directory
    if out_dir is None:
        out_dir = ocbpy.boundaries.files.get_boundary_directory()

    if not os.path.isdir(out_dir):
        raise ValueError("can't find the output directory")

    # Download the zenodo archive, capturing the output
    zenodo_io = StringIO()
    sys.stdout = zenodo_io
    sys.stderr = zenodo_io

    zenodo_checksum = os.path.join(out_dir, 'md5sums.txt')
    zenodo_get.zenodo_get([doi, '-o', out_dir])

    # Parse the output and retrieve files from the zip archive
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    zen_msg = zenodo_io.getvalue()
    zen_split = zen_msg.split()

    if zen_msg.find('Checksum is correct') < 0 and zen_msg.find(
            'already downloaded correctly') < 0:
        raise IOError('Bad checksum, see: {:s}\n{:s}'.format(zenodo_checksum,
                                                             zen_msg))

    # Remove the checksum file if the download problem wasn't found there
    os.remove(zenodo_checksum)

    # Get the archive name from the output
    try:
        link_ind = zen_split.index('Link:') + 1

        # If the archive is already available, message may differ
        zip_name = os.path.split(zen_split[link_ind])
        while zip_name[-1].find(".zip") <= 0:
            zip_name = os.path.split(zip_name[0])

        # Set the archive name
        archive_name = os.path.join(out_dir, zip_name[-1])
    except (ValueError, IndexError):
        raise IOError('unable to identify zenodo archive: {:}'.format(zen_msg))

    if not os.path.isfile(archive_name):
        raise IOError('error downloading archive to output dir: {:}'.format(
            archive_name))

    # Access the zip archive
    with zipfile.ZipFile(archive_name, 'r') as zref:
        namelist = zref.namelist()

        if get_all:
            # If all boundaries are desired, extract all
            zref.extractall(out_dir)
            out_files = [os.path.join(out_dir, fname) for fname in namelist]
        else:
            # Otherwise, extract files for the desired satellites and times
            out_files = list()

            for fname in namelist:
                if evaluate_dmsp_boundary_file(fname, stime, etime, sat_nums):
                    out_files.append(zref.extract(fname, path=out_dir))

    # Remove the zip archive, if desired
    if rm_temp:
        os.remove(archive_name)

    # Return list of available files for these satellites and times
    return out_files


def evaluate_dmsp_boundary_file(fname, stime, etime, sat_nums):
    """Evaluate a boundary file name for time range and satellite ID.

    Parameters
    ----------
    fname : str
        Filename with the format:
        dmsp-fII_ssj_precipitating-electrons-ions_YYYYMMDD_vXXX_boundaries.csv
        where II is the satellite ID, YYYY is the year, MM is the month, DD
        is the day of month, and XXX is the version number.
    stime : dt.datetime or NoneType
        Start time or None to get all boundaries
    etime : dt.datetime or NoneType
        End time or None to get all boundaries
    sat_nums : list
        Desired satellite ID numbers

    Returns
    -------
    good_file : bool
        True if the satellite has the desired ID and time, False otherwise

    """
    # Initialize the output
    good_file = False

    # Ensure the time bounds will result in the desired range with an evaluation
    if stime is None:
        stime = dt.datetime(1900, 1, 1)  # A time before DMSP launched

    if etime is None:
        etime = dt.datetime.utcnow()  # A time beyond data uploads

    # Get the file satelite ID
    sat_id = int(fname.split('dmsp-f')[-1].split('_ssj')[0])

    if sat_id in sat_nums:
        # Test the file time
        if stime is None and etime is None:
            good_file = True
        else:
            # Get the file time
            ftime = dt.datetime.strptime(fname.split('_')[3], '%Y%m%d')

            if (stime is None and ftime < etime) or (
                    etime is None and ftime >= stime) or (
                        ftime >= stime and ftime < etime):
                good_file = True

    return good_file


def format_ssj_boundary_files(csv_files, ref_alt=830.0,
                              method='GEOCENTRIC|ALLOWTRACE'):
    """Create SSJ boundary files for a list of DMSP SSJ daily satellite files.

    Parameters
    ----------
    csv_files : list
        List of SSJ CSV boundary files with directory structure
    ref_alt : float
        Reference altitude for boundary locations in km (default=830.0)
    method : str
        AACGMV2 method, may use 'TRACE', 'ALLOWTRACE', 'BADIDEA', 'GEOCENTRIC'
        [2]_ (default='GEOCENTRIC|ALLOWTRACE')

    Returns
    -------
    bound_files : list
        List of successfully updated .csv boundary files

    Notes
    -----
    Output format is 'sc date time r x y fom x_1 y_1 x_2 y_2'
    where:

    ===== ==================================================================
     sc    Spacecraft number
     date  YYYY-MM-DD
     time  HH:MM:SS of midpoint between the two measurements for this pass
     r     Half the distance between the two pass boundaries
     x     Distance between the midpoint of the two pass boundaries
           and the AACGMV2 pole in degrees along the dusk-dawn meridian
     y     distance between the midpoint of the two pass boundaries and the
           AACGMV2 pole in degrees along the midnight-noon meridian
     fom   FOM for the boundaries found along this pass
     x_1   x coordinate of the first boundary
     y_1   y coordinate of the first boundary
     x_2   x coordinate of the second boundary
     y_2   y coordinate of the second boundary
    ===== ==================================================================

    Because two points are not enough to define the OCB or EAB across all local
    times, a circle that intersects the two boundary pass points is defined and
    the boundary location saved.  The DMSP SSJ boundary correction function
    will use this information to only return values within a small distance of
    the boundary locations [5]_, [7]_.

    Separate files are created for each boundary and hemisphere, dates and
    spacecraft are combined.

    See Also
    --------
    aacgmv2

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
            if len(data.shape) != 2 or data.shape[1] != len(header_list):
                bad_files.append(infile)
            else:
                # Establish the desired data indices
                time_ind = {bb: [header_list.index('UTSec {:s}{:d}'.format(
                    bb, i)) for i in [1, 2]] for bb in bound_prefix.keys()}
                lat_ind = {bb:
                           [header_list.index(
                               'SC_GEOCENTRIC_LAT {:s}{:d}'.format(bb, i))
                            for i in [1, 2]] for bb in bound_prefix.keys()}
                lon_ind = {bb:
                           [header_list.index(
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

                        # Get the X-Y coordinates of each pass where X is
                        # positive towards dawn and Y is positive towards noon
                        theta = ocbpy.ocb_time.hr2rad(mloc[2] - 6.0)
                        x = (90.0 - abs(mloc[0])) * np.cos(theta)
                        y = (90.0 - abs(mloc[0])) * np.sin(theta)

                        # Get the distance between the Cartesian points
                        rad = np.sqrt((x[0] - x[1])**2 + (y[0] - y[1])**2) / 2.0

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
    """Download DMSP SSJ data and create boundary files for each hemisphere.

    Parameters
    ----------
    stime : dt.datetime or NoneType
        Start time or NoneType to fetch all
    etime : dt.datetime
        End time or NoneType to fetch all
    out_dir : str or NoneType
        Output directory or None to download to ocbpy boundary directory
        (default=None)
    rm_temp : bool
        Remove all files that are not the final boundary files (default=True)
    ref_alt : float
        Reference altitude for boundary locations in km (default=830.0)
    method : str
        AACGMV2 method, may use 'TRACE', 'ALLOWTRACE', 'BADIDEA', 'GEOCENTRIC'
        [2]_ (default='GEOCENTRIC|ALLOWTRACE')

    Returns
    -------
    bound_files : list
        List of the boundary file names

    Raises
    ------
    ValueError
        If the DMSP SSJ files could not be downloaded

    See Also
    --------
    aacgmv2, requests.exceptions.ProxyError

    """
    # Fetch the DMSP SSJ boundary files from the Zenodo archive
    csv_files = fetch_ssj_boundary_files(stime, etime, out_dir=out_dir,
                                         rm_temp=rm_temp)

    # Test to see if there are any DMSP processed files
    if len(csv_files) == 0:
        raise ValueError("".join(["unable to download the SSJ files from ",
                                  "{:} to {:}".format(stime, etime)]))

    # Create the boundary files
    bound_files = format_ssj_boundary_files(csv_files, ref_alt=ref_alt,
                                            method=method)

    # Remove the CSV files, as their data has been processed
    if rm_temp:
        for tmp_file in csv_files:
            os.remove(tmp_file)

    return bound_files
