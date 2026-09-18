#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DOI: 10.5281/zenodo.1179230
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Download and format AMPERE R1/R2 boundary files.

References
----------
.. [12] Milan, Stephen (2023). AMPERE R1/R2 FAC radii v2. University of
        Leicester. Dataset. https://doi.org/10.25392/leicester.data.22241338

"""

from io import StringIO
import logging
import numpy as np
import os

import ocbpy

try:
    import pyfigshare
except ImportError as ierr:
    raise ImportError(''.join(['unable to load `pyfigshare` module; avalable ',
                               'from PyPi.\n', str(ierr)]))


def fetch_ampere_boundary_files(out_dir=None, figshare_id=22241338,
                                file_id=None):
    """Download AMPERE R1/R2 files and place them in a specified directory.

    Parameters
    ----------
    out_dir : str or NoneType
        Output directory or None to download to ocbpy boundary directory
        (default=None)
    figshare_id : int
        Figshare identification number for the AMPERE R1/R2 boundary file
        archive [12]_ (default=22241338)
    file_id : int, str, list, or NoneType
        If not None, only download the file(s) with these figshare file id(s).
        Accepts a single id, a list/tuple of ids, or a comma-separated
        string such as ``"123,456"`` (default=None)

    Returns
    -------
    out_files : array-like
        List of filenames corresponding to downloaded boundary files that have
        been sorted

    Raises
    ------
    ValueError
        If an unknown output directory is provided.
    IOError
        If unable to download all the targeted files
    ImportError
        If called and pyfigshare is not available
    HTTPError
        If an unknown figshare ID is provided

    Notes
    -----
    If a file already exists, the routine will add the file to the output list
    without downloading it again.

    """
    # Get and test the output directory
    if out_dir is None:
        out_dir = ocbpy.boundaries.files.get_boundary_directory()

    if not os.path.isdir(out_dir):
        raise ValueError("can't find the output directory")

    # Set up a pipe for the logging output, pyfigshare uses loguru
    log_out = StringIO()
    pyfigshare.figshare.logger.add(logging.StreamHandler(log_out))

    # Download the figshare archive
    pyfigshare.download(figshare_id, outdir=out_dir, file_id=file_id)

    # Get the list of downloaded files from the logging info
    out_lines = log_out.getvalue()
    out_files = [os.path.join(out_dir, out_line.split(' - ')[1])
                 for out_line in out_lines.split('\n')
                 if out_line.find('download_article') > 0]

    # Verify the files exist
    check_files = np.array([os.path.isfile(out_file) for out_file in out_files])
    if not check_files.all():
        raise IOError(
            "{:d}/{:d} files not downloaded, check log output:\n{:s}".format(
                np.sum(~check_files), check_files.shape[0], out_lines))

    # Return array of available files for these satellites and times
    return np.sort(out_files)


def format_ampere_boundary_files(figshare_files, out_dir=None, ocb_bnd='rb',
                                 eab_bnd='re'):
    """Create a AMPERE OCB/EAB boundary file for a list of yearly R1/R2 files.

    Parameters
    ----------
    figshare_files : list
        List of AMPERE R1/R2 boundary files with directory structure that have
        been sorted by time
    out_dir : str or NoneType
        Output directory for formated AMPERE boundary files or None to use
        default ocbpy boundary directory (default=None)
    ocb_bnd : str
        Boundary to use for the OCB (default='rb')
    eab_bnd : str
        Boundary to use for the EAB (default='re')

    Returns
    -------
    bound_files : list
        List of successfully updated boundary files

    Raises
    ------
    ValueError
        If an unknown OCB or EAB proxy boundary is supplied, no good files
        are provided, or the output directory does not exist
    KeyError
        If the hemisphere is not the last string separated by underscores in the
        filename

    Notes
    -----
    Output format is 'date time r x0 y0 fom'
    where:

    ===== ==================================================================
     date  YYYYMMDD
     time  HH:MM of boundary fits
     r     Radius of the circle fit
     x0    x-coordinate of circle centre measured along the dawn-dust and
           noon-midnight meridean of the MLT/MLat coordinate system with
           positive values towards dawn and noon.
     y0    x-coordinate of circle centre measured along the dawn-dust and
           noon-midnight meridean of the MLT/MLat coordinate system with
           positive values towards dawn and noon.
     fom   Peak-to-peak value of the bipolar signature (micro-Amps / m^2)
    ===== ==================================================================

    Separate files are created for each boundary and hemisphere, dates are
    combined.

    """
    # Get and test the output directory
    if out_dir is None:
        out_dir = ocbpy.boundaries.files.get_boundary_directory()

    if not os.path.isdir(out_dir):
        raise ValueError("can't find the output directory")

    # Ensure the desired boundaries exist and set indices
    ocb_bnds = {'r1': 2, 'rb': 3}
    eab_bnds = {'r2': 4, 're': 5}
    date_ind = 0
    time_ind = 1
    x0_ind = 6
    y0_ind = 7
    fom_ind = 8

    if ocb_bnd not in ocb_bnds.keys():
        raise ValueError('unknown OCB proxy [{:}], expects one of: {:}'.format(
            ocb_bnd, repr(ocb_bnds.keys())))

    if eab_bnd not in eab_bnds.keys():
        raise ValueError('unknown EAB proxy [{:}], expects one of: {:}'.format(
            eab_bnd, repr(eab_bnds.keys())))

    # Error catch for input being a filename
    figshare_files = np.asarray(figshare_files)
    if len(figshare_files.shape) == 0:
        figshare_files = np.asarray([figshare_files])

    # Remove any bad files and identify hemispheres
    good_files = {'north': list(), 'south': list()}
    for i, infile in enumerate(figshare_files):
        if not os.path.isfile(infile):
            ocbpy.logger.warning("bad input file: {:}".format(infile))
        else:
            # Append to the good list for each hemisphere
            good_files[infile.split("_")[-1][:5]].append(infile)

    if len(good_files['north']) == 0 and len(good_files['south']) == 0:
        raise ValueError("empty list of input files")

    # Set the hemisphere suffix and boundary prefix
    hemi_prefix = {1: "north", -1: "south"}
    bound_suffix = {ocb_bnd: '.ocb', eab_bnd: '.eab'}

    # Initialize the file lists
    bad_files = list()

    # Specify the output file information
    outfile_prefix = os.path.join(out_dir, "amp_")

    bound_files = {hh: {bb: "".join([outfile_prefix, hemi_prefix[hh], "_radii",
                                     bound_suffix[bb]])
                        for bb in bound_suffix.keys()}
                   for hh in hemi_prefix.keys()
                   if len(good_files[hemi_prefix[hh]]) > 0}

    for hh in bound_files.keys():
        # Initalize the output file pointers and open files for this hemisphere
        fpout = {bb: None for bb in bound_files[hh].keys()}

        with open(bound_files[hh][ocb_bnd], 'w') as fpout[ocb_bnd], \
             open(bound_files[hh][eab_bnd], 'w') as fpout[eab_bnd]:
            # Cycle through all the figshare files, outputing appropriate data
            # into the desired boundary and hemisphere file
            for infile in good_files[hemi_prefix[hh]]:
                # Determine the number of comment lines
                skiprows = 0
                with open(infile, 'r') as fpin:
                    head_line = fpin.readline()
                    while head_line.find("%") == 0:
                        skiprows += 1
                        head_line = fpin.readline()

                # Load the file data
                data = np.loadtxt(infile, skiprows=skiprows, dtype='str')
                if len(data.shape) != 2 or data.shape[1] != 9:
                    bad_files.append(infile)
                else:
                    # Select the desired data
                    ocb_dat = data[:, [date_ind, time_ind, ocb_bnds[ocb_bnd],
                                       x0_ind, y0_ind, fom_ind]]
                    eab_dat = data[:, [date_ind, time_ind, ocb_bnds[ocb_bnd],
                                       x0_ind, y0_ind, fom_ind]]

                    # Format the desired data
                    ocb_line = "\n".join([" ".join(dat) for dat in ocb_dat])
                    eab_line = "\n".join([" ".join(dat) for dat in eab_dat])

                    # Writing the output to the correct file
                    fpout[ocb_bnd].write("{:s}\n".format(ocb_line))
                    fpout[eab_bnd].write("{:s}\n".format(eab_line))

    # If some input files were not processed, inform the user
    if len(bad_files) > 0:
        ocbpy.logger.warning("unable to format {:d} input files: {:}".format(
            len(bad_files), bad_files))

    # Recast the output file dictionary as a flat list
    bound_files = np.array([[fname for fname in ff.values()]
                            for ff in bound_files.values()])

    return list(bound_files.flatten())


def fetch_format_ampere_boundary_files(out_dir=None, file_id=None, ocb_bnd='rb',
                                       eab_bnd='re', rm_temp=True):
    """Download DMSP SSJ data and create boundary files for each hemisphere.

    Parameters
    ----------
    out_dir : str or NoneType
        Output directory or None to download to ocbpy boundary directory
        (default=None)
    file_id : int, str, list, or NoneType
        If not None, only download the file(s) with these figshare file id(s).
        Accepts a single id, a list/tuple of ids, or a comma-separated
        string such as ``"123,456"`` (default=None)
    ocb_bnd : str
        Boundary to use for the OCB (default='rb')
    eab_bnd : str
        Boundary to use for the EAB (default='re')
    rm_temp : bool
        Remove all files that are not the final boundary files (default=True)

    Returns
    -------
    bound_files : list
        List of the boundary file names

    Raises
    ------
    ValueError
        If the AMPERE R1/R2 files could not be downloaded

    """
    # Fetch the DMSP SSJ boundary files from the Zenodo archive
    figshare_files = fetch_ampere_boundary_files(out_dir=out_dir,
                                                 file_id=file_id)

    # Test to see if there are any DMSP processed files
    if len(figshare_files) == 0:
        raise ValueError("".join(["unable to download the AMPERE files using ",
                                  "{:s} file ID".format("no" if file_id is None
                                                        else repr(file_id))]))

    # Create the boundary files
    bound_files = format_ampere_boundary_files(figshare_files, out_dir=out_dir,
                                               ocb_bnd=ocb_bnd, eab_bnd=eab_bnd)

    # Remove the figshare files, as their data has been processed
    if rm_temp:
        for tmp_file in figshare_files:
            os.remove(tmp_file)

    return bound_files
