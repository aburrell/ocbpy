#!/usr/bin/env python
''' General loading routines for data files

Routines
-------------------------------------------------------------------------------
test_file          test to see whether file exists and is small enough to load
load_ascii_data    load time-sorted ascii data file
-------------------------------------------------------------------------------
'''
import numpy as np
import logging
import datetime as dt

def test_file(filename):
    '''Test to ensure the file is small enough to read in.  Python can only
    allocate 2GB of data without crashing

    Parameters
    ------------
    filename : (str)
        Filename to test

    Returns
    ---------
    good_flag : (bool)
        True if good, bad if false
    '''
    from os import path

    if not path.isfile(filename):
        logging.warn("name provided is not a file")
        return False
    
    fsize = path.getsize(filename)

    if(fsize > 2.0e9):
        logging.warn("File size [{:.2f} GB > 2 GB]".format(fsize*1e-9))
        return False
    elif(fsize == 0):
        logging.warn("empty file [{:s}]".format(filename))
        return False

    return True

def load_ascii_data(filename, hlines, miss=None, fill=np.nan, hsplit=None,
                    inline_comment=None, invalid_raise=False, datetime_cols=[],
                    datetime_fmt=None, int_cols=[], str_cols=[],
                    max_str_length=50, header=list()):
    ''' Load an ascii data file into a dict of numpy array. 

    Parameters
    ------------
    filename : (str)
        data file name
    hlines : (int)
        number of lines in header.  If zero, must include header.
    miss : (str or list)
        Denotes missing value options (default=None)
    fill : (any non-list)
        fill value (default=NaN)
    hsplit : (str, NoneType)
        character seperating data labels in header.  None splits on all
        whitespace characters. (default=None)
    inline_comment : (str or NoneType)
        If there are comments inline, denote the charater that indicates it has
        begun. If there are no comments inline, leave as the default.
        (default=None)
    invalid_raise : (bool)
        Should the routine fail if a row of data with a different number of
        columns is encountered?  If false, these lines will be skipped and
        all other lines will be read in.  (default=False)
    datetime_cols : (list of ints)
        If there are date strings or values that should be converted to a
        datetime object, list them in order here. Not processed as floats.
        (default=[])
    datetime_fmt : (str or NoneType)
        Format needed to convert the datetime_cols entries into a datetime
        object.  (default=None)
    int_cols : (list of ints)
        Data that should be processed as integers, not floats. (default=[])
    str_cols : (list of ints)
        Data that should be processed as strings, not floats. (default=[])
    max_str_length : (int)
        Maximum allowed string length. (default=50)
    header : (list of str)
        Header string(s) where the last line contains whitespace separated data
        names (default=list())

    Returns
    ----------
    header : (list of strings)
        Contains all specified header lines
    out : (dict of numpy.arrays)
        The dict keys are specified by the header data line, the data
        for each key are stored in the numpy array

    Notes
    -------
    Data is assumed to be float unless otherwise stated.
    '''
    #-----------------------------------------------------------------------
    # Test to ensure the file is small enough to read in.  Python can only
    # allocate 2GB of data.  If you load something larger, python will crash
    if not test_file(filename):
        return header, dict()

    #-------------------------------------------------------------
    # Make sure the max_str_length is long enough to read datetime
    if datetime_fmt is not None and max_str_length < len(datetime_fmt):
        max_str_length = len(datetime_fmt)
        if datetime_fmt.find("%Y") >= 0 or datetime_fmt.find("%j") >= 0:
            max_str_length += 2
        if(datetime_fmt.find("%a") >= 0 or datetime_fmt.find("%b") >= 0 or
           datetime_fmt.find("%Z") >= 0):
            max_str_length += 1
        if(datetime_fmt.find("%B") >= 0 or datetime_fmt.find("%X") >= 0 or
           datetime_fmt.find("%x") >= 0):
            max_str_length += 10
        if datetime_fmt.find("%f") >= 0:
            max_str_length += 4
        if datetime_fmt.find("%z") >= 0:
            max_str_length += 3
        if datetime_fmt.find("%c") >= 0:
            max_str_length += 20

    #----------------------------------------------
    # Open the datafile and read the header rows
    f = open(filename, "r")

    if not f:
        logging.error("unable to open input file [{:s}]".format(filename))
        return header, dict()

    for h in range(hlines):
        header.append(f.readline())

    f.close()
    #---------------------------------------------------------------------
    # Create the output dictionary keylist
    if len(header) == 0:
        logging.error("unable to find header of [{:d}] lines".format(hlines))
        return header, dict()
    
    if inline_comment is None:
        keyheader = header[-1]
    else:
        keyheader = header[-1].split(inline_comment)[0]

    keyheader = keyheader.replace("#", "")
    keylist = keyheader.split(hsplit)
    nhead = len(keylist)
    out = {k:list() for k in keylist}

    #---------------------------------------------------------------------
    # Build the dtype list
    ldtype = [float for i in range(nhead)]

    for icol in int_cols:
        ldtype[icol] = int

    for icol in str_cols:
        ldtype[icol] = '|S{:d}'.format(max_str_length)
    
    #---------------------------------------------------------------------
    # Build and add the datetime objects to the output dictionary
    dt_keys = ['datetime', 'DATETIME', 'DT', 'dt']
    if len(datetime_cols) > 0 and datetime_fmt is not None:
        idt = 0
        while out.has_key(dt_keys[idt]): idt += 1

        if idt < len(dt_keys):
            keylist.append(dt_keys[idt])
            out[dt_keys[idt]] = list()

        # Change the datetime column input from float to string
        for icol in datetime_cols:
            ldtype[icol] = '|S{:d}'.format(max_str_length)
    else:
        idt = len(dt_keys)

    #-------------------------------------------
    # Open the datafile and read the data rows
    try:
        temp = np.genfromtxt(filename, skip_header=hlines, missing_values=miss,
                             filling_values=fill, comments=inline_comment,
                             invalid_raise=False, dtype=ldtype)
    except:
        logging.error("unable to read data in file [{:s}]".format(filename))
        return header, out

    if len(temp) > 0:
        noff = 0
        # When dtype is specified, output comes as a np.array of np.void objects
        for line in temp:
            if len(line) == nhead:
                for num,name in enumerate(keylist):
                    if len(name) > 0:
                        if idt < len(dt_keys) and name == dt_keys[idt]:
                            # Build a string to cast datetime
                            dtstring = line[datetime_cols[0]]
                            for icol in range(len(datetime_cols)-1):
                                i = datetime_cols[icol+1]
                                dtstring = "{:s} {:s}".format(dtstring, line[i])
                                
                            # Convert the string into a datetime object
                            try:
                                ftime = dt.datetime.strptime(dtstring,
                                                             datetime_fmt)
                            except ValueError as v:
                                if(len(v.args) > 0 and \
                            v.args[0].startswith('unconverted data remains: ')):
                                    vsplit = v.args[0].split(" ")
                                    ftime = dt.datetime.strptime( \
                                    dtstring[:-(len(vsplit[-1]))], datetime_fmt)
                                else:
                                    raise v

                            # Save the output data
                            out[dt_keys[idt]].append(ftime)
                        else:
                            out[name].append(line[num-noff])
                    else:
                        noff += 1
            else:
                estr = "unknown genfromtxt output for [{:s}]".format(filename)
                logging.error(estr)
                return header, dict()

    del temp
    # Cast all lists and numpy arrays
    for k in out.keys():
        try:
            out[k] = np.array(out[k], dtype=type(out[k][0]))
        except:
            pass

    return header, out
