#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB & GC
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""
Open-Closed field line Boundary (OCB) magnetic gridding

Functions
---------
match_data_ocb      Matches data and OCB records
normal_evar         Normalise a variable proportional to the electric field
normal_curl_evar    Normalise a variable proportional to the curl of the
                    electric field

Classes
-------
OCBoundary    OCB data for different times
VectorData    Vector data point

Modules
-------
boundaries     Boundary file utilities
instruments    Instrument-specific OCB gridding functions
ocb_time       Time manipulation routines
ocb_scaling    Scaling functions for OCB gridded data
ocb_correction Boundary correction utilities

"""

from __future__ import absolute_import, unicode_literals

import logging

from ocbpy import ocboundary, ocb_scaling, ocb_time, ocb_correction
from ocbpy.ocboundary import OCBoundary, match_data_ocb
from ocbpy import instruments, boundaries

# Define the global variables
__version__ = str('0.2.0')

# Define a logger object to allow easier log handling
logging.raiseExceptions = False
logger = logging.getLogger('ocbpy_logger')
