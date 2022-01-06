#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB & GC
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Open-Closed field line Boundary (OCB) magnetic gridding

Functions
---------
match_data_ocb      Matches data and OCB records
normal_evar         Normalise a variable proportional to the electric field
normal_curl_evar    Normalise a variable proportional to the curl of the
                    electric field

Classes
-------
DualBoundary  EAB and OCB data for different times
EABoundary    EAB data for different times
OCBoundary    OCB data for different times
VectorData    Vector data point

Modules
-------
boundaries     Boundary file utilities
instruments    Instrument-specific OCB gridding functions
cycle_boundary Boundary class cycling functions
ocb_time       Time manipulation routines
ocb_scaling    Scaling functions for OCB gridded data
ocb_correction Boundary correction utilities

"""

# Define a logger object to allow easier log handling
import logging

logging.raiseExceptions = False
logger = logging.getLogger('ocbpy_logger')

# Import the package modules and top-level classes
from ocbpy import boundaries
from ocbpy import cycle_boundary
from ocbpy import eaboundary
from ocbpy import instruments
from ocbpy import ocboundary
from ocbpy import ocb_correction
from ocbpy import ocb_scaling
from ocbpy import ocb_time

from ocbpy._boundary import DualBoundary
from ocbpy._boundary import EABoundary
from ocbpy._boundary import OCBoundary
from ocbpy.cycle_boundary import match_data_ocb

# Define the global variables
__version__ = str('0.2.1')
