#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------
"""
ocboundary
-----------

Open-Closed field line Boundary (OCB) magnetic gridding

Functions
---------------------------------------------------------------------------
match_data_ocb      Matches data and OCB records
normal_evar         Normalise a variable proportional to the electric field
normal_curl_evar    Normalise a variable proportional to the curl of the
                    electric field
---------------------------------------------------------------------------

Classes
---------------------------------------------------------------------------
OCBoundary    OCB data for different times
---------------------------------------------------------------------------

Modules
---------------------------------------------------------------------------
instruments    Instrument-specific OCB gridding functions
---------------------------------------------------------------------------
"""
from __future__ import (absolute_import, unicode_literals)
import logging

__version__ = str('0.1a1')
__default_file__ = "boundaries/si13_north_circle"

# Imports
#---------------------------------------------------------------------

try:
    from ocbpy import (ocboundary, ocb_scaling)
    from ocbpy.ocboundary import (OCBoundary, match_data_ocb)
except ImportError as e:
    logging.exception('problem importing ocboundary: ' + str(e))

try:
    from ocbpy import instruments
except ImportError as e:
    logging.exception('problem importing instruments: ' + str(e))
