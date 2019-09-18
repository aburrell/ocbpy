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
VectorData    Vector data point
---------------------------------------------------------------------------

Modules
---------------------------------------------------------------------------
instruments    Instrument-specific OCB gridding functions
ocb_time       Time manipulation routines
---------------------------------------------------------------------------
"""
from __future__ import (absolute_imports)

from . import (ocboundary, ocb_scaling, ocb_time, ocb_correction)
from .ocboundary import (OCBoundary, match_data_ocb)
from . import (instruments)

__version__ = str('0.2b2')
__default_file__ = "boundaries/si13_north_circle"
