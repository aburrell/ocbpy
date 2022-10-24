#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB & GC
# Full license can be found in License.md
# ----------------------------------------------------------------------------
"""Auroral oval and polar cap normalised location calculation tools."""

# Define a logger object to allow easier log handling
import logging

logging.raiseExceptions = False
logger = logging.getLogger('ocbpy_logger')

# Import the package modules and top-level classes

from ocbpy import _boundary
from ocbpy import boundaries
from ocbpy import cycle_boundary
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
__version__ = str('0.3.0')
