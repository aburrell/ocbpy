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

from ocbpy import _boundary  # noqa F401 E402
from ocbpy import boundaries  # noqa F401
from ocbpy import cycle_boundary  # noqa F401
from ocbpy import instruments  # noqa F401
from ocbpy import ocb_correction  # noqa F401
from ocbpy import ocb_scaling  # noqa F401
from ocbpy import ocb_time  # noqa F401

from ocbpy._boundary import DualBoundary  # noqa F401
from ocbpy._boundary import EABoundary  # noqa F401
from ocbpy._boundary import OCBoundary  # noqa F401
from ocbpy.cycle_boundary import match_data_ocb  # noqa F401

# Define the global variables
__version__ = str('0.3.0')
