# -*- coding: utf-8 -*-
# Copyright (C) 2017 AGB
# Full license can be found in LICENSE.txt
# ----------------------------------------------------------------------------
"""Instrument specific Boundary support."""

from ocbpy import logger
from ocbpy.instruments import general
from ocbpy.instruments.general import test_file
from ocbpy.instruments import supermag
from ocbpy.instruments import vort

try:
    from ocbpy.instruments import pysat_instruments
except ImportError as ierr:
    logger.warning(ierr)
