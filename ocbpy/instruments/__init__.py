# -*- coding: utf-8 -*-
# Copyright (C) 2017 AGB
# Full license can be found in LICENSE.txt
# ----------------------------------------------------------------------------
"""Instrument specific Boundary support."""

from ocbpy import logger
from ocbpy.instruments import general  # noqa F401
from ocbpy.instruments.general import test_file  # noqa F401
from ocbpy.instruments import supermag  # noqa F401
from ocbpy.instruments import vort  # noqa F401

try:
    from ocbpy.instruments import pysat_instruments  # noqa F401
except ImportError as ierr:
    logger.warning(ierr)
