# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB * GC
# Full license can be found in LICENSE.txt
# ----------------------------------------------------------------------------
"""Boundary file utilities."""

from ocbpy import logger
from ocbpy.boundaries import files

try:
    from ocbpy.boundaries import dmsp_ssj_files
except ImportError as ierr:
    logger.warning(ierr)
