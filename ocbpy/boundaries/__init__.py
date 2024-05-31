#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DOI: 10.5281/zenodo.1179230
# Full license can be found in LICENSE.txt
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Boundary file utilities."""

from ocbpy import logger
from ocbpy.boundaries import files  # noqa F401

try:
    from ocbpy.boundaries import dmsp_ssj_files  # noqa F401
except ImportError as ierr:
    logger.warning(ierr)
