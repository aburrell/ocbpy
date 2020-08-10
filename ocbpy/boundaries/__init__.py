# -*- coding: utf-8 -*-
# Copyright (C) 2019, AGB * GC
# Full license can be found in LICENSE.txt
# ----------------------------------------------------------------------------
"""Boundary file utilities

Contains
--------
files          Boundary file utilities
dmsp_ssj_files DMSP SSJ boundary file utilities

"""
from __future__ import absolute_import
import logging

from ocbpy.boundaries import files

try:
    from ocbpy.boundaries import dmsp_ssj_files
except ImportError as ierr:
    logging.warning(ierr)
