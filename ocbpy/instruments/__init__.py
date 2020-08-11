# -*- coding: utf-8 -*-
# Copyright (C) 2017 AGB
# Full license can be found in LICENSE.txt
# ----------------------------------------------------------------------------
"""Instrument specific Open-Closed field line Boundary (OCB) magnetic gridding

Contains
--------
supermag    SuperMAG data available at: http://supermag.jhuapl.edu/
vort        SuperDARN vorticity data may be obtained from: gchi@bas.ac.uk
general     General file loading and testing routines
pysat       General pysat Instrument loading routines: https://github.com/pysat

"""
from __future__ import (absolute_import)
import logging

from ocbpy.instruments import general
from ocbpy.instruments.general import test_file
from ocbpy.instruments import supermag
from ocbpy.instruments import vort

try:
    from ocbpy.instruments import pysat_instruments
except ImportError as ierr:
    logging.warning(ierr)
