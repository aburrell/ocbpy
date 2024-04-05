#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017, AGB
# Full license can be found in License.md
# -----------------------------------------------------------------------------
"""Common test classes and variables."""

from io import StringIO
import logging
import os
import pathlib
import unittest

import ocbpy

test_dir = os.path.join(str(pathlib.Path(
    ocbpy.tests.__file__).resolve().parent), "test_data")


class TestLogWarnings(unittest.TestCase):
    """Unit tests for logging warnings."""

    def setUp(self):
        """Initialize the logging test environment."""

        self.lwarn = u''
        self.lout = u''
        self.log_capture = StringIO()
        ocbpy.logger.addHandler(logging.StreamHandler(self.log_capture))
        ocbpy.logger.setLevel(logging.WARNING)
        return

    def tearDown(self):
        """Tear down the logging test environment."""

        del self.lwarn, self.lout, self.log_capture
        return

    def eval_logging_message(self):
        """Evaluate the logging message."""

        # Test logging error message and data output
        self.lout = self.log_capture.getvalue()
        self.assertRegex(self.lout, self.lwarn)
        return
