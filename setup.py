#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------

from __future__ import absolute_import
from os import path
from setuptools import setup, find_packages
from sys import version_info

# Define a read function for using README for long_description

def read(fname, fkwargs=dict()):
    return open(path.join(path.dirname(__file__), fname), **fkwargs).read()

# Define default kwargs for python2/3
read_kwargs = dict()
if version_info.major == 3:
    read_kwargs = {"encoding":"utf8"}

# Define a test suite

def ocb_test_suite():
    import unittest

    test_loader = unittest.TestLoader()
    test_path = path.join(path.dirname(__file__), 'ocbpy/tests')
    test_suite = test_loader.discover(test_path, pattern='test_*.py')
    return test_suite

# Run setup

setup(name='ocbpy',
      version='0.2b2',
      url='https://github.com/aburrell/ocbpy',
      author='Angeline G. Burrell',
      author_email='agb073000@utdallas.edu',
      description='Location relative to open/closed field line boundary',
      long_description=read('README.md', read_kwargs),
      long_description_content_type="text/markdown",
      packages=find_packages(),
      classifiers=[
          "Development Status :: 4 - Beta",
          "Topic :: Scientific/Engineering :: Physics",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Natural Language :: English",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: Microsoft :: Windows",
          "Operating System :: POSIX",
      ],
      install_requires=[
          'numpy',
          'logbook'
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='setup.ocb_test_suite',
)
