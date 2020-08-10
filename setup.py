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
    read_kwargs = {"encoding": "utf8"}

# Run setup
setup(name='ocbpy',
      version='0.2.0',
      url='https://github.com/aburrell/ocbpy',
      author='Angeline G. Burrell',
      author_email='angeline.burrell@nrl.navy.mil',
      description='Location relative to open/closed field line boundary',
      long_description=read('README.md', read_kwargs),
      long_description_content_type="text/markdown",
      packages=find_packages(),
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "Topic :: Scientific/Engineering :: Physics",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Natural Language :: English",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: Microsoft :: Windows",
          "Operating System :: POSIX",
      ],
      install_requires=[
          'numpy',
          'aacgmv2',
      ],
      extras_require = {'pysat_instruments': ['pysat>=2.0.0'],
                        'dmsp_ssj': ['ssj_auroral_boundaries']},
      include_package_data=True,
      zip_safe=False,
)
