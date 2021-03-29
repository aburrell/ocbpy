#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2017, AGB & GC
# Full license can be found in License.md
#-----------------------------------------------------------------------------

from os import path
from setuptools import setup, find_packages

# Define a read function for using README for long_description
def read(fname, fkwargs=dict()):
    return open(path.join(path.dirname(__file__), fname), **fkwargs).read()

# Define default kwargs for python 3
read_kwargs = {"encoding": "utf8"}

# Run setup
setup(name='ocbpy',
      version='0.2.1',
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
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
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
