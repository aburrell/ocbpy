Changelog
=========

Summary of all changes made since the first stable release

0.3.0 (XX-XX-2022)
------------------
* REL: Added a .zenodo.json file
* DOC: Improved the PEP8 compliance in the documentation examples
* DOC: Improved the docstring numpydoc compliance
* DOC: Updated cross-referencing and added missing API sections
* BUG: Fixed header initialization error general instrument loading routine
* ENH: Added a setup configuration file
* ENH: Changed class `__repr__` to produce a string `eval` can use as input
* ENH: Updated the IMAGE OCB files and added EAB files
* MAINT: Removed support for Python 2.7, 3.5, and 3.6; added support for 3.10
* MAINT: Improved PEP8 compliance
* MAINT: Updated pysat routines to v3.0.0 standards
* TST: Integrated Requires.io
* TST: Added flake8 and documentation tests to CI
* TST: Moved all configurations to setup.cfg, removing .coveragecfg
* TST: Added pysat xarray tests to the pysat test suite
* TST: Added new unit tests for `__repr__` methods
* TST: Migrated to GitHub Actions from Travis CI and Appveyor

0.2.1 (11-24-2020)
------------------
* DOC: Updated examples in README
* BUG: Fixed an error in determining the sign and direction of OCB vectors
* STY: Changed a ValueError in VectorData to logger warning


0.2.0 (10-08-2020)
------------------
* First stable release
