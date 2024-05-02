Changelog
=========

Summary of all changes made since the first stable release

0.4.0 (XX-XX-2024)
------------------
* DEP: Deprecated functions that depend on ssj_auroral_boundary_package
* DEP: Removed classes and kwargs deprecated in v0.3.0
* MAINT: Added a pyproject.toml file and removed setup.py
* MAINT: Updated numpy logic to address DeprecationWarnings
* MAINT: Updated GitHub Action yamls to use pyproject.toml and cycle versions
* MAINT: Updated GitHub Action yamls to use coveralls-reporter
* ENH: Added AMPERE OCBs up to 2022, using updated AMPERE data
* ENH: Added AMPERE EABs, using the Heppner-Maynard boundary as a valid EAB
* ENH: Changed default directory ID to use `pathlib`
* ENH: Allow data padding in `pysat_instrument` functions
* ENH: Created separate vector transformation functions to support multiple
       coordinate systems
* ENH: Updated VectorData and pysat_instruments to allow geographic inputs
* BUG: Fixed a typo in the documentation's pysat example
* BUG: Added an error catch for badly formatted SuperMAG file reading
* BUG: Fixed the flat/zero masking for vector pole angles with array input
* TST: Added a new CI test for the Test PyPi Release Candidate
* TST: Reduced duplication by creating a common test class and test variable
* TST: Added a ReadTheDocs yaml
* DOC: Improved docstring style compliance and correctness
* DOC: Updated pysat example to use geodetic inputs and improve the EAB plot

0.3.0 (10-21-2022)
------------------
* BUG: Fixed header initialization error general instrument loading routine
* BUG: Fixed time cycling in the `supermag2ascii_ocb` function
* DEP: Moved OCBoundary class to hidden sub-module, `_boundary`
* DEP: Moved `ocboundary` functions to new sub-module, `cycle_boundary`
* DEP: Deprecated kwargs no longer needed to select good IMAGE data
* DEP: Changed the format for vector data stored in pysat Instruments
* DOC: Improved the PEP8 and numpydoc compliance in the documentation examples
* DOC: Updated citations
* DOC: Updated cross-referencing and added missing API sections
* DOC: Added examples for DMSP SSJ boundaries, pysat Instruments, and the
       DualBoundary class, updated the README example
* DOC: Improved documentation configuration
* ENH: Added a setup configuration file
* ENH: Changed class `__repr__` to produce a string `eval` can use as input
* ENH: Updated the IMAGE OCB files and added EAB files
* ENH: Added EAB and dual-boundary classes
* ENH: Added function to select data along a satellite track
* ENH: Changed attributes in VectorData into properties to ensure expected
       behaviour if altering the class data after initialisation
* ENH: Added IMAGE SI12, SI13, and WIC DMSP corrections to `harmonic`
* ENH: Made scaling optional for SuperMAG and SuperDARN vorticity data
* MAINT: Removed support for Python 2.7, 3.5, and 3.6; added support for 3.10
* MAINT: Improved PEP8 compliance
* MAINT: Updated pysat routines to v3.0.0 standards
* MAINT: Updated CDF installation
* MAINT: Removed now-unnecessary logic in unit tests for Windows OS
* REL: Added a .zenodo.json file
* TST: Integrated and removed Requires.io; it requires a payed plan for GitHub
* TST: Added flake8 and documentation tests to CI
* TST: Moved all configurations to setup.cfg, removing .coveragecfg
* TST: Improved test coverage, specifically adding pysat xarray tests and
       expanding unit tests for `__repr__` methods
* TST: Migrated to GitHub Actions from Travis CI and Appveyor

0.2.1 (11-24-2020)
------------------
* DOC: Updated examples in README
* BUG: Fixed an error in determining the sign and direction of OCB vectors
* STY: Changed a ValueError in VectorData to logger warning


0.2.0 (10-08-2020)
------------------
* First stable release
