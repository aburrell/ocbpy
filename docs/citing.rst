.. _cite:

Citation Guide
==============

When publishing work that uses OCBpy, please cite both the package and the
boundary data set.  Specifying which version of OCBpy used will also improve
the reproducibility of your presented results.


.. _cite-ocbpy:

OCBpy
-----

* Burrell, A. G., et al. (2022). aburrell/ocbpy:
  Version 0.3.0. Zenodo. doi:10.5281/zenodo.1217177.

.. code-block:: latex
   
    @Misc{ocbpy,
          author = {Burrell, A. G. and Chisham, G. and Reistad, J. P.},
	  title  = {aburrell/ocbpy: Version 0.3.0},
  	  year   = {2022},
	  date   = {2022-10-21},
	  doi    = {10.5281/zenodo.1179230},
	  url    = {http://doi.org/10.5281/zenodo.1179230},
	  }

This package was first described in the python in heliophysics over article,
which may also be cited if a description of the package is desired.

* Burrell, A. G., et al. (2018). Snakes on a spaceship — An overview of Python
  in heliophysics. Journal of Geophysical Research: Space Physics, 123,
  10,384–10,402. doi:10.1029/2018JA025877.

.. include:: ../AUTHORS.rst


.. _cite-image:
	     
IMAGE FUV Boundaries
--------------------

Please cite both the papers discussing the instrument and the appropriate
boundary retrieval method.

* **SI12/SI13**: Mende, S., et al. Space Science Reviews (2000) 91: 287-318.
  http://doi.org/10.1023/A:1005292301251.
* **WIC**: Mende, S., et al. Space Science Reviews (2000) 91: 271-285.
  http://doi.org/10.1023/A:1005227915363.
* **OCB**: Chisham, G. (2017) A new methodology for the development of
  high‐latitude ionospheric climatologies and empirical models,
  J. Geophys. Res. Space Physics, 122, 932–947, doi:10.1002/2016JA023235.
* **OCB**: Chisham, G. et al. (2022) Ionospheric Boundaries Derived from Auroral
  Images. JGR Space Physics, 127, 7, e2022JA030622, doi:10.1029/2022JA030622.
* **OCB**: Chisham, G. (2022). Ionospheric boundaries derived from IMAGE
  satellite mission data (May 2000-October 2002), version 2.0. [Dataset]. NERC
  EDS UK Polar Data Centre.
  https://doi.org/10.5285/fa592594-93e0-4ee1-8268-b031ce21c3ca


.. _cite-ampere:

AMPERE Boundaries
-----------------

Please follow the AMPERE data usage requirements provided by
`APL <https://ampere.jhuapl.edu/info/?page=infoRulesTab>`_, acknowledge the
AMPERE team and the use of AMPERE data, cite the boundary retrieval method, and
(if using the OCB, not the EAB/HMB) the OCB correction method. The V2 data set
also includes fits to the R1 and R2 peaks, for those interested.

* **FAC**: Milan, S. E., et al. (2015): Principal Component Analysis of
  Birkeland currents determined by the Active Magnetosphere and Planetary
  Electrodynamics Response Experiment, J. Geophys. Res. Space Physics, 120,
  doi:10.1002/2015JA021680
* **FAC**: Milan, Stephen (2023): AMPERE R1/R2 FAC radii v2. University of
  Leicester. Dataset. https://doi.org/10.25392/leicester.data.22241338.v1
* **OCB**: Burrell, A. G., et al. (2020): AMPERE Polar Cap Boundaries,
  Ann. Geophys., 38, 481-490, http://doi.org/10.5194/angeo-38-481-2020


.. _cite-dmsp:

DMSP SSJ Boundaries
-------------------

The archived DMSP SSJ boundaries are retrieved using
`zenodo_get <https://github.com/dvolgyes/zenodo_get>`_.  The citations for the
boundary method and data set are provided below.

* **SSJ Auroral Boundaries**: Kilcommons, L. M., R. J. Redmon, and D. J. Knipp
  (2017), A new DMSP magnetometer and auroral boundary data set and estimates
  of field-aligned currents in dynamic auroral boundary coordinates, J. Geophys.
  Res. Space Physics, 122, 9068–9079, doi:10.1002/2016JA023342.

* **SSJ Auroral Boundaries (2010-2014)**: Kilcommons, L., et al. (2019).
  Defense Meteorology Satellite Program (DMSP) Electron Precipitation (SSJ)
  Auroral Boundaries, 2010-2014 (Version 1.0.0) [Data set]. Zenodo.
  http://doi.org/10.5281/zenodo.3373812
