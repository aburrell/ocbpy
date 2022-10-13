.. _cite:

Citation Guide
==============

When publishing work that uses OCBpy, please cite both the package and the
boundary data set.  Specifying which version of OCBpy used will also improve
the reproducibility of your presented results.


.. _cite-ocbpy:

OCBpy
-----

* Burrell, A. G., et al. (2020). aburrell/ocbpy:
  Version 0.2.0. Zenodo. doi:10.5281/zenodo.1217177.

.. code-block:: latex
   
    @Misc{ocbpy,
          author = {Burrell, A. G. and Chisham, G. and Reistad, J. P.},
	  title  = {aburrell/ocbpy: Version 0.2.0},
  	  year   = {2020},
	  date   = {2020-06-10},
	  doi    = {10.5281/zenodo.1179230},
	  url    = {http://doi.org/10.5281/zenodo.1217177},
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
  J. Geophys. Res. Space Physics, 122, 932–947,
  http://doi.org/10.1002/2016JA023235.
* **OCB**: Chisham, G. et al. (2022) Ionospheric Boundaries Derived from Auroral
  Images. In Prep.
* **OCB**: Chisham, G. (2017) Auroral Boundary Derived from IMAGE Satellite
  Mission Data (May 2000 - Oct 2002), Version 1.1, Polar Data Centre, Natural
  Environment Research Council, UK.
  http://doi.org/10.5285/75aa66c1-47b4-4344-ab5d-52ff2913a61e.
* **OCB**: Chisham, G., et al. (2022) Ionospheric boundaries derived from
  auroral images. Journal of Geophysical Research: Space Physics, 127,
  e2022JA030622. https://doi.org/10.1029/2022JA030622.


.. _cite-ampere:

AMPERE Boundaries
-----------------

Please follow the AMPERE data usage requirements provided by
`APL <https://ampere.jhuapl.edu/info/?page=infoRulesTab>`_ and cite the R1/R2
FAC boundary retrieval method and the OCB correction method.

* **FAC**: Milan, S. E. (2019): AMPERE R1/R2 FAC radii. figshare. Dataset.
  https://doi.org/10.25392/leicester.data.11294861.v1
* **OCB**: Burrell, A. G., et al. (2020): AMPERE Polar Cap Boundaries,
  Ann. Geophys., 38, 481-490, http://doi.org/10.5194/angeo-38-481-2020


.. _cite-dmsp:

DMSP SSJ Boundaries
-------------------

The DMSP SSJ boundaries are retrieved using
`ssj_auroral_boundary <https://github.com/lkilcommons/ssj_auroral_boundary>`_.
Please follow the citation guidelines on their page.  The general reference
for the DMSP SSJ boundary data set is provided below.

* **SSJ Auroral Boundaries (2010-2014)**: Kilcommons, L., et al. (2019).
  Defense Meteorology Satellite Program (DMSP) Electron Precipitation (SSJ)
  Auroral Boundaries, 2010-2014 (Version 1.0.0) [Data set]. Zenodo.
  http://doi.org/10.5281/zenodo.3373812
