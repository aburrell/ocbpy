OCB Datasets
=============

OCBs must be obtained from observations for this coordinate transformation.
Data from three auroral instruments provide northern hemisphere OCB locations
for 3 May 2000 03:01:42 UT - 22 Aug 2002 00:01:28, though not all of the times
included in these files contain high-quality estimations of the OCB.
Recommended selection criteria are included as defaults in the OCBoundary class.
You can read more about the OCB determination and this selection criteria in
`Chisham (2017) <http://onlinelibrary.wiley.com/doi/10.1002/2016JA023235/pdf>`_.
The three auroral instruments are the IMAGE FUV
`Spectrographic Imager SI12 and SI13 <https://link.springer.com/chapter/10.1007/978-94-011-4233-5_10>`_, as well as the
`Wideband Imaging Camera (WIC) <https://link.springer.com/chapter/10.1007/978-94-011-4233-5_9>`_,.

The OCB datasets can be found in ``ocbpy/boundaries``.

OCB datasets can also be obtained from AMPERE (Active Magnetosphere and
Planetary Electrodynamics Response Experiment) R1/R2 Field-Aligned Current (FAC)
boundary data.  This data is provided for both hemispheres between 2010-2016,
inclusive. Because there is an offset between the R1/R2 FAC boundary and the
OCB, a correction is required.  This correction can be implemented using the
routines in **ocbpy.ocb_correction**.  More information about the method behind
the identification of these boundaries and their offset can be found in
`Milan et al. (2015) <http://doi.wiley.com/10.1002/2015JA021680>`_ and
`Burrell et al. (2019) <https://doi.org/10.5194/angeo-2019-113>`_.
