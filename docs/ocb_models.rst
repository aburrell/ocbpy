.. _bound-model:


Boundary Models
===============

Another option for providing boundaries are model specifications. Several
empirically derived mathematical models are included in
:py:mod:`ocbpy.boundaries.models` to allow access to these formulations. These
models typically depend on magnetic local time (MLT) and a geomagnetic or solar
wind index. OCBpy requires that these functions provide the boundary location in
co-latitude (degrees of magnetic latitude away from the pole), have MLT as
the first input argument, and that all other inputs be keyword arguments.


.. _bound-model-starkov:

Starkov
-------

The Starkov 1994 model (see :ref:`cite-starkov`) uses a mathematical formulation
based on All-Sky Imager data and the Auroral Electrojet Lower envelope index.
They specify three boundaries: the polar edge of the auroral oval, the
equatorward edge of the auroral oval, and the equatorward edge of the diffuse
aurora (usually more equatorward than the discrete edge). These may be accessed
in the code here using the boundary keyworkds: 'ocb', 'eab', and 'diffuse',
respectively.


.. _bound-model-guss:

Gussenhoven
-----------

The Gussenhoven 1983 model (see :ref:`cite-guss`) uses a mathematical
formulation based on DMSP data and the Kp index. They specify a single boundary:
the equatorward edge of the diffuse aurora. As this model defines the boundary
using separate linear fits for different MLT bins, the results may be returned
at the binned times ('binned'), at the nearest binned time ('closest'), or at
the requested times using a circle fit to all of binned times ('circle').


.. _bound-model-champ:

CH-Aurora-2014
--------------

The CH-Aurora-2014 model (see :ref:`cite-xiong`) uses a mathematical
formulation based on CHAMP field-aligned current (FAC) data and a delayed
time-history of the Newell Coupling Function. They specify both auroral
boundaries in each hemisphere.  Although this model may be driven entirely by
the Newell Coupling Function, the authors recommend adjusting the model output
with CHAMP measurements of the FAC boundaries.


.. _bound-model-module:

Boundary Models Module
----------------------
.. automodule:: ocbpy.boundaries.models
    :members:
