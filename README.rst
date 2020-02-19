******
o3soil
******

A toolkit of extensions for using o3seespy for geotechnical simulations


Package overview
================

o3soil is high-level package that takes advantage of several libraries to perform geotechnical analyses
without rewriting common functions.

The package is based on the generic ``sfsimodels`` package soil object (``sfsimodels.Soil()``), soil
profile object (``sfsimodels.SoilProfile()``) and building object (``sfsimodels.Building()``), as well as the
the acceleration time series object from the eqsig package (``eqsig.AccSignal()``).
These objects allow soil, building and ground shaking to be defined in an agnostic way which is compatible
with other software and Python packages and export them to json file format.
The toolkits in this package convert the generic objects into o3seespy objects and run an OpenSees analysis.

The eco-system of Python packages that relate to this package are outlined in the figure below.

.. image:: https://eng-tools.github.io/static/img/package-space.svg
    :width: 80%
    :align: center
    :alt: Geotechnical Python packages


Key areas:
----------

 - Site response analysis
 - Seismic Soil-foundation-structure interaction
 - Foundation bearing capacity
 - Liquefaction triggering
 - Element test behaviour
 - Lateral spreading