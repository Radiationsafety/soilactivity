SoilActivity — Radionuclide Activity Reconstruction and Dosimetry
====================================================================

**Version:** 0.4.0

SoilActivity is a Python package for reconstructing spatial distributions of
radionuclide activity in soil from in-situ gamma spectrometry measurements. It
provides tools for 3D volumetric unfolding, 2D Fredholm equation-based SAD
(surface activity distribution) reconstruction, dosimetry calculations,
photon attenuation and buildup factor modelling, and spatial statistics.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api
   methods
   examples
   radionuclides
   references


Features
--------

- **3D volumetric unfolding** — Reconstruct depth profiles of radionuclide
  activity from measured spectra using MLEM or Tikhonov regularisation.
- **2D Fredholm SAD reconstruction** — Solve the Fredholm integral equation
  of the first kind to recover surface activity distributions from dose-rate
  maps, with optional building visibility masks.
- **Method of Conversion Coefficients (MCC)** — Convert ADER to SAD using
  energy-specific conversion coefficients.
- **Dosimetry** — ICRP 74 conversion coefficients for ambient dose equivalent
  and air kerma calculations.
- **Photon attenuation** — NIST XCOM mass attenuation coefficients for
  elements, compounds, and mixtures.
- **Buildup factors** — ANS-6.4.3 exposure buildup factors with Geometric
  Progression parametrisation.
- **Spatial statistics** — Lorenz curve analysis, Gini coefficient, and
  information correlation coefficient for characterising activity heterogeneity.
- **66 unfolding methods** — Comprehensive benchmark suite of direct, iterative,
  Bayesian, optimisation, and evolutionary approaches to spectrum unfolding.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
