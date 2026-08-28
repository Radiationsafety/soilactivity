References
==========

Key publications and data sources used in SoilActivity.

Fredholm Equation and SAD Reconstruction
-----------------------------------------

Chizhov A, Kashparov V, Lund E, Zvonova I, Golikov V (2019)
    Reconstruction of 137Cs contamination in the Bryansk–Belarus
    border area using the Fredholm integral equation.
    *Journal of Radiological Protection* **39**, 354–372.
    DOI: 10.1088/1361-6498/ab0613.
    Introduces the Fredholm equation approach for SAD reconstruction from
    dose-rate maps, including building visibility masks and Tikhonov
    regularisation.

Chizhov A, Kashparov V (2023)
    Application of the Fredholm equation for reconstructing surface
    activity distributions of radionuclides.
    Introduces the normalising factor framework and extends the method to
    radionuclide mixtures.

Chizhov A, Kashparov V (2024)
    Further developments in Fredholm-based SAD reconstruction.
    Extensions including non-negativity constraints, alternative kernels,
    and refined visibility models.

Specific Air Kerma Rate (SAKR)
------------------------------

Jacob P, Paretzke HG, Rosenbaum H, Zankl M (1990)
    GSF-2/90. *Calculation of organ doses from environmental gamma-rays
    using human phantoms and Monte Carlo methods*.
    GSF – National Research Center for Environment and Health, Neuherberg.
    Provides the SAKR values for Cs-137 on roofs (1.82 nGy h⁻¹ per kBq m⁻²)
    used in the Fredholm equation normalisation.

Conversion Coefficients
-----------------------

ICRP Publication 74 (1996)
    "Conversion Coefficients for use in Radiological Protection against External Radiation", Annals of the ICRP 26(3/4).
    Provides H\*(10)/K\_a and H\*(10)/Φ conversion coefficients as
    functions of photon energy, used in the dosimetry module.

Buildup Factors
---------------

ANS-6.4.3 (1991)
    *American National Standard for Gamma-Ray Attenuation Coefficients
    and Buildup Factors for Engineering Materials*.
    American Nuclear Society.
    Provides the Geometric Progression (GP) fit parameters for exposure
    buildup factors used in the buildup module.

Trubey DK (1988)
    *New Gamma-Ray Buildup Factor Data for Point Kernel Calculations:
    ANS-6.4.3 Standard Reference Data*.
    ORNL/RSIC-49. Oak Ridge National Laboratory.
    Original tabulation of buildup factors for 23 materials over a wide
    range of energies and depths.

Chernobyl Data
--------------

Kashparov VA, Ahamdach N, Zvarich SI, Yoschenko VI, Maloshtan IM, Dewiere L (2018)
    Soil contamination with ⁹⁰Sr in the near zone of the
    Chernobyl Nuclear Power Plant.
    *Science of the Total Environment* **622**, 937–944.

Kashparov VA et al (2020)
    Radionuclide contamination of the Chernobyl Exclusion Zone:
    data, models, and predictions.
    Provides fuel composition vectors and activity fractions used in
    :data:`~soilactivity.radionuclides.CHERNOBYL_FUEL_VECTOR_131D`.

Photon Attenuation Data
-----------------------

NIST XCOM
    *Photon Cross Sections Database*.
    National Institute of Standards and Technology.
    https://physics.nist.gov/xcom
    Provides mass attenuation coefficients (μ/ρ) and mass energy-absorption
    coefficients (μ\_en/ρ) for elements Z=1–100 over 1 keV – 100 GeV.
    Used by the :mod:`soilactivity.attenuation` module.

Kerma Constants
---------------

Mashkovich VV, Kudryavtseva AV (1995)
    *Protection from Ionizing Radiation* (in Russian). 4th edition.
    Energoatomizdat, Moscow. Primary source for kerma constants
    in the former Soviet literature.

Ninkovic MM, Adrovic F (2012)
    Air Kerma Rate Constants for Nuclides Important to Gamma Ray Dosimetry.
    DOI: 10.5772/39170.
    Provides evaluated kerma constants for 21 radionuclides.
