API Reference
=============

Auto-generated documentation for all public modules in soilactivity.

Core
----

.. automodule:: soilactivity.core
   :members: Unfolder, UnfoldingResult
   :show-inheritance:

SAD Reconstruction
-------------------

.. automodule:: soilactivity.reconstructor
   :members: SadReconstructor, SadResult
   :show-inheritance:

Fredholm Equation
-----------------

.. automodule:: soilactivity.fredholm
   :members: build_fredholm_matrix, build_fredholm_matrix_no_vis, solve_fredholm_tikhonov, solve_fredholm_tikhonov_nn, raster_coords, raster_to_vector, vector_to_raster

Buildup Factors (ANS-6.4.3)
----------------------------

.. automodule:: soilactivity.buildup
   :members: get_buildup, gp_buildup_water, buildup_for_mixture, AVAILABLE_MATERIALS, ANS_ENERGIES, ANS_DEPTHS

Photon Attenuation (NIST XCOM)
-------------------------------

.. automodule:: soilactivity.attenuation
   :members: lookup_mu_rho, lookup_mu_en_rho, lookup, mixture_mu_rho, mixture_mu_en_rho, linear_attenuation, mean_free_path, NIST_AIR_DRY_COMPOSITION, NIST_WATER_COMPOSITION, NIST_CONCRETE_COMPOSITION, NIST_SOIL_COMPOSITION

Dosimetry (ICRP 74)
--------------------

.. automodule:: soilactivity.dosimetry
   :members: h_star_10_over_Ka, h_star_10_over_phil, kerma_per_fluence_air, point_source_dose_rate, ICRP74_ENERGIES_MEV

Radionuclides
-------------

.. automodule:: soilactivity.radionuclides
   :members: KERMA_CONSTANTS, NORMALIZING_FACTORS, NORMALIZING_FACTORS_BY_RADIONUCLIDE, SAKR_CS137_ROOF, CHERNOBYL_FUEL_VECTOR_131D, get_normalizing_factor, mixture_kerma_constant, mixture_sakr

Method of Conversion Coefficients
----------------------------------

.. automodule:: soilactivity.mcc
   :members: mcc_ader_to_sad, mcc_sad_to_ader, mcc_coefficient, mcc_total_activity

Lorenz Curve / Gini
---------------------

.. automodule:: soilactivity.lorenz
   :members: lorenz_curve, lorenz_gini_coefficient, lorenz_compactness_ratio

Diagnostics
-----------

.. automodule:: soilactivity.diagnostics
   :members: slae_condition_number, slae_error_bound, slae_finer_error_estimate

Visibility
----------

.. automodule:: soilactivity.visibility
   :members: compute_visibility_matrix, visibility_radius_mask

Interpolation (legacy)
------------------

.. automodule:: soilactivity.interpolation

Spatial Interpolation
--------------------

.. automodule:: soilactivity.spatial_interpolation
   :members:
   :show-inheritance:

Sensitivity
-----------

.. automodule:: soilactivity.sensitivity

Correlation
-----------

.. automodule:: soilactivity.correlation
   :members: information_correlation_coefficient, entropy

Solvers
--------

.. automodule:: soilactivity.solvers
