"""Bundled data files for soilactivity.

Includes:
- ``buildup_factors_ans643.json``: full ANSI/ANS-6.4.3-1991 exposure buildup
  factor tables for 26 materials (elements + water/air/concrete).
  25 energies (0.015-15 MeV) x 16 depths (0.5-40 mfp) = 400 cells per material.
  Source: Trubey 1988, ORNL/RSIC-49.

- ``gp_coefficients_water_ans643.json``: GP (Geometric Progression, Harima)
  fitting coefficients for water. 24 energies x 5 params (b, c, a, Xk, d)
  for two response functions (water-kerma, air-kerma).
"""
