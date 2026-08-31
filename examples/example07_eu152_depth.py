"""Example 07: Eu-152 Depth Profile Inversion (twin experiment).

Demonstrates the fredholm depth-inversion subpackage:
- Multi-line (3 Eu-152 lines) x multi-height (0.5 m + 2 m) kernel
- Non-parametric Tikhonov/GCV inversion -> a(z)
- Parametric transport-chemistry inversion -> (A0, t_eff)
- Comparison with true profile
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from soilactivity.depth_inversion import (
    GammaLine, DepthInverter, pulse_profile, DECAY_S,
)

YEAR_S = 3.156e7

# --- Gamma lines for Eu-152 ---
lines = [
    GammaLine(121.78, 0.284, 23.8, 0.0185),
    GammaLine(344.28, 0.266, 12.2, 0.0114),
    GammaLine(1408.01, 0.209, 7.4, 0.0066),
]

# --- Build inverter (3 lines x 2 heights = 6 data channels) ---
inv = DepthInverter(lines, heights=[0.5, 2.0], z_max=0.6, n_z=24)

# --- True profile: pulse deposition 25 years ago, R=1401 ---
t = 25.0 * YEAR_S
a_true = pulse_profile(inv.z, 1e5, t, 1e-10, 0.0, 1401.0, DECAY_S["Eu-152"])

# --- Synthetic data ---
scale = 2.0e4
rng = np.random.default_rng(7)
counts = rng.poisson(np.maximum((inv.K @ a_true) * scale, 1.0)) / scale

print("=== Non-parametric Tikhonov/GCV ===")
res = inv.fit(counts, criterion="gcv")
print(res.report())

print("\n=== Parametric pulse fit ===")
out = inv.fit_parametric(counts, family="pulse", nuclide="Eu-152",
                          D=1e-10, R=1401.0)
print("parametric: A0={:.2e} Bq/m2, t={:.1f} years, chi2={:.1f}".format(
    out["A0"], out["t_years"], out["chi2"]))

# --- True values for comparison ---
areal_true = float(np.sum(a_true * inv.dz))
cdf = np.cumsum(np.clip(a_true, 0.0, None) * inv.dz)
z_med_true = float(np.interp(0.5 * areal_true, cdf, inv.z))
print("\n=== Truth ===")
print("areal A={:.3e} Bq/m2, z_median={:.1f} cm".format(
    areal_true, z_med_true * 100))
