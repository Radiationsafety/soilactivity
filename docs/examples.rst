Examples Gallery
=================

This page describes the example notebooks shipped with SoilActivity.

Synthetic Examples
------------------

**example00 — Fourier Convolution Unfolding**
    Demonstrates the simplest approach: solving the Fredholm equation in the
    Fourier domain when no buildings obstruct the line of sight. A synthetic
    Gaussian-shaped activity distribution is generated on a regular raster,
    convolved with the point-source dose-rate kernel, and then recovered by
    deconvolution. This is the fastest method and serves as a useful baseline.

**example01 — Tikhonov Regularisation on Synthetic Data**
    Shows Tikhonov regularisation applied to a synthetic dose-rate map. Several
    values of the regularisation parameter :math:`\alpha` are compared, and the
    condition number and error bound diagnostics are demonstrated.

**example02 — MLEM Unfolding on Synthetic Data**
    Applies the MLEM iterative algorithm to a synthetic spectrum. The effect of
    iteration count (implicit regularisation) is explored, and the result is
    compared to the Tikhonov solution.

Real Data Examples
------------------

**example03 — Chernobyl Exclusion Zone**
    Reconstructs the surface activity distribution from dose-rate measurements
    collected in the Chernobyl Exclusion Zone using Cs-137 kerma constants and
    the Fredholm equation. Demonstrates use of the Chernobyl fuel vector for
    radionuclide mixtures.

**example04 — Semipalatinsk Test Site**
    Applies SAD reconstruction to dose-rate data from the Semipalatinsk nuclear
    test site. Illustrates handling of irregular measurement geometries and
    heterogeneous source distributions.

**example05 — Co-60 Source Mapping**
    Demonstrates SAD reconstruction for a Co-60 point source using the
    higher-energy kerma constant (137.0 aGy m²/s/Bq) and corresponding
    dose-rate kernel.

66 bssunfold Methods
---------------------

The ``examples/bssunfold_methods/`` directory contains 66 Jupyter notebooks,
each implementing a different spectrum unfolding algorithm applied to a common
benchmark problem. See :doc:`methods` for the full list grouped by category.

Each notebook follows a similar structure:

1. Problem setup (response matrix, measured spectrum)
2. Solver configuration and execution
3. Result visualisation (reconstructed spectrum, residual)
4. Quality metrics (chi-squared, L2 norm, computational time)

The notebooks can be run independently and are suitable for both learning and
benchmarking purposes. They require the ``bssunfold`` package as a dependency.

Running the Examples
---------------------

The examples are Jupyter notebooks located in the ``examples/`` directory of the
source distribution. To run them::

    cd examples
    jupyter notebook

The ``bssunfold_methods/`` subdirectory contains the 66-method benchmark suite.

For the core examples (00–05), install the package with::

    pip install numpy scipy matplotlib

The bssunfold method notebooks additionally require::

    pip install bssunfold  # plus method-specific dependencies
