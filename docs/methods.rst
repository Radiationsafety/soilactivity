Methods Overview
=================

SoilActivity implements several families of computational methods for
reconstructing radionuclide activity from radiation measurements.

3D Volumetric Unfolding
------------------------

The :class:`~soilactivity.core.Unfolder` class reconstructs depth-dependent
activity profiles from in-situ gamma spectra. Two solvers are available:

**MLEM (Maximum Likelihood Expectation Maximisation)**
    An iterative statistical algorithm that finds the maximum-likelihood
    solution under Poisson statistics. The iteration count acts as an
    implicit regularisation parameter.

**Tikhonov Regularisation**
    Solves the linear system :math:`\mathbf{R} \mathbf{a} = \mathbf{n}` with
    a smoothness penalty. The regularisation parameter :math:`\alpha` controls
    the trade-off between data fidelity and solution smoothness.

Both methods operate on a response matrix :math:`\mathbf{R}` that encodes the
physics of photon transport from each depth layer to the detector.

2D Fredholm SAD Reconstruction
------------------------------

The :class:`~soilactivity.reconstructor.SadReconstructor` solves the
Fredholm integral equation of the first kind:

.. math::

    P(\mathbf{r}) = W \int Q(\mathbf{r}, \mathbf{r}') \, \mathrm{Vis}(\mathbf{r}, \mathbf{r}') \, A(\mathbf{r}') \, d\mathbf{r}'

where :math:`P` is the measured dose rate, :math:`A` is the unknown surface
activity distribution (SAD), :math:`Q` is the point-source dose-rate kernel,
:math:`\mathrm{Vis}` is the visibility function (accounting for buildings
and other barriers), and :math:`W` is a normalising factor.

The discrete system is solved via Tikhonov regularisation with non-negativity
constraints.

Fourier Convolution (example00)
----------------------------------

For scenarios without barriers (full visibility), the Fredholm equation
reduces to a 2D convolution, which can be solved efficiently in the Fourier
domain. This is demonstrated in ``example00`` and is the computationally
cheapest approach.

66 bssunfold Methods
---------------------

A comprehensive benchmark of 66 spectrum unfolding methods is provided in the
``examples/bssunfold_methods/`` directory. Each method is implemented in a
dedicated Jupyter notebook. The methods span seven categories:

**Direct Methods**

- ``01_cvxpy`` — Convex optimisation via CVXPY (quadratic programming)
- ``02_qpsolvers`` — Quadratic programming via multiple QP solvers
- ``03_tsvd`` — Truncated Singular Value Decomposition
- ``04_lanczos`` — Lanczos bidiagonalisation
- ``05_tikhonov_legendre`` — Tikhonov regularisation in Legendre basis

**Iterative Methods**

- ``06_landweber`` — Landweber iteration
- ``07_mlem`` — Maximum Likelihood Expectation Maximisation
- ``08_mlem_stop`` — MLEM with early stopping criteria
- ``09_mlem_odl`` — MLEM using the ODL framework
- ``10_gravel`` — GRAVEL (Gamma Response Estimation and VErification by
  Log-likelihood)
- ``11_doroshenko`` — Doroshenko's iterative method
- ``12_kaczmarz`` — Kaczmarz (ART) method
- ``13_sart`` — Simultaneous Algebraic Reconstruction Technique
- ``14_cgls`` — Conjugate Gradient Least Squares
- ``15_gks`` — Gold–Kaczmarz–Simultaneous method
- ``16_fista`` — Fast Iterative Shrinkage-Thresholding Algorithm
- ``17_hybrid_gmres`` — Hybrid GMRES with regularisation
- ``30_scipy_lsqr`` — LSQR via SciPy
- ``31_scipy_gmres`` — GMRES via SciPy

**Bayesian Methods**

- ``18_bayes`` — Bayesian unfolding with prior
- ``19_bayes_spline`` — Bayesian unfolding with spline prior
- ``20_bayesian_parametric`` — Bayesian parametric model fitting
- ``21_mcmc_pymc`` — Markov Chain Monte Carlo via PyMC

**Classical Spectrometry Unfolding**

- ``22_maxed`` — MAXED (Maximum Entropy Deconvolution)
- ``23_imaxed`` — Iterative MAXED
- ``24_amaxed`` — Adaptive MAXED
- ``25_amaxed_reg`` — Adaptive MAXED with regularisation
- ``26_statreg`` — Statistical regularisation (REGINA-style)
- ``27_reconst`` — Reconstitution method
- ``28_lmfit_ridge`` — LMFIT with Ridge penalty
- ``29_lmfit_lasso`` — LMFIT with LASSO penalty
- ``45_sandii`` — SAND-II iterative method
- ``46_bunki`` — BUNKI unfolding code
- ``47_bunkiut`` — BUNKI-UT unfolding
- ``48_osem`` — Ordered Subsets Expectation Maximisation
- ``49_mapem_quad`` — MAP-EM with quadratic prior
- ``50_mapem_logcosh`` — MAP-EM with log-cosh prior
- ``51_bsrem`` — Block Sequential Regularised EM
- ``52_ferdor`` — FERDOR spectrum deconvolution
- ``53_rebunki`` — Regularised BUNKI
- ``54_nsduaz`` — NSDUAZ unfolding
- ``61_staysl`` — STAY'SL unfolding

**Optimisation-Based Methods**

- ``32_mystic_fmin`` — Constrained optimisation via Mystic
- ``33_mystic_diffev`` — Differential evolution via Mystic
- ``34_mystic_hybrid`` — Hybrid optimisation via Mystic
- ``35_smt`` — Surrogate Modelling Toolbox
- ``40_cs`` — Compressed sensing / L1 minimisation
- ``41_scip`` — SCIP mixed-integer programming solver
- ``42_docplex`` — IBM CPLEX optimisation
- ``43_epic`` — EPIC unfolding
- ``44_tikhonov_tv`` — Tikhonov with Total Variation regularisation
- ``55_odl_pdhg`` — Primal-Dual Hybrid Gradient via ODL
- ``56_odl_dr`` — Douglas–Rachford splitting via ODL
- ``57_qubo`` — QUBO formulation for quantum/annealing solvers
- ``58_zfit`` — zfit parametric model fitting

**Genetic / Evolutionary Methods**

- ``36_genetic_pso`` — Particle Swarm Optimisation
- ``37_genetic_de`` — Differential Evolution
- ``38_genetic_ga`` — Genetic Algorithm
- ``39_genetic_cmaes`` — Covariance Matrix Adaptation Evolution Strategy

**Other / Hybrid Methods**

- ``59_crystal_ball`` — Crystal Ball parametric fitting
- ``60_rfsp_jul`` — RFSP/JUL reactor physics code interface
- ``62_parametric2`` — Parametric model fitting (variant 2)
- ``63_fruit_like`` — Fruit-fly optimisation inspired approach
- ``64_hybrid_parametric`` — Hybrid parametric approach
- ``65_combined`` — Combined method
- ``66_composite`` — Composite method

Spatial Statistics
------------------

**Lorenz Curve** — :func:`~soilactivity.lorenz.lorenz_curve` computes the
Lorenz curve for a 2D activity distribution, used to quantify the degree of
spatial concentration.

**Gini Coefficient** — :func:`~soilactivity.lorenz.lorenz_gini_coefficient`
derives the Gini coefficient from the Lorenz curve, providing a single-number
measure of heterogeneity (0 = uniform, 1 = maximally concentrated).

**Compactness Ratio** — :func:`~soilactivity.lorenz.lorenz_compactness_ratio`
computes a compactness metric comparing the activity distribution to a
hypothetical uniform distribution.

**Information Correlation Coefficient (ICC)** —
:func:`~soilactivity.correlation.information_correlation_coefficient` computes
the Linfoot information correlation coefficient between two distributions,
useful for comparing measured vs. reconstructed activity maps.

**Entropy** — :func:`~soilactivity.correlation.entropy` computes the Shannon
entropy of a discrete probability distribution, used as a regularisation
penalty in some Bayesian unfolding methods.

Spatial Interpolation
--------------------

The ``soilactivity.spatial_interpolation`` module provides a unified framework
for 2-D spatial interpolation of environmental measurements (dose rates, activity
densities, etc.) with 14 backends, automated method selection, and sensitivity
analysis.

Interpolator2D
~~~~~~~~~~~~~~

:class:`~soilactivity.spatial_interpolation.Interpolator2D` wraps multiple
interpolation backends behind a single ``fit`` / ``predict`` / ``predict_grid``
API. The ``method`` parameter selects one of 14 backends:

**RBF (scipy)** — Radial basis function interpolation. Four kernel variants
are available:

- ``rbf_tps`` — Thin-plate spline (default). Smooth, globally supported,
  suitable for evenly distributed data.
- ``rbf_linear`` — Linear RBF. Less smooth than TPS, faster to evaluate.
- ``rbf_cubic`` — Cubic RBF. Produces C1-continuous surfaces.
- ``rbf_gaussian`` — Gaussian RBF. Locally supported (controlled by
  ``smoothing``), suitable for dense data.

**Delaunay (scipy griddata)** — Triangulation-based interpolation:

- ``nearest`` — Nearest-neighbour lookup. Fast, no smoothing, produces
  discontinuous surfaces. Useful for classification maps.
- ``linear_delaunay`` — Piecewise linear interpolation on the Delaunay
  triangulation. Fast and robust.
- ``cubic_delaunay`` — Clough-Tocher cubic interpolation (C1-continuous).
  Smoother than linear but slower; may overshoot near sharp gradients.

**Deterministic**

- ``idw`` — Inverse Distance Weighting. Uses the 12 nearest neighbours
  with ``w_i = 1/d_i^2``. Simple, fast, and controllable via ``power`` and
  ``max_neighbors`` parameters. Implemented via ``scipy.spatial.cKDTree``.

**Meteorological schemes**

- ``barnes`` — Barnes successive-correction interpolation. Applies
  Gaussian-weighted averaging in multiple passes with decreasing scale
  length (``kappa``). Commonly used in meteorology for objective analysis.
- ``cressman`` — Cressman analysis scheme. Weight function
  ``w = (R² - r²)/(R² + r²)`` inside a fixed influence ``radius``. Produces
  smoother fields than nearest-neighbour.

**Geostatistical**

- ``kriging`` — Ordinary Kriging via ``pykrige``. Fits a variogram model
  to the data and provides Best Linear Unbiased Prediction (BLUP) with
  built-in prediction variance (accessible via ``.uncertainty()``).

**Gaussian Process (scikit-learn)**

- ``gp_rbf`` — Gaussian Process with RBF (squared exponential) kernel.
  Provides smooth interpolation with uncertainty quantification.
- ``gp_matern32`` — Gaussian Process with Matern 3/2 kernel. Less smooth
  than RBF, suitable for data with moderate roughness.
- ``gp_matern52`` — Gaussian Process with Matern 5/2 kernel. Once
  differentiable, a good default for environmental data.

All GP methods optimise kernel hyperparameters via maximum likelihood
with 5 random restarts (configurable via ``n_restarts_optimizer``). If
``smoothing > 0``, a WhiteKernel noise term is added. Prediction standard
deviation is available via ``.uncertainty()``.

**Uncertainty quantification** — The ``.uncertainty(xi, yi)`` method returns
prediction standard deviations for ``gp_*`` and ``kriging`` backends, and
``None`` for all others. This enables confidence-interval mapping and
masking of unreliable extrapolation regions.

InterpolationAutoSelector
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~soilactivity.spatial_interpolation.InterpolationAutoSelector`
automatically selects the best interpolation method from a candidate list
using cross-validation:

- For N >= 30 data points: k-fold CV (default k=5).
- For N < 30: leave-one-out CV.
- Metrics computed per fold: RMSE, MAE, R², wall-clock time.
- Selection criterion: lowest mean RMSE; ties broken by highest R².

The ``.select()`` method returns a dict with ``best_method``, ``best_score``,
and the full ``results`` list. The ``.get_ranking()`` method returns all
methods sorted by RMSE. The ``.plot_comparison()`` method produces a
horizontal bar chart of cross-validated RMSE values. The
``.get_recommendation()`` method returns a human-readable summary string.

Methods whose dependencies are missing (e.g. ``pykrige`` for ``kriging``)
are gracefully skipped with an error note, allowing mixed candidate lists
without install-time failures.

MeasurementSensitivityAnalyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~soilactivity.spatial_interpolation.MeasurementSensitivityAnalyzer`
quantifies how much each individual measurement point influences the
interpolated field. This is the spatial analog of ``bssunfold``'s
``unfold_interpret`` module and follows the ``pyoptexplain`` pattern of
perturbation-based explanation.

**Leave-one-out analysis** (``sensitivity_leave_one_out``) removes each
data point in turn, rebuilds the interpolation, and measures the change
at all evaluation grid locations. For each point *i*, the following
metrics are computed:

- ``max_influence`` — maximum absolute change across the evaluation grid.
- ``mean_influence`` — mean absolute change.
- ``influence_area_km2`` — area (in km²) where the change exceeds 5 % of
  the maximum change for that point.

**Perturbation analysis** (``sensitivity_perturbation``) perturbs each
point's z-value by ``delta_frac * max(|z[i]|, std(z))`` and measures the
resulting field change. This is useful when data points carry
measurement uncertainty and one wants to know how sensitive the
interpolated map is to small errors at each location.

The ``ranking()`` method returns all points sorted by ``max_influence``
descending. The ``critical_points(percentile)`` method filters to points
above a given influence percentile (default 90 %). The ``influence_map(xi, yi)``
method produces a 2-D grid showing the total influence at each location,
and ``plot_influence(xi, yi)`` renders it as a heatmap with measurement
points overlaid.

SparseResultInterpolator
~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~soilactivity.spatial_interpolation.SparseResultInterpolator`
addresses the common scenario where a Fredholm or volumetric reconstruction
yields results at only a few spatial locations (e.g. individual boreholes
or detector positions). It interpolates these sparse results onto a dense
regular grid with uncertainty quantification.

The ``.fit_sparse(points, values, uncertainty)`` method accepts an
(N, 2) array of coordinates, N values, and optional per-point
uncertainties. If the method is a GP and uncertainties are provided,
they are used as the noise prior (``alpha`` parameter).

The ``.interpolate_to_grid(xmin, xmax, ymin, ymax, nx, ny)`` method
returns a :class:`~soilactivity.spatial_interpolation.SparseResult`
dataclass containing:

- ``interpolated`` — (ny, nx) array of interpolated values.
- ``uncertainty`` — (ny, nx) array of prediction std, or ``None``.
- ``confidence_mask`` — boolean mask where ``std / max(|z|) < threshold``.
- ``coverage`` — fraction of grid cells passing the confidence threshold.
- ``method_used`` — name of the interpolation method.
- ``n_input_points`` — number of sparse input points.

Gaussian Process methods (``gp_rbf``, ``gp_matern52``) are recommended for
this use case because they provide built-in uncertainty estimates that
drive the confidence mask.
