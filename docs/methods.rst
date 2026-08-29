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
