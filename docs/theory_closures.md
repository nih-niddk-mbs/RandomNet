# Theory Closures and Criticality Notes

This project intentionally keeps several theory approximations available side by
side.  The random-network simulations are used as the arbiter; no single closure
is assumed to be universally correct.

## Shared Conventions

- Weights have variance `sigma^2 / N`.
- `C_uu` denotes the covariance of the synaptic drive `u`.
- Phase-model output is `g(u) = rho * F(u)` with `rho = 1 / (2*pi)`.
- Phase shot-noise strength is `D_shot = sigma^2 * E[g(u)]`.
- A filtered white shot-noise contribution has variance `beta * D_shot / 2`.
- Quantities named `Q_smooth` or `Q_centered` already include the `sigma^2`
  prefactor when they appear in the `C_uu` equation.
- For phase closures, `R_smooth(c, C0)` means the centered gain covariance
  without `sigma^2`, with lag covariance `c` and fixed variance `C0`.

## Phase Network

The phase implementation lives in `scripts/rn_phase.py`.  The main entry point is:

```python
theory_phase_autocorr(...)
```

Important options:

- `solver="inflated_ic"`: current default closure where the filtered shot-noise
  covariance is treated as a homogeneous contribution that inflates the initial
  variance and the smooth ODE starts with `C'(0)=0`.
- `q_method="gh"`: direct 2-D Gauss-Hermite covariance.
- `q_method="qmc"`: Sobol Gaussian covariance, useful for rough checks.
- `q_method="hermite"`: 1-D Hermite-series covariance.  This avoids cancellation
  in `E[g(u0)g(ut)] - E[g]^2` when `C_uu(0)` is large.

The minimal generalized phase kernel used for exploratory oscillatory fits is:

```text
L_C C = C'' + 2*damping*C' + (omega^2 - beta^2)*C
L_C C = -beta^2 * Q_smooth(C; C0)
```

where `kernel_omega` and `kernel_damping` are dimensionless multiples of `beta`
by default.

The scalar `C_uu` closure is a smooth Gaussian reduction.  It averages over the
phase-density propagator and does not retain the near-threshold advection peaks
of `C_33(v_T,v_T,tau)` around the phase circle.  Near the smooth-feedback
threshold, the full `C_33` advection equation can matter and should be treated
as a separate closure rather than silently folded into the scalar Gaussian
approximation.

## Criticality

We now track multiple thresholds:

- `sigma_c_smooth = 1 / (rho * F'(0))`: old smooth-feedback threshold.
- `sigma_c_shot`: static smooth-feedback threshold renormalized by the
  shot-noise-dressed drive variance.  It solves
  `1 = sigma^2 * rho^2 * <F'(u)>_C^2` together with
  `C = beta * sigma^2 * rho * <F(u)>_C / 2`.
- `sigma_branch`: operational finite-branch threshold for a chosen closure.
- `g_branch = sigma_branch / sigma_c_smooth`.
- A future finite-frequency pole threshold can be added once the full kernel
  `G0^{-1}(iw) - Sigma(iw; C0)` is specified.

The static shot-renormalized threshold is computed by:

```python
phase_shot_renormalized_criticality(...)
```

This routine uses adaptive 1-D quadrature by default, rather than Gauss-Hermite,
because clipping makes `<F'(u)>` discontinuous or singular at threshold for many
`alpha` values.

The operational finite-branch threshold is estimated by:

```python
phase_operational_criticality(...)
plot_phase_operational_criticality(...)
```

`plot_phase_operational_criticality(...)` overlays the smooth, shot-renormalized,
and branch thresholds when available.

For superlinear gain (`alpha < 1`), this branch threshold can occur below the
smooth-feedback threshold.  Current quick estimates for `I=1`, `beta=1`, and the
Hermite/inflated-IC closure give `alpha=0.5` around `g_branch ~= 0.75`.

## Binary Network

The binary implementation lives in `scripts/rn_binary.py`.  The exact
linear-gain theory remains:

```python
theory_binary_autocorr(...)
```

For clipped binary gain, two approximations are kept:

- `theory_binary_clipped(...)`: effective-linear/quasi-static clipped theory.
- `theory_binary_clipped_integral(...)`: direct Gaussian integral closure using
  `p(u)=f_+(u)/(f_+(u)+mu)`.

The integral closure supports:

- `q_method="hermite"`: stable 1-D Hermite covariance of `p(u)`.
- `q_method="gh"`: direct 2-D Gauss-Hermite covariance.
- `intrinsic="telegraph"`: includes intrinsic binary switching covariance.
- `intrinsic="none"`: smooth recurrent component only.

For `C_nn`, the intrinsic telegraph term is essential; the smooth-only branch can
collapse to zero even when the simulation has clear state-switching covariance.

## Useful Plot Entrypoints

```python
plot_phase_corr_params(...)
plot_phase_corr_N(...)
plot_phase_operational_criticality(...)
plot_clipped_vs_linear(...)
```

All generated figures should be written to `data/plots/`.

`scripts/random_network.py` re-exports the split modules for older scripts.
