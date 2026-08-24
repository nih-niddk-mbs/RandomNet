"""Fokker--Planck quadrature for an LIF neuron with Gaussian Markov drive.

The density-level effective-action saddle becomes a local Fokker--Planck
problem when the Gaussian DMFT drive has a finite-dimensional Markov
realization.  This module implements the one-mode Ornstein--Uhlenbeck case,

    dv/dt = I - v + u,          v: threshold -> reset,
    du = -gamma*u*dt + sqrt(2*gamma*variance)*dW.

The finite-volume generator is a continuous-time Markov chain.  Positive
voltage flux leaving the threshold cell is reinserted in the reset cell at the
same drive value.  Propagating the stationary threshold-flux measure gives the
regular two-event correlation; the same-event delta contribution is then
filtered analytically on the discrete time grid.

This is exact for a prescribed OU drive up to state-space discretization.  A
general self-consistent DMFT covariance requires multiple Markov modes or the
infinite-mode limit; projecting it onto one OU mode is a separate approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import expm_multiply, spsolve


@dataclass(frozen=True)
class LIFOUGrid:
    """Finite-volume grid and probability generator for the LIF--OU process."""

    voltage: np.ndarray
    drive: np.ndarray
    dv: float
    du: float
    reset_index: int
    generator: csc_matrix
    event_hazard: np.ndarray


@dataclass(frozen=True)
class LIFOUFokkerPlanckResult:
    """Stationary density and threshold-flux correlations from the FP saddle."""

    tau: np.ndarray
    field_covariance: np.ndarray
    spike_covariance: np.ndarray
    regular_spike_covariance: np.ndarray
    joint_event_density: np.ndarray
    stationary_mass: np.ndarray
    mean_rate: float
    drive_mean: float
    drive_variance: float
    grid: LIFOUGrid


@dataclass(frozen=True)
class ProjectedLIFOUDMFTResult:
    """Diagnostic fixed point after projection onto one exponential covariance."""

    drive_variance: float
    drive_decay: float
    tau: np.ndarray
    projected_covariance: np.ndarray
    dmft_covariance: np.ndarray
    parameter_residuals: np.ndarray
    projection_errors: np.ndarray
    converged: bool
    fokker_planck: LIFOUFokkerPlanckResult


@dataclass(frozen=True)
class LIFOUValidationResult:
    """Comparison of Fokker--Planck quadrature and Gaussian-path integration."""

    tau: np.ndarray
    quadrature_covariance: np.ndarray
    path_covariance: np.ndarray
    quadrature_spike_covariance: np.ndarray
    path_spike_covariance: np.ndarray
    quadrature_rate: float
    path_rate: float
    covariance_relative_error: float
    equal_time_relative_error: float
    rate_relative_error: float


@dataclass(frozen=True)
class LIFOUFirstReturnRenewalResult:
    """First-return density and renewal approximation for an LIF--OU neuron."""

    tau: np.ndarray
    first_return_density: np.ndarray
    survival_probability: np.ndarray
    renewal_density: np.ndarray
    spike_covariance: np.ndarray
    regular_spike_covariance: np.ndarray
    field_covariance: np.ndarray
    mean_rate: float
    mean_interval: float
    interval_cv: float
    return_probability: float
    full_fokker_planck: LIFOUFokkerPlanckResult


@dataclass(frozen=True)
class LIFOUReturnMapResult:
    """Spike-to-spike transition operator and its memory timescale."""

    transition_matrix: np.ndarray
    event_drive_distribution: np.ndarray
    eigenvalues: np.ndarray
    subleading_eigenvalue: complex
    spectral_radius: float
    mixing_spikes: float
    mixing_time: float
    invariance_error: float
    column_sum_error: float


def build_lif_ou_generator(
    I=2.0,
    drive_variance=0.02,
    drive_decay=1.0,
    reset=0.0,
    threshold=1.0,
    n_voltage=160,
    n_drive=81,
    drive_extent_std=5.0,
    voltage_min=0.0,
):
    """Construct a conservative upwind generator for the joint ``(v, u)`` law."""
    if threshold <= voltage_min:
        raise ValueError("threshold must exceed voltage_min")
    if not voltage_min <= reset < threshold:
        raise ValueError("reset must lie in [voltage_min, threshold)")
    if drive_variance <= 0.0 or drive_decay <= 0.0:
        raise ValueError("OU variance and decay must be positive")
    n_voltage = int(max(8, n_voltage))
    n_drive = int(max(9, n_drive))
    if n_drive % 2 == 0:
        n_drive += 1

    voltage_edges = np.linspace(voltage_min, threshold, n_voltage + 1)
    voltage = 0.5 * (voltage_edges[:-1] + voltage_edges[1:])
    dv = float(voltage_edges[1] - voltage_edges[0])
    drive_limit = float(drive_extent_std) * np.sqrt(float(drive_variance))
    drive_edges = np.linspace(-drive_limit, drive_limit, n_drive + 1)
    drive = 0.5 * (drive_edges[:-1] + drive_edges[1:])
    du = float(drive_edges[1] - drive_edges[0])
    reset_index = int(np.argmin(np.abs(voltage - reset)))

    n_state = n_voltage * n_drive
    generator = lil_matrix((n_state, n_state), dtype=float)
    event_hazard = np.zeros(n_state)
    diffusion = float(drive_decay) * float(drive_variance)

    def state_index(voltage_index, drive_index):
        return voltage_index * n_drive + drive_index

    for voltage_index, voltage_value in enumerate(voltage):
        for drive_index, drive_value in enumerate(drive):
            source = state_index(voltage_index, drive_index)
            total_rate = 0.0

            voltage_velocity = float(I) - voltage_value + drive_value
            if voltage_velocity > 0.0:
                rate = voltage_velocity / dv
                destination_voltage = (
                    reset_index
                    if voltage_index == n_voltage - 1
                    else voltage_index + 1
                )
                generator[
                    state_index(destination_voltage, drive_index), source
                ] += rate
                total_rate += rate
                if voltage_index == n_voltage - 1:
                    event_hazard[source] = rate
            elif voltage_velocity < 0.0 and voltage_index > 0:
                rate = -voltage_velocity / dv
                generator[state_index(voltage_index - 1, drive_index), source] += rate
                total_rate += rate

            drive_velocity = -float(drive_decay) * drive_value
            upward_rate = diffusion / du**2 + max(drive_velocity, 0.0) / du
            downward_rate = diffusion / du**2 + max(-drive_velocity, 0.0) / du
            if drive_index < n_drive - 1:
                generator[state_index(voltage_index, drive_index + 1), source] += (
                    upward_rate
                )
                total_rate += upward_rate
            if drive_index > 0:
                generator[state_index(voltage_index, drive_index - 1), source] += (
                    downward_rate
                )
                total_rate += downward_rate

            generator[source, source] -= total_rate

    return LIFOUGrid(
        voltage=voltage,
        drive=drive,
        dv=dv,
        du=du,
        reset_index=reset_index,
        generator=generator.tocsc(),
        event_hazard=event_hazard,
    )


def stationary_lif_ou_mass(grid):
    """Solve the normalized stationary forward equation ``L p = 0``."""
    generator = grid.generator.tolil(copy=True)
    right_hand_side = np.zeros(generator.shape[0])
    normalization_row = 0
    generator[normalization_row, :] = 1.0
    right_hand_side[normalization_row] = 1.0
    mass = np.asarray(
        spsolve(generator.tocsc(), right_hand_side), dtype=float
    )
    negative_scale = max(float(np.max(np.abs(mass))), 1.0)
    if np.min(mass) < -1e-8 * negative_scale:
        raise RuntimeError("stationary Fokker--Planck solve produced negative mass")
    mass = np.maximum(mass, 0.0)
    return mass / np.sum(mass)


def build_lif_ou_absorbing_generator(grid):
    """Remove reset reinjection so threshold flux becomes absorbing loss."""
    generator = grid.generator.tolil(copy=True)
    n_drive = len(grid.drive)
    threshold_voltage_index = len(grid.voltage) - 1
    reset_start = grid.reset_index * n_drive
    threshold_start = threshold_voltage_index * n_drive
    for drive_index in range(n_drive):
        source = threshold_start + drive_index
        hazard = float(grid.event_hazard[source])
        if hazard > 0.0:
            generator[reset_start + drive_index, source] -= hazard
    return generator.tocsc()


def compute_lif_ou_return_map(fokker_planck_result):
    """Integrate the absorbing propagator into a spike-to-spike map.

    If ``B`` injects a drive distribution into the reset voltage cells and
    ``D`` reads threshold flux by drive bin, the return operator is
    ``P = -D L_abs^{-1} B``. Its subleading spectrum measures memory retained
    by the cumulative Gaussian drive from one spike to the next.
    """
    grid = fokker_planck_result.grid
    n_drive = len(grid.drive)
    n_state = grid.generator.shape[0]
    reset_start = grid.reset_index * n_drive
    injection = np.zeros((n_state, n_drive))
    injection[reset_start : reset_start + n_drive, :] = np.eye(n_drive)

    absorbing_generator = build_lif_ou_absorbing_generator(grid)
    integrated_occupancy = np.asarray(
        spsolve(absorbing_generator, -injection), dtype=float
    )
    threshold_start = (len(grid.voltage) - 1) * n_drive
    threshold_slice = slice(threshold_start, threshold_start + n_drive)
    threshold_hazard = grid.event_hazard[threshold_slice]
    transition = (
        threshold_hazard[:, None]
        * integrated_occupancy[threshold_slice, :]
    )
    negative_scale = max(float(np.max(np.abs(transition))), 1.0)
    if np.min(transition) < -1e-8 * negative_scale:
        raise RuntimeError("return operator contains negative probabilities")
    transition = np.maximum(transition, 0.0)
    raw_column_sums = np.sum(transition, axis=0)
    column_sum_error = float(np.max(np.abs(raw_column_sums - 1.0)))
    if np.min(raw_column_sums) <= 0.0:
        raise RuntimeError("return operator has an empty source column")
    transition /= raw_column_sums[None, :]

    stationary_mass = fokker_planck_result.stationary_mass
    stationary_threshold_flux = (
        threshold_hazard * stationary_mass[threshold_slice]
    )
    event_distribution = (
        stationary_threshold_flux / fokker_planck_result.mean_rate
    )
    event_distribution /= np.sum(event_distribution)
    invariance_error = float(
        np.linalg.norm(transition @ event_distribution - event_distribution, 1)
    )

    eigenvalues = np.linalg.eigvals(transition)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    subleading = complex(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0j
    spectral_radius = float(min(abs(subleading), 1.0))
    if spectral_radius <= 0.0:
        mixing_spikes = 0.0
    elif spectral_radius >= 1.0:
        mixing_spikes = np.inf
    else:
        mixing_spikes = float(-1.0 / np.log(spectral_radius))
    mixing_time = mixing_spikes / fokker_planck_result.mean_rate
    return LIFOUReturnMapResult(
        transition_matrix=transition,
        event_drive_distribution=event_distribution,
        eigenvalues=eigenvalues,
        subleading_eigenvalue=subleading,
        spectral_radius=spectral_radius,
        mixing_spikes=mixing_spikes,
        mixing_time=float(mixing_time),
        invariance_error=invariance_error,
        column_sum_error=column_sum_error,
    )


def _filter_spike_covariance(regular_covariance, mean_rate, dt, beta):
    """Convolve the point-process covariance with the synaptic Green function."""
    regular_covariance = np.asarray(regular_covariance, dtype=float)
    n_lag = len(regular_covariance)
    if n_lag < 2:
        raise ValueError("at least two covariance lags are required")
    tau = np.arange(n_lag) * float(dt)
    symmetric_tau = np.concatenate((-tau[:0:-1], tau))
    symmetric_covariance = np.concatenate(
        (regular_covariance[:0:-1], regular_covariance)
    )
    weights = np.full(len(symmetric_tau), float(dt))
    weights[[0, -1]] *= 0.5
    kernel = 0.5 * float(beta) * np.exp(
        -float(beta) * np.abs(tau[:, None] - symmetric_tau[None, :])
    )
    regular_filtered = kernel @ (weights * symmetric_covariance)
    same_event = (
        0.5
        * float(beta)
        * float(mean_rate)
        * np.exp(-float(beta) * tau)
    )
    return same_event + regular_filtered


def solve_lif_ou_fokker_planck(
    I=2.0,
    drive_variance=0.02,
    drive_decay=1.0,
    beta=1.0,
    tau_max=12.0,
    dt=0.02,
    reset=0.0,
    threshold=1.0,
    n_voltage=160,
    n_drive=81,
    drive_extent_std=5.0,
    voltage_min=0.0,
):
    """Solve the stationary LIF--OU density and two-event propagator problem."""
    if tau_max <= 0.0 or dt <= 0.0:
        raise ValueError("tau_max and dt must be positive")
    grid = build_lif_ou_generator(
        I=I,
        drive_variance=drive_variance,
        drive_decay=drive_decay,
        reset=reset,
        threshold=threshold,
        n_voltage=n_voltage,
        n_drive=n_drive,
        drive_extent_std=drive_extent_std,
        voltage_min=voltage_min,
    )
    stationary_mass = stationary_lif_ou_mass(grid)
    mean_rate = float(grid.event_hazard @ stationary_mass)

    n_drive_grid = len(grid.drive)
    threshold_slice = slice(
        (len(grid.voltage) - 1) * n_drive_grid,
        len(grid.voltage) * n_drive_grid,
    )
    threshold_flux = (
        grid.event_hazard[threshold_slice] * stationary_mass[threshold_slice]
    )
    event_source = np.zeros_like(stationary_mass)
    reset_start = grid.reset_index * n_drive_grid
    event_source[reset_start : reset_start + n_drive_grid] = threshold_flux

    n_lag = int(np.floor(float(tau_max) / float(dt))) + 1
    tau = np.arange(n_lag) * float(dt)
    propagated = expm_multiply(
        grid.generator,
        event_source,
        start=0.0,
        stop=float(tau[-1]),
        num=n_lag,
        endpoint=True,
        traceA=float(np.sum(grid.generator.diagonal())),
    )
    joint_event_density = propagated @ grid.event_hazard
    regular_covariance = joint_event_density - mean_rate**2
    spike_covariance = regular_covariance.copy()
    spike_covariance[0] += mean_rate / float(dt)
    field_covariance = _filter_spike_covariance(
        regular_covariance,
        mean_rate,
        dt,
        beta,
    )

    mass_matrix = stationary_mass.reshape(len(grid.voltage), n_drive_grid)
    drive_marginal = np.sum(mass_matrix, axis=0)
    drive_mean = float(grid.drive @ drive_marginal)
    measured_drive_variance = float(
        (grid.drive - drive_mean) ** 2 @ drive_marginal
    )
    return LIFOUFokkerPlanckResult(
        tau=tau,
        field_covariance=field_covariance,
        spike_covariance=spike_covariance,
        regular_spike_covariance=regular_covariance,
        joint_event_density=joint_event_density,
        stationary_mass=stationary_mass,
        mean_rate=mean_rate,
        drive_mean=drive_mean,
        drive_variance=measured_drive_variance,
        grid=grid,
    )


def _renewal_density_from_first_return(first_return_density, dt):
    """Solve ``m = f + f * m`` by causal trapezoid-free grid quadrature."""
    first_return_density = np.asarray(first_return_density, dtype=float)
    renewal_density = np.zeros_like(first_return_density)
    denominator = 1.0 - float(dt) * first_return_density[0]
    if denominator <= 0.0:
        raise RuntimeError("first-return grid is too coarse for renewal quadrature")
    renewal_density[0] = first_return_density[0] / denominator
    for index in range(1, len(first_return_density)):
        convolution = np.dot(
            first_return_density[1 : index + 1],
            renewal_density[index - 1 :: -1],
        )
        renewal_density[index] = (
            first_return_density[index] + float(dt) * convolution
        ) / denominator
    return renewal_density


def solve_lif_ou_first_return_renewal(
    I=2.0,
    drive_variance=0.02,
    drive_decay=1.0,
    beta=1.0,
    tau_max=12.0,
    dt=0.02,
    reset=0.0,
    threshold=1.0,
    n_voltage=160,
    n_drive=81,
    drive_extent_std=5.0,
    voltage_min=0.0,
):
    """Approximate event correlations from the marginal first-return law.

    The first-return density is computed from the absorbing LIF--OU generator,
    initialized with the stationary drive distribution seen at spike times.
    The renewal equation then treats successive intervals as independent.  The
    FPT marginal is therefore resolved by the same state-space quadrature as
    the full propagator; only serial dependence between intervals is discarded.
    """
    full_result = solve_lif_ou_fokker_planck(
        I=I,
        drive_variance=drive_variance,
        drive_decay=drive_decay,
        beta=beta,
        tau_max=tau_max,
        dt=dt,
        reset=reset,
        threshold=threshold,
        n_voltage=n_voltage,
        n_drive=n_drive,
        drive_extent_std=drive_extent_std,
        voltage_min=voltage_min,
    )
    grid = full_result.grid
    mean_rate = float(full_result.mean_rate)
    if mean_rate <= 0.0:
        raise RuntimeError("stationary threshold flux must be positive")

    n_drive_grid = len(grid.drive)
    threshold_start = (len(grid.voltage) - 1) * n_drive_grid
    threshold_slice = slice(threshold_start, threshold_start + n_drive_grid)
    threshold_flux = (
        grid.event_hazard[threshold_slice]
        * full_result.stationary_mass[threshold_slice]
    )
    post_spike_mass = np.zeros_like(full_result.stationary_mass)
    reset_start = grid.reset_index * n_drive_grid
    post_spike_mass[reset_start : reset_start + n_drive_grid] = (
        threshold_flux / mean_rate
    )

    absorbing_generator = build_lif_ou_absorbing_generator(grid)
    propagated = expm_multiply(
        absorbing_generator,
        post_spike_mass,
        start=0.0,
        stop=float(full_result.tau[-1]),
        num=len(full_result.tau),
        endpoint=True,
        traceA=float(np.sum(absorbing_generator.diagonal())),
    )
    survival = np.maximum(np.sum(propagated, axis=1), 0.0)
    first_return_density = np.maximum(propagated @ grid.event_hazard, 0.0)
    renewal_density = _renewal_density_from_first_return(first_return_density, dt)
    regular_covariance = mean_rate * renewal_density - mean_rate**2
    spike_covariance = regular_covariance.copy()
    spike_covariance[0] += mean_rate / float(dt)
    field_covariance = _filter_spike_covariance(
        regular_covariance,
        mean_rate,
        dt,
        beta,
    )

    return_probability = float(
        np.trapezoid(first_return_density, full_result.tau)
    )
    interval_first_moment = float(
        np.trapezoid(full_result.tau * first_return_density, full_result.tau)
    )
    interval_second_moment = float(
        np.trapezoid(
            full_result.tau**2 * first_return_density,
            full_result.tau,
        )
    )
    normalization = max(return_probability, 1e-15)
    mean_interval = interval_first_moment / normalization
    interval_variance = max(
        interval_second_moment / normalization - mean_interval**2,
        0.0,
    )
    interval_cv = np.sqrt(interval_variance) / max(mean_interval, 1e-15)
    return LIFOUFirstReturnRenewalResult(
        tau=full_result.tau,
        first_return_density=first_return_density,
        survival_probability=survival,
        renewal_density=renewal_density,
        spike_covariance=spike_covariance,
        regular_spike_covariance=regular_covariance,
        field_covariance=field_covariance,
        mean_rate=mean_rate,
        mean_interval=mean_interval,
        interval_cv=float(interval_cv),
        return_probability=return_probability,
        full_fokker_planck=full_result,
    )


def _project_covariance_to_ou(tau, covariance, fit_tau_max=None):
    """Project a stationary covariance onto ``variance*exp(-decay*|tau|)``."""
    tau = np.asarray(tau, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    variance = float(covariance[0])
    if variance <= 0.0:
        raise RuntimeError("OU projection requires positive equal-time covariance")
    selected = np.ones(len(tau), dtype=bool)
    if fit_tau_max is not None:
        selected &= tau <= float(fit_tau_max)
    selected[0] = False
    selected_tau = tau[selected]
    selected_covariance = covariance[selected] / variance
    if not len(selected_tau):
        raise ValueError("OU projection interval contains no positive lags")

    def objective(log_decay):
        model = np.exp(-np.exp(log_decay) * selected_tau)
        return float(np.mean((model - selected_covariance) ** 2))

    optimum = minimize_scalar(
        objective,
        bounds=(np.log(1e-3), np.log(1e3)),
        method="bounded",
    )
    decay = float(np.exp(optimum.x))
    projected = variance * np.exp(-decay * tau)
    error = float(
        np.linalg.norm(projected[selected] - covariance[selected])
        / max(np.linalg.norm(covariance[selected]), 1e-15)
    )
    return variance, decay, projected, error


def solve_projected_lif_ou_dmft(
    sigma=0.5,
    I=2.0,
    beta=1.0,
    initial_drive_variance=0.02,
    initial_drive_decay=1.0,
    tau_max=8.0,
    dt=0.03,
    fit_tau_max=None,
    max_iter=20,
    mixing=0.35,
    tolerance=2e-3,
    n_voltage=120,
    n_drive=71,
    drive_extent_std=5.0,
):
    """Test a one-OU-mode projection of the self-consistent LIF DMFT map.

    This is a diagnostic Markov truncation, not the unrestricted DMFT theory.
    The exact Fokker--Planck output covariance is multiplied by ``sigma**2``
    and least-squares projected onto a single exponential before the next
    iteration.  ``projection_errors`` quantifies information discarded by that
    one-mode ansatz.
    """
    if sigma <= 0.0:
        raise ValueError("the projected OU diagnostic requires sigma > 0")
    if not 0.0 < mixing <= 1.0:
        raise ValueError("mixing must lie in (0, 1]")
    drive_variance = float(initial_drive_variance)
    drive_decay = float(initial_drive_decay)
    parameter_residuals = []
    projection_errors = []
    converged = False

    for _ in range(int(max(1, max_iter))):
        fp_result = solve_lif_ou_fokker_planck(
            I=I,
            drive_variance=drive_variance,
            drive_decay=drive_decay,
            beta=beta,
            tau_max=tau_max,
            dt=dt,
            n_voltage=n_voltage,
            n_drive=n_drive,
            drive_extent_std=drive_extent_std,
        )
        dmft_covariance = float(sigma) ** 2 * fp_result.field_covariance
        (
            proposed_variance,
            proposed_decay,
            _proposed_covariance,
            projection_error,
        ) = _project_covariance_to_ou(
            fp_result.tau,
            dmft_covariance,
            fit_tau_max=fit_tau_max,
        )
        residual = float(
            np.hypot(
                (proposed_variance - drive_variance)
                / max(drive_variance, 1e-15),
                np.log(proposed_decay / drive_decay),
            )
        )
        parameter_residuals.append(residual)
        projection_errors.append(projection_error)
        drive_variance = (
            (1.0 - mixing) * drive_variance + mixing * proposed_variance
        )
        drive_decay = float(
            np.exp(
                (1.0 - mixing) * np.log(drive_decay)
                + mixing * np.log(proposed_decay)
            )
        )
        if residual < tolerance:
            converged = True
            break

    fp_result = solve_lif_ou_fokker_planck(
        I=I,
        drive_variance=drive_variance,
        drive_decay=drive_decay,
        beta=beta,
        tau_max=tau_max,
        dt=dt,
        n_voltage=n_voltage,
        n_drive=n_drive,
        drive_extent_std=drive_extent_std,
    )
    dmft_covariance = float(sigma) ** 2 * fp_result.field_covariance
    projected_covariance = drive_variance * np.exp(-drive_decay * fp_result.tau)
    return ProjectedLIFOUDMFTResult(
        drive_variance=drive_variance,
        drive_decay=drive_decay,
        tau=fp_result.tau,
        projected_covariance=projected_covariance,
        dmft_covariance=dmft_covariance,
        parameter_residuals=np.asarray(parameter_residuals),
        projection_errors=np.asarray(projection_errors),
        converged=converged,
        fokker_planck=fp_result,
    )


def validate_lif_ou_fokker_planck(
    I=2.0,
    drive_variance=0.02,
    drive_decay=1.0,
    beta=1.0,
    tau_max=8.0,
    dt=0.02,
    n_time=8192,
    n_paths=512,
    n_voltage=240,
    n_drive=121,
    seed=90210,
):
    """Compare density/propagator quadrature with direct paths for the same OU law."""
    from rn_phase import _gaussian_process_from_spectrum, _phase_spike_spectrum

    n_time = int(max(512, n_time))
    n_paths = int(max(8, n_paths))
    rng = np.random.default_rng(seed)
    period = n_time * float(dt)
    lag = np.minimum(np.arange(n_time), n_time - np.arange(n_time)) * float(dt)
    periodic_covariance = float(drive_variance) * (
        np.exp(-float(drive_decay) * lag)
        + np.exp(-float(drive_decay) * (period - lag))
    ) / (1.0 + np.exp(-float(drive_decay) * period))
    spectrum = np.maximum(
        np.real(np.fft.rfft(periodic_covariance)), 0.0
    )
    n_frequency = n_time // 2 + 1
    normals = (
        rng.normal(size=(n_paths, n_frequency))
        + 1j * rng.normal(size=(n_paths, n_frequency))
    ) / np.sqrt(2.0)
    normals[:, 0] = rng.normal(size=n_paths)
    if n_time % 2 == 0:
        normals[:, -1] = rng.normal(size=n_paths)
    drive = _gaussian_process_from_spectrum(spectrum, normals, n_time)
    initial_voltage = rng.uniform(0.0, 1.0, n_paths)
    (
        filtered_spectrum,
        path_spike_covariance,
        path_rate,
        _sample_rates,
        _density_covariance,
    ) = _phase_spike_spectrum(
        drive,
        I,
        1.0,
        dt,
        initial_voltage,
        phase_model="lif",
        warmup_cycles=2,
        filter_beta=beta,
    )
    path_covariance = np.fft.irfft(filtered_spectrum, n=n_time)
    quadrature = solve_lif_ou_fokker_planck(
        I=I,
        drive_variance=drive_variance,
        drive_decay=drive_decay,
        beta=beta,
        tau_max=tau_max,
        dt=dt,
        n_voltage=n_voltage,
        n_drive=n_drive,
    )
    n_lag = len(quadrature.tau)
    path_covariance = path_covariance[:n_lag]
    path_spike_covariance = path_spike_covariance[:n_lag]
    covariance_relative_error = float(
        np.linalg.norm(quadrature.field_covariance - path_covariance)
        / max(np.linalg.norm(path_covariance), 1e-15)
    )
    equal_time_relative_error = float(
        abs(quadrature.field_covariance[0] - path_covariance[0])
        / max(abs(path_covariance[0]), 1e-15)
    )
    rate_relative_error = float(
        abs(quadrature.mean_rate - path_rate) / max(abs(path_rate), 1e-15)
    )
    return LIFOUValidationResult(
        tau=quadrature.tau,
        quadrature_covariance=quadrature.field_covariance,
        path_covariance=path_covariance,
        quadrature_spike_covariance=quadrature.spike_covariance,
        path_spike_covariance=path_spike_covariance,
        quadrature_rate=quadrature.mean_rate,
        path_rate=float(path_rate),
        covariance_relative_error=covariance_relative_error,
        equal_time_relative_error=equal_time_relative_error,
        rate_relative_error=rate_relative_error,
    )
