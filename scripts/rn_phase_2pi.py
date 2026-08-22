"""Gaussian four-field propagators and Wick closures for the phase model.

At prescribed Q these routines discretize the Gaussian 2PI generalized
activity equations.  The homogeneous solver closes Q with the threshold-flux
Wick moment.  Q and Qhat fluctuations belong to the next finite-N order and
are outside this leading Gaussian calculation.  The older dense fixed-Q
iteration is retained only as a discretization diagnostic.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedQPropagators:
    """A discretized Hessian, its propagator, and block metadata."""

    hessian: np.ndarray
    covariance: np.ndarray
    blocks: dict
    left_residual: float
    right_residual: float
    dt: float
    dv: float
    n_time: int
    n_phase: int


@dataclass(frozen=True)
class FixedQGaussianSolution:
    """Four-field Gaussian propagators with an iterated bilocal kernel."""

    propagators: FixedQPropagators
    flux_kernel: np.ndarray
    mean_rate: np.ndarray
    residual_history: np.ndarray
    converged: bool


@dataclass(frozen=True)
class UniformFixedQGaussianSolution:
    """Structured homogeneous reduction of the fixed-Q diagnostic."""

    C11: np.ndarray
    C33_threshold: np.ndarray
    C13: np.ndarray
    C31: np.ndarray
    flux_kernel: np.ndarray
    mean_rate: np.ndarray
    retarded_u: np.ndarray
    phase_evolution: np.ndarray
    residual_history: np.ndarray
    converged: bool
    dt: float
    dv: float


@dataclass(frozen=True)
class FixedQGaussianStability:
    """Leading feedback eigenvalue of the fixed-Q diagnostic."""

    unit_sigma_eigenvalue: float
    critical_sigma: float
    eigenmatrix: np.ndarray
    residual: float
    iterations: int


@dataclass(frozen=True)
class UniformGaussian2PISolution:
    """Homogeneous Hartree/Wick solution with a consistent bilocal kernel."""

    C11: np.ndarray
    C33_threshold: np.ndarray
    C13: np.ndarray
    C31: np.ndarray
    flux_kernel: np.ndarray
    mean_rate: np.ndarray
    retarded_u: np.ndarray
    phase_evolution: np.ndarray
    residual_history: np.ndarray
    converged: bool
    stable: bool
    physical: bool
    feedback_eigenvalue: float
    min_covariance_eigenvalue: float
    min_flux_eigenvalue: float
    max_normalized_covariance: float
    dt: float
    dv: float


@dataclass(frozen=True)
class UniformGaussian2PIStability:
    """Perron eigenpair of the positive Hartree/Wick feedback operator."""

    unit_sigma_eigenvalue: float
    critical_sigma: float
    eigenmatrix: np.ndarray
    residual: float
    iterations: int


@dataclass(frozen=True)
class StationaryGaussianAdvectionSolution:
    """Self-consistent Gaussian-advection closure of C11 and C33."""

    C11_lag: np.ndarray
    C33_flux_lag: np.ndarray
    C33_binned_lag: np.ndarray
    flux_kernel_lag: np.ndarray
    integrated_drive_variance: np.ndarray
    residual_history: np.ndarray
    converged: bool
    physical: bool
    min_flux_spectrum: float
    dt: float
    phase_bin_width: float


def _retarded_time_operator(n_time, dt, decay_rate):
    operator = np.zeros((n_time, n_time))
    diagonal = 1.0 / dt + decay_rate
    for time in range(n_time):
        operator[time, time] = diagonal
        if time:
            operator[time, time - 1] = -1.0 / dt
    return operator


def _retarded_time_response(n_time, dt, decay_rate):
    """Analytic inverse of the backward-Euler retarded time operator."""
    decay = 1.0 / (1.0 + decay_rate * dt)
    gain = dt * decay
    row = np.arange(n_time)[:, None]
    column = np.arange(n_time)[None, :]
    lag = row - column
    return np.where(lag >= 0, gain * decay**lag, 0.0)


def _filter_two_time(kernel, dt, decay_rate):
    """Apply the retarded filter in both time arguments in O(n_time^2)."""
    from scipy.signal import lfilter

    decay = 1.0 / (1.0 + decay_rate * dt)
    gain = dt * decay
    filtered = lfilter([gain], [1.0, -decay], kernel, axis=0)
    return lfilter([gain], [1.0, -decay], filtered, axis=1)


def _periodic_upwind_derivative(n_phase, dv):
    derivative = np.zeros((n_phase, n_phase))
    for phase in range(n_phase):
        derivative[phase, phase] = 1.0 / dv
        derivative[phase, (phase - 1) % n_phase] = -1.0 / dv
    return derivative


def _spectral_phase_step(n_phase, dv, displacement):
    if n_phase % 2 == 0:
        raise ValueError(
            "spectral transport requires an odd n_phase to avoid the "
            "unpaired Nyquist mode"
        )
    wave_number = 2.0 * np.pi * np.fft.fftfreq(n_phase, d=dv)
    identity_spectrum = np.fft.fft(np.eye(n_phase), axis=0)
    step = np.fft.ifft(
        np.exp(-1j * wave_number * displacement)[:, None]
        * identity_spectrum,
        axis=0,
    )
    return np.real_if_close(step, tol=1000).real


def _retarded_transport_operator(
    n_time, n_phase, dt, velocity, dv, scheme="upwind"
):
    velocity = np.broadcast_to(
        np.asarray(velocity, dtype=float), (n_time, n_phase)
    )
    scheme = str(scheme).lower()
    if scheme == "spectral":
        if not np.allclose(velocity, velocity[0, 0]):
            raise ValueError("spectral transport requires constant velocity")
        phase_step = _spectral_phase_step(
            n_phase, dv, float(velocity[0, 0]) * dt
        )
        size = n_time * n_phase
        operator = np.zeros((size, size))
        identity = np.eye(n_phase)
        for time in range(n_time):
            rows = slice(time * n_phase, (time + 1) * n_phase)
            operator[rows, rows] = identity
            if time:
                previous = slice((time - 1) * n_phase, time * n_phase)
                operator[rows, previous] = -phase_step
        return operator
    if scheme != "upwind":
        raise ValueError("transport_scheme must be 'upwind' or 'spectral'")
    backward = _periodic_upwind_derivative(n_phase, dv)
    size = n_time * n_phase
    operator = np.zeros((size, size))
    identity = np.eye(n_phase)
    for time in range(n_time):
        rows = slice(time * n_phase, (time + 1) * n_phase)
        operator[rows, rows] = identity / dt + velocity[time, :, None] * backward
        if time:
            previous = slice((time - 1) * n_phase, time * n_phase)
            operator[rows, previous] = -identity / dt
    return operator


def _field_blocks(n_time, n_phase):
    n_density = n_time * n_phase
    stops = np.cumsum((0, n_time, n_time, n_density, n_density))
    names = ("u", "u_response", "density", "density_response")
    return {
        name: slice(int(stops[index]), int(stops[index + 1]))
        for index, name in enumerate(names)
    }


def uniform_phase_covariance(n_phase, circumference=2.0 * np.pi):
    """Connected covariance of an iid point phase on a finite-volume grid."""
    dv = circumference / n_phase
    mean_density = 1.0 / circumference
    return (
        np.eye(n_phase) * mean_density / dv
        - np.full((n_phase, n_phase), mean_density**2)
    )


def free_phase_flux_covariance(
    n_time, dt, velocity=1.0, circumference=2.0 * np.pi
):
    """Exact time-bin covariance of a uniformly initialized free oscillator."""
    width = min(abs(float(velocity)) * dt, circumference)
    rate = abs(float(velocity)) / circumference
    times = np.arange(n_time) * dt
    separation = np.mod(
        abs(float(velocity)) * (times[:, None] - times[None, :]),
        circumference,
    )
    overlap = np.maximum(0.0, width - separation)
    overlap += np.maximum(0.0, width - (circumference - separation))
    return overlap / (circumference * dt**2) - rate**2


def _integrated_stationary_covariance(covariance, dt):
    """Return int_0^tau (tau-s) C(s) ds on a nonnegative lag grid."""
    covariance = np.asarray(covariance, dtype=float)
    integral = np.zeros_like(covariance)
    integral[1:] = np.cumsum(
        0.5 * (covariance[:-1] + covariance[1:]) * dt
    )
    integrated = np.zeros_like(covariance)
    integrated[1:] = np.cumsum(
        0.5 * (integral[:-1] + integral[1:]) * dt
    )
    scale = max(float(np.max(np.abs(integrated))), 1.0)
    integrated[integrated < 0.0] = np.maximum(
        integrated[integrated < 0.0], -1e-12 * scale
    )
    return np.maximum(integrated, 0.0)


def _gaussian_advected_density_covariance(
    tau,
    integrated_variance,
    *,
    F0,
    F1,
    circumference,
    n_modes,
    bin_width=None,
):
    """Threshold-density covariance under additive stationary Gaussian drive.

    With a uniform initial phase, each Fourier mode is dephased by the exact
    Gaussian characteristic function of the integrated drive.  ``bin_width``
    selects a normalized top-hat threshold observable.  When it is None, the
    band-limited phase-grid density used by the Hartree solver is returned.
    """
    tau = np.asarray(tau, dtype=float)
    integrated_variance = np.asarray(integrated_variance, dtype=float)
    wave_unit = 2.0 * np.pi / circumference
    covariance = np.zeros_like(tau)
    for mode in range(1, int(n_modes) + 1):
        wave_number = wave_unit * mode
        coefficient = 1.0
        if bin_width is not None:
            coefficient = np.sinc(mode * float(bin_width) / circumference) ** 2
        covariance += coefficient * np.cos(wave_number * F0 * tau) * np.exp(
            -(wave_number * F1) ** 2 * integrated_variance
        )
    return 2.0 * covariance / circumference**2


def solve_stationary_phase_gaussian_advection(
    *,
    n_periods=12,
    n_phase=129,
    beta=1.0,
    sigma=1.0,
    F0=1.0,
    F1=1.0,
    circumference=2.0 * np.pi,
    phase_bin_width=0.25,
    max_iter=400,
    mixing=0.15,
    tolerance=1e-8,
):
    """Solve the stationary Gaussian-advection 2PI closure.

    The density modes are advected exactly conditional on an additive
    Gaussian drive. Gaussian integration by parts then makes C33 a memory
    functional of C11. The threshold-flux kernel is closed with the same Wick
    moment as the homogeneous Gaussian 2PI approximation and filtered through
    the causal synaptic response.
    """
    n_phase = int(n_phase)
    n_periods = int(n_periods)
    if n_phase < 3 or n_phase % 2 == 0:
        raise ValueError("Gaussian advection requires odd n_phase >= 3")
    if n_periods < 4:
        raise ValueError("n_periods must be at least 4")
    if F0 <= 0.0 or beta <= 0.0 or circumference <= 0.0:
        raise ValueError("F0, beta, and circumference must be positive")
    if not 0.0 < phase_bin_width <= circumference:
        raise ValueError("phase_bin_width must lie in (0, circumference]")
    if n_periods % 2:
        n_periods += 1

    dt = circumference / (F0 * n_phase)
    n_time = n_periods * n_phase
    half = n_time // 2
    tau_positive = np.arange(half + 1) * dt
    lag_index = np.minimum(np.arange(n_time), n_time - np.arange(n_time))
    n_modes = (n_phase - 1) // 2
    mean_density = 1.0 / circumference

    omega = 2.0 * np.pi * np.fft.rfftfreq(n_time, d=dt)
    decay = np.exp(-beta * dt)
    synaptic_filter = (beta * dt) ** 2 / np.maximum(
        1.0 + decay**2 - 2.0 * decay * np.cos(omega * dt), 1e-30
    )
    drive_spectrum = np.zeros(len(omega))
    residual_history = []
    converged = sigma == 0.0
    physical = True
    min_flux_spectrum = 0.0

    for _iteration in range(int(max_iter)):
        C11_circular = np.fft.irfft(drive_spectrum, n=n_time)
        integrated = _integrated_stationary_covariance(
            C11_circular[:half + 1], dt
        )
        C33_flux_positive = _gaussian_advected_density_covariance(
            tau_positive,
            integrated,
            F0=F0,
            F1=F1,
            circumference=circumference,
            n_modes=n_modes,
        )
        C33_flux = C33_flux_positive[lag_index]
        flux_kernel = (
            F0**2 * C33_flux
            + F1**2 * (mean_density**2 + C33_flux) * C11_circular
        )
        flux_spectrum = np.fft.rfft(flux_kernel).real
        min_flux_spectrum = float(np.min(flux_spectrum))
        flux_scale = max(float(np.max(np.abs(flux_spectrum))), 1.0)
        if min_flux_spectrum < -1e-8 * flux_scale:
            physical = False
        proposed = sigma**2 * synaptic_filter * np.maximum(flux_spectrum, 0.0)
        proposed[0] = 0.0
        residual = float(
            np.linalg.norm(proposed - drive_spectrum)
            / max(np.linalg.norm(proposed), 1e-12)
        )
        residual_history.append(residual)
        drive_spectrum = (
            (1.0 - mixing) * drive_spectrum + mixing * proposed
        )
        if residual < tolerance:
            converged = True
            break

    C11_circular = np.fft.irfft(drive_spectrum, n=n_time)
    C11_positive = C11_circular[:half + 1]
    integrated = _integrated_stationary_covariance(C11_positive, dt)
    C33_flux_positive = _gaussian_advected_density_covariance(
        tau_positive,
        integrated,
        F0=F0,
        F1=F1,
        circumference=circumference,
        n_modes=n_modes,
    )
    C33_binned_positive = _gaussian_advected_density_covariance(
        tau_positive,
        integrated,
        F0=F0,
        F1=F1,
        circumference=circumference,
        n_modes=max(n_modes, int(np.ceil(circumference / phase_bin_width)) * 4),
        bin_width=phase_bin_width,
    )
    C33_flux = C33_flux_positive[lag_index]
    flux_kernel = (
        F0**2 * C33_flux
        + F1**2 * (mean_density**2 + C33_flux) * C11_circular
    )
    flux_spectrum = np.fft.rfft(flux_kernel).real
    min_flux_spectrum = float(np.min(flux_spectrum))
    flux_scale = max(float(np.max(np.abs(flux_spectrum))), 1.0)
    physical = bool(physical and min_flux_spectrum >= -1e-8 * flux_scale)
    return StationaryGaussianAdvectionSolution(
        C11_lag=C11_positive,
        C33_flux_lag=C33_flux_positive,
        C33_binned_lag=C33_binned_positive,
        flux_kernel_lag=flux_kernel[:half + 1],
        integrated_drive_variance=integrated,
        residual_history=np.asarray(residual_history),
        converged=converged,
        physical=physical,
        min_flux_spectrum=min_flux_spectrum,
        dt=float(dt),
        phase_bin_width=float(phase_bin_width),
    )


def solve_phase_fixed_q_propagators(
    flux_kernel,
    *,
    beta=1.0,
    sigma=1.0,
    velocity=1.0,
    coupling_gradient=None,
    dt=0.1,
    n_phase=16,
    circumference=2.0 * np.pi,
    initial_phase_covariance=None,
    transport_scheme="upwind",
):
    """Invert every block of the Gaussian phase-model Hessian.

    ``flux_kernel`` is the row-sum-corrected threshold-flux covariance Q(t,s).
    The returned matrix uses field ordering (u, u_response, density,
    density_response).  Random initial phases enter through the boundary
    response-response block and are therefore propagated by the same causal
    transport operator as the other density sectors.
    """
    flux_kernel = np.asarray(flux_kernel, dtype=float)
    if flux_kernel.ndim != 2 or flux_kernel.shape[0] != flux_kernel.shape[1]:
        raise ValueError("flux_kernel must be a square two-time matrix")
    n_time = flux_kernel.shape[0]
    if not np.allclose(flux_kernel, flux_kernel.T, atol=1e-10):
        raise ValueError("flux_kernel must be symmetric")
    if n_phase < 3:
        raise ValueError("n_phase must be at least 3")
    if dt <= 0.0 or circumference <= 0.0:
        raise ValueError("dt and circumference must be positive")

    dv = circumference / n_phase
    retarded_u = _retarded_time_operator(n_time, dt, beta)
    advanced_u = retarded_u.T
    retarded_density = _retarded_transport_operator(
        n_time, n_phase, dt, velocity, dv, scheme=transport_scheme
    )
    advanced_density = retarded_density.T

    if coupling_gradient is None:
        coupling_gradient = np.zeros((n_time, n_phase))
    coupling_gradient = np.broadcast_to(
        np.asarray(coupling_gradient, dtype=float), (n_time, n_phase)
    )
    density_to_drive = np.zeros((n_time, n_time * n_phase))
    for time in range(n_time):
        density_to_drive[
            time, time * n_phase:(time + 1) * n_phase
        ] = coupling_gradient[time]
    drive_to_density = density_to_drive.T

    blocks = _field_blocks(n_time, n_phase)
    total_size = 2 * n_time + 2 * n_time * n_phase
    hessian = np.zeros((total_size, total_size))
    u = blocks["u"]
    ur = blocks["u_response"]
    density = blocks["density"]
    density_response = blocks["density_response"]
    hessian[u, ur] = advanced_u
    hessian[ur, u] = retarded_u
    hessian[u, density_response] = density_to_drive
    hessian[density_response, u] = drive_to_density
    hessian[density, density_response] = advanced_density
    hessian[density_response, density] = retarded_density
    hessian[ur, ur] = -beta**2 * sigma**2 * flux_kernel

    if initial_phase_covariance is None:
        initial_phase_covariance = uniform_phase_covariance(
            n_phase, circumference
        )
    initial_phase_covariance = np.asarray(
        initial_phase_covariance, dtype=float
    )
    if initial_phase_covariance.shape != (n_phase, n_phase):
        raise ValueError(
            "initial_phase_covariance must have shape (n_phase, n_phase)"
        )
    first_transport = retarded_density[:n_phase, :n_phase]
    boundary_noise = (
        first_transport
        @ initial_phase_covariance
        @ first_transport.T
    )
    boundary = slice(
        density_response.start,
        density_response.start + n_phase,
    )
    hessian[boundary, boundary] = -boundary_noise

    covariance = np.linalg.inv(hessian)
    identity = np.eye(total_size)
    scale = max(np.linalg.norm(identity), 1.0)
    left_residual = np.linalg.norm(hessian @ covariance - identity) / scale
    right_residual = np.linalg.norm(covariance @ hessian - identity) / scale
    return FixedQPropagators(
        hessian=hessian,
        covariance=covariance,
        blocks=blocks,
        left_residual=float(left_residual),
        right_residual=float(right_residual),
        dt=float(dt),
        dv=float(dv),
        n_time=n_time,
        n_phase=int(n_phase),
    )


def propagator_block(solution, row, column):
    """Return one named block from a fixed-Q propagator solution."""
    return solution.covariance[
        solution.blocks[row], solution.blocks[column]
    ]


def fixed_q_linear_flux_kernel(
    solution,
    *,
    F0=1.0,
    F1=1.0,
    mean_density=None,
    threshold_index=None,
    replace_diagonal=True,
):
    """Evaluate the Gaussian Wick equation for F(u)=F0+F1*u."""
    n_time = solution.n_time
    n_phase = solution.n_phase
    if threshold_index is None:
        threshold_index = n_phase - 1
    threshold_index = int(threshold_index) % n_phase
    if mean_density is None:
        mean_density = np.full(n_time, 1.0 / (n_phase * solution.dv))
    mean_density = np.broadcast_to(
        np.asarray(mean_density, dtype=float), (n_time,)
    )
    threshold = np.arange(n_time) * n_phase + threshold_index

    C11 = propagator_block(solution, "u", "u")
    C13 = propagator_block(solution, "u", "density")[:, threshold]
    C31 = propagator_block(solution, "density", "u")[threshold, :]
    C33 = propagator_block(solution, "density", "density")[
        np.ix_(threshold, threshold)
    ]
    equal_mixed = C13[np.arange(n_time), np.arange(n_time)]
    mean_rate = F0 * mean_density + F1 * equal_mixed

    density_t = mean_density[:, None]
    density_s = mean_density[None, :]
    kernel = F0**2 * C33
    kernel += F0 * F1 * (density_s * C31 + density_t * C13)
    kernel += F1**2 * (
        density_t * density_s * C11
        + C11 * C33
        + C13 * C31
    )
    kernel = 0.5 * (kernel + kernel.T)
    if replace_diagonal:
        diagonal = mean_rate / solution.dt - mean_rate**2
        kernel[np.diag_indices(n_time)] = diagonal
    return kernel, mean_rate


def solve_phase_fixed_q_gaussian(
    *,
    n_time=12,
    n_phase=12,
    dt=0.1,
    beta=1.0,
    sigma=1.0,
    F0=1.0,
    F1=1.0,
    circumference=2.0 * np.pi,
    coupling_gradient=None,
    initial_phase_covariance=None,
    max_iter=40,
    mixing=0.25,
    tolerance=1e-7,
    transport_scheme="upwind",
):
    """Iterate the four-field Gaussian propagators and an external Q closure.

    This dense reference implementation prioritizes direct residual checks.
    It is intended for derivation tests and small grids; the paper-scale
    implementation will use the same equations with structured causal solves.
    """
    n_time = int(n_time)
    n_phase = int(n_phase)
    mean_rate = F0 / circumference
    flux_kernel = np.eye(n_time) * (
        mean_rate / dt - mean_rate**2
    )
    residual_history = []
    converged = False
    propagators = None
    rate = np.full(n_time, mean_rate)

    for _iteration in range(int(max_iter)):
        propagators = solve_phase_fixed_q_propagators(
            flux_kernel,
            beta=beta,
            sigma=sigma,
            velocity=F0,
            coupling_gradient=coupling_gradient,
            dt=dt,
            n_phase=n_phase,
            circumference=circumference,
            initial_phase_covariance=initial_phase_covariance,
            transport_scheme=transport_scheme,
        )
        proposed, rate = fixed_q_linear_flux_kernel(
            propagators,
            F0=F0,
            F1=F1,
        )
        scale = max(np.linalg.norm(flux_kernel), 1e-12)
        residual = float(np.linalg.norm(proposed - flux_kernel) / scale)
        residual_history.append(residual)
        flux_kernel = (1.0 - mixing) * flux_kernel + mixing * proposed
        flux_kernel = 0.5 * (flux_kernel + flux_kernel.T)
        if residual < tolerance:
            converged = True
            break

    propagators = solve_phase_fixed_q_propagators(
        flux_kernel,
        beta=beta,
        sigma=sigma,
        velocity=F0,
        coupling_gradient=coupling_gradient,
        dt=dt,
        n_phase=n_phase,
        circumference=circumference,
        initial_phase_covariance=initial_phase_covariance,
        transport_scheme=transport_scheme,
    )
    final_kernel, rate = fixed_q_linear_flux_kernel(
        propagators,
        F0=F0,
        F1=F1,
    )
    final_residual = float(
        np.linalg.norm(final_kernel - flux_kernel)
        / max(np.linalg.norm(flux_kernel), 1e-12)
    )
    if not residual_history or final_residual != residual_history[-1]:
        residual_history.append(final_residual)
    converged = converged or final_residual < tolerance
    return FixedQGaussianSolution(
        propagators=propagators,
        flux_kernel=flux_kernel,
        mean_rate=rate,
        residual_history=np.asarray(residual_history),
        converged=converged,
    )


def solve_uniform_phase_fixed_q_gaussian(
    *,
    n_time=512,
    n_phase=65,
    dt=0.05,
    beta=1.0,
    sigma=1.0,
    F0=1.0,
    F1=1.0,
    circumference=2.0 * np.pi,
    initial_phase_covariance=None,
    max_iter=200,
    mixing=0.2,
    tolerance=1e-8,
    transport_scheme="spectral",
    threshold_regularization="time_bin",
):
    """Solve the homogeneous fixed-Q diagnostic on a structured grid.

    Uniform phase density and phase-independent velocity imply B=0.  The full
    propagator equations then give C13=C31=0 exactly, while C33 is transported
    from its initial covariance and C11 is closed through the bilocal Wick
    equation.  No propagator equation is dropped before making this symmetry
    reduction.
    """
    n_time = int(n_time)
    n_phase = int(n_phase)
    dv = circumference / n_phase
    if initial_phase_covariance is None:
        initial_phase_covariance = uniform_phase_covariance(
            n_phase, circumference
        )
    initial_phase_covariance = np.asarray(initial_phase_covariance, dtype=float)

    retarded_operator = _retarded_time_operator(n_time, dt, beta)
    retarded_u = np.linalg.inv(retarded_operator)
    transport_scheme = str(transport_scheme).lower()
    if transport_scheme == "spectral":
        phase_step = _spectral_phase_step(n_phase, dv, F0 * dt)
    elif transport_scheme == "upwind":
        backward = _periodic_upwind_derivative(n_phase, dv)
        phase_step = np.linalg.solve(
            np.eye(n_phase) / dt + F0 * backward,
            np.eye(n_phase) / dt,
        )
    else:
        raise ValueError("transport_scheme must be 'upwind' or 'spectral'")
    phase_evolution = np.empty((n_time, n_phase, n_phase))
    phase_evolution[0] = np.eye(n_phase)
    for time in range(1, n_time):
        phase_evolution[time] = phase_step @ phase_evolution[time - 1]

    threshold_rows = phase_evolution[:, -1, :]
    phase_grid_C33 = (
        threshold_rows @ initial_phase_covariance @ threshold_rows.T
    )
    mean_density = 1.0 / circumference
    mean_rate = np.full(n_time, F0 * mean_density)
    threshold_regularization = str(threshold_regularization).lower()
    if threshold_regularization == "time_bin":
        free_flux = free_phase_flux_covariance(
            n_time, dt, velocity=F0, circumference=circumference
        )
        C33_threshold = free_flux / max(F0**2, 1e-30)
    elif threshold_regularization == "phase_grid":
        C33_threshold = phase_grid_C33
        free_flux = F0**2 * C33_threshold
        free_flux[np.diag_indices(n_time)] = (
            mean_rate / dt - mean_rate**2
        )
    else:
        raise ValueError(
            "threshold_regularization must be 'time_bin' or 'phase_grid'"
        )
    flux_kernel = free_flux.copy()

    residual_history = []
    converged = False
    C11 = np.zeros_like(flux_kernel)
    for _iteration in range(int(max_iter)):
        C11 = (
            beta**2
            * sigma**2
            * retarded_u
            @ flux_kernel
            @ retarded_u.T
        )
        proposed = (
            F0**2 * C33_threshold
            + F1**2
            * (mean_density**2 * C11 + C11 * C33_threshold)
        )
        proposed[np.diag_indices(n_time)] = np.diag(free_flux)
        proposed = 0.5 * (proposed + proposed.T)
        residual = float(
            np.linalg.norm(proposed - flux_kernel)
            / max(np.linalg.norm(flux_kernel), 1e-12)
        )
        residual_history.append(residual)
        flux_kernel = (1.0 - mixing) * flux_kernel + mixing * proposed
        if residual < tolerance:
            converged = True
            break

    C11 = (
        beta**2
        * sigma**2
        * retarded_u
        @ flux_kernel
        @ retarded_u.T
    )
    zeros = np.zeros((n_time, n_time))
    return UniformFixedQGaussianSolution(
        C11=C11,
        C33_threshold=C33_threshold,
        C13=zeros,
        C31=zeros.copy(),
        flux_kernel=flux_kernel,
        mean_rate=mean_rate,
        retarded_u=retarded_u,
        phase_evolution=phase_evolution,
        residual_history=np.asarray(residual_history),
        converged=converged,
        dt=float(dt),
        dv=float(dv),
    )


def _uniform_gaussian_components(
    *,
    n_time,
    n_phase,
    dt,
    beta,
    F0,
    circumference,
    initial_phase_covariance,
    transport_scheme,
):
    """Construct the causal responses and freely advected density covariance."""
    n_time = int(n_time)
    n_phase = int(n_phase)
    dv = circumference / n_phase
    if initial_phase_covariance is None:
        initial_phase_covariance = uniform_phase_covariance(
            n_phase, circumference
        )
    initial_phase_covariance = np.asarray(initial_phase_covariance, dtype=float)
    if initial_phase_covariance.shape != (n_phase, n_phase):
        raise ValueError(
            "initial_phase_covariance must have shape (n_phase, n_phase)"
        )

    retarded_u = _retarded_time_response(n_time, dt, beta)
    transport_scheme = str(transport_scheme).lower()
    if transport_scheme == "spectral":
        phase_step = _spectral_phase_step(n_phase, dv, F0 * dt)
    elif transport_scheme == "upwind":
        backward = _periodic_upwind_derivative(n_phase, dv)
        phase_step = np.linalg.solve(
            np.eye(n_phase) / dt + F0 * backward,
            np.eye(n_phase) / dt,
        )
    else:
        raise ValueError("transport_scheme must be 'upwind' or 'spectral'")

    phase_evolution = np.empty((n_time, n_phase, n_phase))
    phase_evolution[0] = np.eye(n_phase)
    for time in range(1, n_time):
        phase_evolution[time] = phase_step @ phase_evolution[time - 1]
    threshold_rows = phase_evolution[:, -1, :]
    C33_threshold = (
        threshold_rows @ initial_phase_covariance @ threshold_rows.T
    )
    C33_threshold = 0.5 * (C33_threshold + C33_threshold.T)
    return retarded_u, phase_evolution, C33_threshold, dv


def uniform_phase_gaussian_2pi_stability(
    *,
    n_time=512,
    n_phase=65,
    dt=0.05,
    beta=1.0,
    F0=1.0,
    F1=1.0,
    circumference=2.0 * np.pi,
    initial_phase_covariance=None,
    transport_scheme="spectral",
    max_iter=1000,
    tolerance=1e-10,
):
    """Find the stability threshold of the homogeneous Hartree/Wick branch."""
    retarded_u, _evolution, C33, _dv = _uniform_gaussian_components(
        n_time=n_time,
        n_phase=n_phase,
        dt=dt,
        beta=beta,
        F0=F0,
        circumference=circumference,
        initial_phase_covariance=initial_phase_covariance,
        transport_scheme=transport_scheme,
    )
    mean_density = 1.0 / circumference
    density_second_moment = mean_density**2 + C33

    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

    n_time = int(n_time)

    def apply(vector):
        matrix = np.asarray(vector).reshape(n_time, n_time)
        mapped = beta**2 * F1**2 * _filter_two_time(
            density_second_moment * matrix, dt, beta
        )
        return mapped.ravel()

    dimension = n_time * n_time
    operator = LinearOperator(
        (dimension, dimension), matvec=apply, dtype=float
    )
    n_eigenvalues = min(3, dimension - 2)
    try:
        eigenvalues, eigenvectors = eigs(
            operator,
            k=n_eigenvalues,
            which="LR",
            v0=np.eye(n_time).ravel(),
            tol=tolerance,
            maxiter=int(max_iter),
        )
    except ArpackNoConvergence as error:
        if error.eigenvalues is None or len(error.eigenvalues) == 0:
            raise RuntimeError(
                "Hartree/Wick stability Arnoldi solve did not converge"
            ) from error
        eigenvalues = error.eigenvalues
        eigenvectors = error.eigenvectors
    leading = int(np.argmax(eigenvalues.real))
    eigenvalue_complex = eigenvalues[leading]
    eigenvector = eigenvectors[:, leading]
    eigenvalue = float(eigenvalue_complex.real)
    eigenmatrix = np.real(eigenvector.reshape(n_time, n_time))
    eigenmatrix = 0.5 * (eigenmatrix + eigenmatrix.T)
    eigenmatrix /= max(np.linalg.norm(eigenmatrix), 1e-30)
    residual = float(
        np.linalg.norm(apply(eigenvector) - eigenvalue_complex * eigenvector)
        / max(np.linalg.norm(eigenvector), 1e-30)
    )
    critical_sigma = (
        np.inf if eigenvalue <= 0.0 else 1.0 / np.sqrt(eigenvalue)
    )
    return UniformGaussian2PIStability(
        unit_sigma_eigenvalue=eigenvalue,
        critical_sigma=float(critical_sigma),
        eigenmatrix=eigenmatrix,
        residual=residual,
        iterations=int(max_iter),
    )


def solve_uniform_phase_gaussian_2pi(
    *,
    n_time=512,
    n_phase=65,
    dt=0.05,
    beta=1.0,
    sigma=1.0,
    F0=1.0,
    F1=1.0,
    circumference=2.0 * np.pi,
    initial_phase_covariance=None,
    max_iter=1000,
    mixing=1.0,
    tolerance=1e-9,
    stability_tolerance=1e-8,
    transport_scheme="spectral",
):
    """Solve the homogeneous Hartree/Wick equations below their instability.

    The equal-time and unequal-time threshold kernel are obtained from one
    Gaussian Wick contraction using the bare phase propagator. No event
    diagonal is inserted afterward. This makes the feedback map positive by
    the Schur-product theorem, while retaining the Hartree approximation.
    """
    retarded_u, phase_evolution, C33, dv = _uniform_gaussian_components(
        n_time=n_time,
        n_phase=n_phase,
        dt=dt,
        beta=beta,
        F0=F0,
        circumference=circumference,
        initial_phase_covariance=initial_phase_covariance,
        transport_scheme=transport_scheme,
    )
    stability = uniform_phase_gaussian_2pi_stability(
        n_time=n_time,
        n_phase=n_phase,
        dt=dt,
        beta=beta,
        F0=F0,
        F1=F1,
        circumference=circumference,
        initial_phase_covariance=initial_phase_covariance,
        transport_scheme=transport_scheme,
        tolerance=stability_tolerance,
    )
    feedback_eigenvalue = float(sigma**2 * stability.unit_sigma_eigenvalue)
    stable = feedback_eigenvalue < 1.0 - stability_tolerance
    mean_density = 1.0 / circumference
    mean_rate = np.full(int(n_time), F0 * mean_density)
    density_second_moment = mean_density**2 + C33
    source_flux = F0**2 * C33
    zeros = np.zeros((int(n_time), int(n_time)))

    if not stable:
        invalid = np.full_like(C33, np.nan)
        return UniformGaussian2PISolution(
            C11=invalid,
            C33_threshold=C33,
            C13=zeros,
            C31=zeros.copy(),
            flux_kernel=invalid.copy(),
            mean_rate=mean_rate,
            retarded_u=retarded_u,
            phase_evolution=phase_evolution,
            residual_history=np.asarray([], dtype=float),
            converged=False,
            stable=False,
            physical=False,
            feedback_eigenvalue=feedback_eigenvalue,
            min_covariance_eigenvalue=np.nan,
            min_flux_eigenvalue=np.nan,
            max_normalized_covariance=np.nan,
            dt=float(dt),
            dv=float(dv),
        )

    C11 = zeros.copy()
    residual_history = []
    converged = False
    for _iteration in range(int(max_iter)):
        flux_kernel = source_flux + F1**2 * density_second_moment * C11
        proposed = beta**2 * sigma**2 * _filter_two_time(
            flux_kernel, dt, beta
        )
        proposed = 0.5 * (proposed + proposed.T)
        residual = float(
            np.linalg.norm(proposed - C11)
            / max(np.linalg.norm(proposed), 1e-12)
        )
        residual_history.append(residual)
        C11 = (1.0 - mixing) * C11 + mixing * proposed
        if residual < tolerance:
            converged = True
            break

    flux_kernel = source_flux + F1**2 * density_second_moment * C11
    flux_kernel = 0.5 * (flux_kernel + flux_kernel.T)
    covariance_eigenvalues = np.linalg.eigvalsh(C11)
    flux_eigenvalues = np.linalg.eigvalsh(flux_kernel)
    diagonal = np.diag(C11)
    denominator = np.sqrt(np.maximum(diagonal[:, None] * diagonal[None, :], 0.0))
    normalized = np.zeros_like(C11)
    valid = denominator > 1e-14
    normalized[valid] = np.abs(C11[valid]) / denominator[valid]
    min_covariance_eigenvalue = float(covariance_eigenvalues[0])
    min_flux_eigenvalue = float(flux_eigenvalues[0])
    max_normalized_covariance = float(np.max(normalized))
    covariance_scale = max(float(np.max(np.abs(diagonal))), 1.0)
    flux_scale = max(float(np.max(np.abs(np.diag(flux_kernel)))), 1.0)
    physical = bool(
        min_covariance_eigenvalue >= -1e-10 * covariance_scale
        and min_flux_eigenvalue >= -1e-10 * flux_scale
        and max_normalized_covariance <= 1.0 + 1e-8
    )
    return UniformGaussian2PISolution(
        C11=C11,
        C33_threshold=C33,
        C13=zeros,
        C31=zeros.copy(),
        flux_kernel=flux_kernel,
        mean_rate=mean_rate,
        retarded_u=retarded_u,
        phase_evolution=phase_evolution,
        residual_history=np.asarray(residual_history),
        converged=converged,
        stable=True,
        physical=physical,
        feedback_eigenvalue=feedback_eigenvalue,
        min_covariance_eigenvalue=min_covariance_eigenvalue,
        min_flux_eigenvalue=min_flux_eigenvalue,
        max_normalized_covariance=max_normalized_covariance,
        dt=float(dt),
        dv=float(dv),
    )


def uniform_phase_fixed_q_stability(
    *,
    n_time=512,
    n_phase=65,
    dt=0.05,
    beta=1.0,
    F0=1.0,
    F1=1.0,
    circumference=2.0 * np.pi,
    max_iter=1000,
    tolerance=1e-9,
):
    """Find the Gaussian covariance instability of the uniform solution."""
    reference = solve_uniform_phase_fixed_q_gaussian(
        n_time=n_time,
        n_phase=n_phase,
        dt=dt,
        beta=beta,
        sigma=0.0,
        F0=F0,
        F1=F1,
        circumference=circumference,
        max_iter=1,
    )
    multiplier = F1**2 * (
        (1.0 / circumference) ** 2 + reference.C33_threshold
    )
    multiplier = multiplier.copy()
    multiplier[np.diag_indices(n_time)] = 0.0
    response = reference.retarded_u

    from scipy.sparse.linalg import LinearOperator, eigs

    diagonal = np.diag_indices(n_time)

    def apply(vector):
        matrix = np.asarray(vector).reshape(n_time, n_time)
        mapped = beta**2 * response @ (multiplier * matrix) @ response.T
        return mapped.ravel()

    dimension = n_time * n_time
    operator = LinearOperator(
        (dimension, dimension), matvec=apply, dtype=float
    )
    initial = np.ones((n_time, n_time))
    initial[diagonal] = 0.0
    eigenvalues, eigenvectors = eigs(
        operator,
        k=1,
        which="LM",
        v0=initial.ravel(),
        tol=tolerance,
        maxiter=int(max_iter),
    )
    eigenvalue_complex = eigenvalues[0]
    eigenvector = eigenvectors[:, 0]
    residual = float(
        np.linalg.norm(apply(eigenvector) - eigenvalue_complex * eigenvector)
        / max(np.linalg.norm(eigenvector), 1e-30)
    )
    eigenvalue = float(abs(eigenvalue_complex))
    eigenmatrix = np.real(eigenvector.reshape(n_time, n_time))
    eigenmatrix /= max(np.linalg.norm(eigenmatrix), 1e-30)
    critical_sigma = (
        np.inf if eigenvalue <= 0.0 else 1.0 / np.sqrt(eigenvalue)
    )
    return FixedQGaussianStability(
        unit_sigma_eigenvalue=eigenvalue,
        critical_sigma=float(critical_sigma),
        eigenmatrix=eigenmatrix,
        residual=residual,
        iterations=int(max_iter),
    )
