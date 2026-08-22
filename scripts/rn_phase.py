"""Phase/spiking-network simulations, theory closures, and plots."""

import os

import numpy as np
import matplotlib.pyplot as plt

from rn_core import autocorr, default_results_dir, make_weights, rng
from rn_phase_2pi import (
    solve_uniform_phase_fixed_q_gaussian,
    solve_uniform_phase_gaussian_2pi,
)

# -----------------------------------------------------------------------------
# 1b. PHASE NEURON NETWORK
#     phi_i in [-pi, pi], dphi_i/dt = alpha * max(I + u_i, 0)^(1/alpha)
#     spikes when phi_i crosses pi; u_i is driven by spike input through W.
# -----------------------------------------------------------------------------

def sim_phase_network(
    N=512,
    I=1.0,
    alpha=1.0,
    sigma=7.0,
    beta=1.0,
    T=3000.0,
    dt=0.02,
    lam=1,
    burn=300,
    tau_max=50.0,
    n_probe=None,
    return_spike=False,
    return_u=False,
    return_phase_density=False,
    phase_bin_width=0.25,
    synapse_update="exact",
    rng=rng,
):
    """Simulate a phase-reset network and return C_uu(tau).

    If return_spike is True, also return C_spk(tau) estimated from the
    whole-network population spike-rate time series r(t)=N_spk(t)/(N*dt).
    This is much smoother than averaging per-neuron sparse spike trains.
    """
    W = make_weights(N, sigma, lam, rng)
    phi = rng.uniform(-np.pi, np.pi, N)
    u = np.zeros(N)
    synapse_update = str(synapse_update).lower()
    if synapse_update not in ("exact", "euler"):
        raise ValueError("synapse_update must be 'exact' or 'euler'")

    def F(u_):
        return alpha * np.clip(I + u_, 0.0, 1e12) ** (1.0 / alpha)

    def spike_weights(phi_old, rate, spike_counts):
        """Return per-neuron spike counts, optionally filtered by event time."""
        if synapse_update == "euler":
            return spike_counts
        weighted = np.zeros_like(spike_counts, dtype=float)
        spiking = np.flatnonzero(spike_counts > 0)
        for j in spiking:
            count = int(spike_counts[j])
            if count <= 0 or rate[j] <= 0.0:
                continue
            thresholds = np.pi + 2.0 * np.pi * np.arange(count)
            crossing_times = (thresholds - phi_old[j]) / rate[j]
            crossing_times = np.clip(crossing_times, 0.0, dt)
            weighted[j] = np.sum(np.exp(-beta * (dt - crossing_times)))
        return weighted

    def update_u(u_, drive_):
        # Spikes are delta events: integrating beta*W*dN gives beta*W*count,
        # not beta*W*count*dt.  The exact option also applies exponential leak
        # over the step, including within-step decay from estimated spike times.
        if synapse_update == "exact":
            return np.exp(-beta * dt) * u_ + beta * drive_
        return u_ + (-beta * u_) * dt + beta * drive_

    nb = int(burn / dt)
    for _ in range(nb):
        phi_old = phi.copy()
        rate = F(u)
        phi = phi_old + rate * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        # Correct multi-spike: wrap phi into (-pi, pi) regardless of how many
        # cycles were completed in this step (F(u)*dt can exceed 2*pi for large u).
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        filtered_counts = spike_weights(phi_old, rate, spike_counts)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            drive = W @ filtered_counts
        u = update_u(u, drive)
        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)

    nt = int(T / dt)
    if n_probe is None:
        n_probe = N
    n_probe = int(max(1, min(N, n_probe)))
    probe_idx = np.arange(n_probe)
    U_probe = np.zeros((nt, n_probe))
    R_pop = np.zeros(nt, dtype=float) if return_spike else None
    Eta_probe = (
        np.zeros((nt, n_probe), dtype=np.float32)
        if return_phase_density
        else None
    )
    phase_bin_width = float(phase_bin_width)
    if return_phase_density and not 0.0 < phase_bin_width <= 2.0 * np.pi:
        raise ValueError("phase_bin_width must lie in (0, 2*pi]")
    for t in range(nt):
        phi_old = phi.copy()
        rate = F(u)
        phi = phi_old + rate * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        filtered_counts = spike_weights(phi_old, rate, spike_counts)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            drive = W @ filtered_counts
        u = update_u(u, drive)
        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
        U_probe[t] = u[probe_idx]
        if return_spike:
            R_pop[t] = np.mean(spike_counts) / dt
        if return_phase_density:
            Eta_probe[t] = (
                phi[probe_idx] >= np.pi - phase_bin_width
            ) / phase_bin_width

    max_lag = int(tau_max / dt)
    # Average per-neuron autocorrelation.  At short lags this includes the
    # exactly filtered same-event contribution of the point process.
    C = np.mean([autocorr(U_probe[:, i], max_lag) for i in range(n_probe)], axis=0)
    tau = np.arange(len(C)) * dt
    if not return_spike and not return_u and not return_phase_density:
        return tau, C

    out = [tau, C]
    if return_spike:
        C_spk = autocorr(R_pop, max_lag)
        out.append(C_spk)
    if return_u:
        out.append(U_probe)
    if return_phase_density:
        C_density = np.mean(
            [autocorr(Eta_probe[:, i], max_lag) for i in range(n_probe)],
            axis=0,
        )
        out.append(C_density)
    return tuple(out)


def _theory_phase_scalar_autocorr(
    I=1.0,
    alpha=1.0,
    sigma=7.0,
    beta=1.0,
    C0=None,
    tau_max=50,
    dtau=0.01,
    n_quad=24,
    solver="cusp",
    q_method="gh",
    n_qmc=2048,
    hermite_order=32,
    warn_on_no_branch=True,
):
    """
    Solve the scalar Gaussian 2PI closure for the phase network with lambda=1.

    With row-sum correction (lambda=1), W@1=0 so E[u_i]=0 exactly, meaning
    C_11(tau->inf)=0 and C_eq=0. The centered SCS with gain g is directly correct:
    no g_shifted or C_eq machinery needed.

    The same-spike term is imposed through the exact cusp condition

        C'(0+) = -beta**2 * sigma**2 * mean_rate / 2,

    and C(0) is fixed by the corresponding energy equation.  For positive lag,
    the smooth off-diagonal closure obeys

        C'' = beta**2 * (C - sigma**2 * Cov[g(u(0)), g(u(tau))]).

    q_method="gh" uses tensor Gauss-Hermite quadrature for Q_smooth. q_method="qmc"
    uses common-random Sobol Gaussian samples, which is slower but more robust
    for hard rectification and strongly nonlinear gains. q_method="hermite" uses
    a 1-D Hermite expansion of g(u) and evaluates the centered covariance as a
    series in C(tau)/C(0), avoiding cancellation from subtracting E[g]^2.
    The smooth Gaussian closure averages over phases and therefore does not
    retain the near-threshold C33 advection peaks described in the notes.  Those
    peaks require a richer phase-density closure than this scalar C_uu solver.
    warn_on_no_branch controls whether missing cusp branches are printed; scans
    turn this off to avoid repetitive console noise.
    """
    rho = 1.0 / (2.0 * np.pi)

    def g(u):
        return rho * alpha * np.clip(I + u, 0.0, 1e12) ** (1.0 / alpha)

    Fprime0 = float(np.maximum(I, 1e-10) ** (1.0 / alpha - 1.0))
    sigma_c = 1.0 / (rho * Fprime0)

    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    gh_w2 = np.outer(gh_w, gh_w)
    q_method = str(q_method).lower()
    use_hermite_series = q_method in ("hermite", "hermite-series", "series")
    if q_method in ("sobol", "qmc", "mc"):
        try:
            from scipy.special import ndtri
            from scipy.stats import qmc

            sampler = qmc.Sobol(d=2, scramble=True, seed=12345)
            m = int(np.ceil(np.log2(max(16, int(n_qmc)))))
            u_sobol = sampler.random_base2(m)
            eps = np.finfo(float).eps
            z_qmc = ndtri(np.clip(u_sobol, eps, 1.0 - eps))
        except Exception:
            q_rng = np.random.default_rng(12345)
            z_qmc = q_rng.normal(size=(max(16, int(n_qmc)), 2))
        z0_qmc = z_qmc[:, 0]
        z1_qmc = z_qmc[:, 1]
    else:
        z0_qmc = z1_qmc = None
    hermite_cache = {}

    def hermite_coeffs(C0_val):
        """
        Return normalized Hermite coefficients b_n=E[g(sZ) phi_n(Z)].

        phi_n(Z)=He_n(Z)/sqrt(n!) are orthonormal under Z~N(0,1).  The
        centered covariance is sum_{n>=1} b_n^2 rho^n, so the mean mode b_0 is
        intentionally omitted by Q_centered.
        """
        C0_val = float(C0_val)
        key = round(C0_val, 12)
        if key in hermite_cache:
            return hermite_cache[key]

        z = np.sqrt(2.0) * gh_x
        vals = g(np.sqrt(max(C0_val, 0.0)) * z)
        w = gh_w / np.sqrt(np.pi)
        order = int(max(1, hermite_order))
        coeffs = np.zeros(order + 1)

        phi_nm1 = np.ones_like(z)
        coeffs[0] = float(np.dot(w, vals * phi_nm1))
        if order >= 1:
            phi_n = z
            coeffs[1] = float(np.dot(w, vals * phi_n))
            for n in range(1, order):
                phi_np1 = (z * phi_n - np.sqrt(n) * phi_nm1) / np.sqrt(n + 1.0)
                coeffs[n + 1] = float(np.dot(w, vals * phi_np1))
                phi_nm1, phi_n = phi_n, phi_np1

        # Avoid unbounded cache growth during nonlinear solves and scans.
        if len(hermite_cache) > 2048:
            hermite_cache.clear()
        hermite_cache[key] = coeffs
        return coeffs

    def mu_g(C_val):
        """E[g(u)] for u ~ N(0, C_val) via 1-D GH quadrature."""
        C_val = float(C_val)
        if C_val <= 0.0:
            return float(g(0.0))
        if z0_qmc is not None:
            return float(np.mean(g(np.sqrt(C_val) * z0_qmc)))
        s = np.sqrt(2.0 * C_val)
        return float(np.sum(gh_w * g(s * gh_x)) / np.sqrt(np.pi))

    def Q_centered(C_tau, C0_val):
        """sigma^2 * Cov[g(u(0)), g(u(tau))] for a Gaussian pair."""
        C0_val = float(C0_val)
        if C0_val <= 0.0:
            return 0.0
        rho_tau = float(np.clip(C_tau / C0_val, -0.999999, 0.999999))
        if use_hermite_series:
            coeffs = hermite_coeffs(C0_val)
            powers = rho_tau ** np.arange(1, len(coeffs))
            return sigma**2 * float(np.dot(coeffs[1:] ** 2, powers))
        if z0_qmc is not None:
            s = np.sqrt(C0_val)
            x = s * z0_qmc
            y = s * (rho_tau * z0_qmc + np.sqrt(1.0 - rho_tau**2) * z1_qmc)
            gx = g(x)
            gy = g(y)
            return sigma**2 * float(np.mean(gx * gy) - np.mean(gx) * np.mean(gy))
        scale = np.sqrt(2.0 * C0_val)
        x = scale * gh_x[:, None]
        y = scale * (
            rho_tau * gh_x[:, None] + np.sqrt(1.0 - rho_tau**2) * gh_x[None, :]
        )
        raw = float(np.sum(gh_w2 * g(x) * g(y)) / np.pi)
        mu = mu_g(C0_val)
        return sigma**2 * (raw - mu**2)

    beta_val = max(float(beta), 1e-10)
    C0_hi = max(50.0, 10.0 * sigma ** 4 / (16.0 * np.pi ** 3))

    def smooth_energy(C_val, C0_val, n_grid=384):
        """H(C; C0) = C^2 - 2 int_0^C Q_smooth(x; C0) dx."""
        C_val = float(C_val)
        C0_val = float(C0_val)
        if C_val <= 0.0 or C0_val <= 0.0:
            return np.nan
        C_grid = np.linspace(0.0, C_val, n_grid)
        Q_grid = np.array([Q_centered(c, C0_val) for c in C_grid])
        return float(C_val**2 - 2.0 * np.trapezoid(Q_grid, C_grid))

    ntau = int(tau_max / dtau)
    tau = np.arange(ntau) * dtau

    solver_key = str(solver).lower()
    if solver_key not in ("cusp", "same_spike", "same-spike"):
        raise ValueError("scalar phase theory supports solver='cusp'")

    def solve_cusp_c0():
        if C0 is not None:
            guess_val = float(C0)
            if guess_val > 0.0:
                return guess_val

        def balance(C_total):
            C_total = float(C_total)
            if C_total <= 0.0:
                return np.nan
            cusp_energy = 0.25 * beta_val**2 * sigma**4 * mu_g(C_total) ** 2
            return smooth_energy(C_total, C_total) - cusp_energy

        lo, hi = 1e-5, C0_hi
        linear = np.linspace(lo, min(hi, 10.0), 80)
        log = np.logspace(np.log10(lo), np.log10(hi), 80)
        candidates = np.unique(np.sort(np.concatenate([linear, log])))
        values = np.array([balance(c) for c in candidates])
        finite = np.isfinite(values)
        candidates, values = candidates[finite], values[finite]
        if len(candidates) == 0:
            return np.nan

        for i in range(len(candidates) - 1):
            if values[i] == 0.0:
                return float(candidates[i])
            if values[i] < 0.0 < values[i + 1]:
                a, b = float(candidates[i]), float(candidates[i + 1])
                fa = float(values[i])
                for _ in range(60):
                    m = 0.5 * (a + b)
                    fm = balance(m)
                    if not np.isfinite(fm) or abs(fm) < 1e-10:
                        return float(m)
                    if fa * fm <= 0.0:
                        b = m
                    else:
                        a, fa = m, fm
                return float(0.5 * (a + b))
        if warn_on_no_branch:
            print("  no finite cusp-closure branch found")
        return np.nan

    def cusp_solution():
        C_total0 = solve_cusp_c0()
        if not np.isfinite(C_total0):
            return np.full_like(tau, np.nan, dtype=float)

        # Use the conserved energy to select the decaying stationary branch.
        # Direct integration of the second-order equation eventually excites its
        # exponentially growing companion after C has become numerically tiny.
        n_grid = 4096
        positive = np.unique(
            np.concatenate(
                (
                    np.geomspace(C_total0 * 1e-12, C_total0, n_grid),
                    np.linspace(C_total0 / n_grid, C_total0, n_grid),
                )
            )
        )
        C_grid = np.concatenate(([0.0], positive))
        Q_grid = np.array([Q_centered(c, C_total0) for c in C_grid])
        integral = np.zeros_like(C_grid)
        integral[1:] = np.cumsum(
            0.5 * (Q_grid[1:] + Q_grid[:-1]) * np.diff(C_grid)
        )
        energy = C_grid**2 - 2.0 * integral
        energy_scale = max(C_total0**2, 1.0)
        if np.min(energy) < -1e-7 * energy_scale:
            if warn_on_no_branch:
                print("  cusp energy has no real decaying branch")
            return np.full_like(tau, np.nan, dtype=float)
        speed = beta_val * np.sqrt(np.maximum(energy[1:], 1e-30))

        C_desc = positive[::-1]
        speed_desc = speed[::-1]
        dC = -np.diff(C_desc)
        segment_speed = 0.5 * (speed_desc[:-1] + speed_desc[1:])
        tau_desc = np.concatenate(
            ([0.0], np.cumsum(dC / np.maximum(segment_speed, 1e-30)))
        )
        return np.interp(tau, tau_desc, C_desc, left=C_total0, right=0.0)

    return tau, cusp_solution(), sigma_c


def _gaussian_process_from_spectrum(spectrum, normals, n_time):
    """Synthesize real stationary Gaussian paths with a target FFT spectrum."""
    spectrum = np.maximum(np.asarray(spectrum, dtype=float), 0.0)
    coeff = normals * np.sqrt(n_time * spectrum)[None, :]
    return np.fft.irfft(coeff, n=n_time, axis=1)


def _phase_spike_spectrum(
    drive,
    I,
    alpha,
    dt,
    initial_phase,
    phase_bin_width=None,
):
    """Advect phase trajectories and return their centered spike spectrum."""
    n_samples, n_time = drive.shape
    phase = np.asarray(initial_phase, dtype=float).copy()
    counts = np.zeros((n_samples, n_time), dtype=np.float32)
    density = None
    if phase_bin_width is not None:
        phase_bin_width = float(phase_bin_width)
        if not 0.0 < phase_bin_width <= 2.0 * np.pi:
            raise ValueError("phase_bin_width must lie in (0, 2*pi]")
        density = np.zeros((n_samples, n_time), dtype=np.float32)

    for k in range(n_time):
        velocity = alpha * np.clip(I + drive[:, k], 0.0, 1e12) ** (1.0 / alpha)
        phase += velocity * dt
        spike_count = np.floor((phase + np.pi) / (2.0 * np.pi))
        spiking = spike_count > 0.0
        phase[spiking] = ((phase[spiking] + np.pi) % (2.0 * np.pi)) - np.pi
        counts[:, k] = spike_count
        if density is not None:
            density[:, k] = (phase >= np.pi - phase_bin_width) / phase_bin_width

    rate = counts.astype(float) / dt
    sample_mean_rates = np.mean(rate, axis=1)
    mean_rate = float(np.mean(sample_mean_rates))
    # Estimate the connected stationary covariance.  Centering each trajectory
    # removes its finite-window DC error, matching the per-neuron temporal
    # centering used by sim_phase_network and preventing that error from being
    # amplified by the slow synaptic mode.
    centered_rate = rate - np.mean(rate, axis=1, keepdims=True)
    transformed = np.fft.rfft(centered_rate, axis=1)
    spectrum = np.mean(np.abs(transformed) ** 2, axis=0) / n_time
    covariance = np.fft.irfft(spectrum, n=n_time)
    density_covariance = None
    if density is not None:
        density -= np.mean(density, axis=1, keepdims=True)
        transformed_density = np.fft.rfft(density, n=2 * n_time, axis=1)
        density_covariance = np.mean(
            np.fft.irfft(
                np.abs(transformed_density) ** 2,
                n=2 * n_time,
                axis=1,
            )[:, :n_time],
            axis=0,
        ) / (n_time - np.arange(n_time))
    return spectrum, covariance, mean_rate, sample_mean_rates, density_covariance


def theory_phase_density_autocorr(
    I=1.0,
    alpha=1.0,
    sigma=7.0,
    beta=1.0,
    tau_max=50.0,
    dtau=0.02,
    internal_dt=0.02,
    n_time=16384,
    n_samples=64,
    max_iter=24,
    mixing=0.25,
    tolerance=2e-3,
    seed=1729,
    return_diagnostics=False,
    phase_bin_width=0.25,
):
    """Stationary event-DMFT closure of the Gaussian 2PI drive equation.

    For a trial C11, Gaussian single-site drives are synthesized in Fourier
    space.  The deterministic phase density is then advected through each
    drive, producing the complete spike covariance.  Filtering that covariance
    with beta/(beta+i*omega) closes a C11 -> R_off -> C11 loop.
    The binned spike covariance contains the exact same-event contribution as
    well as the distinct-return peaks generated by C33 transport.

    For prescribed Q, the causal Gaussian 2PI equations give
    C11 = beta**2 * sigma**2 * R Q R.T.  This routine closes that equation by
    evaluating Q from conditional deterministic phase trajectories.  It does
    not independently integrate C13, C31, C33, or their response functions;
    the trajectory average implicitly replaces that density-sector closure.
    """
    n_time = int(max(512, n_time))
    n_samples = int(max(4, n_samples))
    dt = float(internal_dt)
    if dt <= 0.0 or dtau <= 0.0:
        raise ValueError("internal_dt and dtau must be positive")
    beta = max(float(beta), 1e-12)
    sigma = float(sigma)
    rng_local = np.random.default_rng(seed)

    n_freq = n_time // 2 + 1
    normals = (
        rng_local.normal(size=(n_samples, n_freq))
        + 1j * rng_local.normal(size=(n_samples, n_freq))
    ) / np.sqrt(2.0)
    normals[:, 0] = 0.0
    if n_time % 2 == 0:
        normals[:, -1] = rng_local.normal(size=n_samples)
    initial_phase = rng_local.uniform(-np.pi, np.pi, n_samples)
    static_normals = rng_local.normal(size=n_samples)

    omega = 2.0 * np.pi * np.fft.rfftfreq(n_time, d=dt)
    decay = np.exp(-beta * dt)
    # The spike spectrum is formed from binned rates.  This is the exact
    # transfer function of u[k+1]=decay*u[k]+beta*dt*s[k]; it converges to
    # beta**2/(beta**2+omega**2) and preserves the same-event cusp at finite dt.
    synaptic_filter = (beta * dt) ** 2 / np.maximum(
        1.0 + decay**2 - 2.0 * decay * np.cos(omega * dt), 1e-30
    )
    baseline_rate = alpha * max(float(I), 0.0) ** (1.0 / alpha) / (2.0 * np.pi)
    circular_lag = np.minimum(np.arange(n_time), n_time - np.arange(n_time)) * dt
    # Numerical initial iterate only: filtering the exact same-event
    # covariance sigma**2 * baseline_rate * delta(tau) gives this exponential.
    # It is replaced by the measured spike spectrum during fixed-point
    # iteration and is not an additional term in the closure.
    initial_covariance = (
        0.5 * beta * sigma**2 * baseline_rate * np.exp(-beta * circular_lag)
    )
    drive_spectrum = np.maximum(np.real(np.fft.rfft(initial_covariance)), 0.0)
    drive_spectrum[0] = 0.0
    static_variance = 0.0

    history = []
    spike_covariance = np.zeros(n_time)
    mean_rate = baseline_rate
    converged = False
    for iteration in range(int(max_iter)):
        drive = _gaussian_process_from_spectrum(drive_spectrum, normals, n_time)
        drive += np.sqrt(max(static_variance, 0.0)) * static_normals[:, None]
        spike_spectrum, spike_covariance, mean_rate, sample_mean_rates, _ = (
            _phase_spike_spectrum(
            drive, I, alpha, dt, initial_phase
            )
        )
        proposed = np.maximum(sigma**2 * synaptic_filter * spike_spectrum, 0.0)
        proposed[0] = 0.0
        proposed_static = sigma**2 * float(np.var(sample_mean_rates))
        dynamic_scale = float(np.linalg.norm(drive_spectrum))
        scale = max(np.hypot(dynamic_scale, static_variance), 1e-12)
        residual = float(
            np.hypot(
                np.linalg.norm(proposed - drive_spectrum),
                proposed_static - static_variance,
            )
            / scale
        )
        history.append(residual)
        drive_spectrum = (1.0 - mixing) * drive_spectrum + mixing * proposed
        static_variance = (
            (1.0 - mixing) * static_variance + mixing * proposed_static
        )
        if residual < tolerance:
            converged = True
            break

    covariance = np.fft.irfft(drive_spectrum, n=n_time)
    max_internal_lag = min(int(np.ceil(tau_max / dt)) + 1, n_time // 2)
    internal_tau = np.arange(max_internal_lag) * dt
    n_lag = int(tau_max / dtau)
    tau = np.arange(n_lag) * dtau
    result = np.interp(tau, internal_tau, covariance[:max_internal_lag])
    sigma_c = _phase_sigma_c(I, alpha)
    if not return_diagnostics:
        return tau, result, sigma_c

    diagnostic_drive = _gaussian_process_from_spectrum(
        drive_spectrum, normals, n_time
    )
    diagnostic_drive += (
        np.sqrt(max(static_variance, 0.0)) * static_normals[:, None]
    )
    (
        _spike_spectrum,
        spike_covariance,
        mean_rate,
        _sample_mean_rates,
        phase_density_covariance,
    ) = _phase_spike_spectrum(
        diagnostic_drive,
        I,
        alpha,
        dt,
        initial_phase,
        phase_bin_width=phase_bin_width,
    )

    same_event = mean_rate / dt
    off_covariance = spike_covariance.copy()
    off_covariance[0] -= same_event
    diagnostics = dict(
        converged=converged,
        iterations=iteration + 1,
        residual_history=np.asarray(history),
        mean_rate=mean_rate,
        spike_covariance=spike_covariance,
        off_spike_covariance=off_covariance,
        internal_dt=dt,
        drive_spectrum=drive_spectrum,
        static_variance=static_variance,
        phase_bin_width=float(phase_bin_width),
        phase_density_covariance=phase_density_covariance,
    )
    return tau, result, sigma_c, diagnostics


def _phase_twotime_paths(drive, initial_phase, I, alpha, beta, dt, phase_bin_width):
    """Propagate the effective phase process and its filtered threshold flux."""
    n_samples, n_time = drive.shape
    phase = np.asarray(initial_phase, dtype=float).copy()
    phases = np.empty((n_samples, n_time), dtype=np.float32)
    density = np.empty_like(phases)
    filtered_flux = np.zeros_like(drive)
    velocity_derivative = np.zeros_like(drive)
    synapse = np.zeros(n_samples)
    decay = np.exp(-beta * dt)

    for k in range(n_time):
        total_drive = I + drive[:, k]
        positive = total_drive > 0.0
        velocity = np.zeros(n_samples)
        velocity[positive] = (
            alpha * total_drive[positive] ** (1.0 / alpha)
        )
        velocity_derivative[positive, k] = (
            total_drive[positive] ** (1.0 / alpha - 1.0)
        )
        old_phase = phase.copy()
        unwrapped = old_phase + velocity * dt
        counts = np.floor((unwrapped + np.pi) / (2.0 * np.pi)).astype(int)
        spiking = counts > 0
        phase = ((unwrapped + np.pi) % (2.0 * np.pi)) - np.pi

        weighted_counts = np.zeros(n_samples)
        for sample in np.flatnonzero(spiking):
            thresholds = np.pi + 2.0 * np.pi * np.arange(counts[sample])
            crossing_times = (thresholds - old_phase[sample]) / velocity[sample]
            crossing_times = np.clip(crossing_times, 0.0, dt)
            weighted_counts[sample] = np.sum(
                np.exp(-beta * (dt - crossing_times))
            )
        synapse = decay * synapse + beta * weighted_counts
        filtered_flux[:, k] = synapse
        phases[:, k] = phase
        density[:, k] = (phase >= np.pi - phase_bin_width) / phase_bin_width

    return phases, density, filtered_flux, velocity_derivative


def _sample_gaussian_kernel(covariance, normals):
    """Sample a zero-mean Gaussian process from a positive semidefinite kernel."""
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(eigenvalues)), 1.0)
    eigenvalues[eigenvalues < 1e-12 * scale] = 0.0
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]
    return normals @ factor.T


def _lag_average(matrix, max_lag, row_later=True, start=0):
    """Average a two-time matrix along diagonals after an initial transient."""
    values = np.empty(max_lag)
    for lag in range(max_lag):
        if row_later:
            diagonal = np.diag(matrix[start:, start:], k=-lag)
        else:
            diagonal = np.diag(matrix[start:, start:], k=lag)
        values[lag] = np.mean(diagonal) if len(diagonal) else np.nan
    return values


def theory_phase_twotime_dmft(
    I=1.0,
    alpha=1.0,
    sigma=7.0,
    beta=1.0,
    tau_max=20.0,
    dtau=0.05,
    internal_dt=0.05,
    n_time=640,
    n_samples=512,
    max_iter=30,
    mixing=0.2,
    tolerance=2e-2,
    transient_fraction=0.35,
    phase_bin_width=0.25,
    response_modes=(1, 2, 3),
    seed=271828,
    return_diagnostics=False,
):
    """Solve the nonstationary two-time event-DMFT closure.

    The recurrent field kernel is updated in the time domain from the exact
    filtered threshold flux of the effective deterministic phase process.
    Uniform initial phases provide the C33 boundary measure.  No stationarity,
    periodic covariance, return kernel, or diffusion coefficient is assumed.

    This is the full two-time version of the DMFT closure of Q in the 2PI
    drive equation. Mixed covariance and causal phase-response sectors are
    measured from the same paths and returned as diagnostics, but are not
    evolved as independent variational propagators.
    """
    n_time = int(max(32, n_time))
    n_samples = int(max(8, n_samples))
    dt = float(internal_dt)
    if dt <= 0.0 or dtau <= 0.0:
        raise ValueError("internal_dt and dtau must be positive")
    if not 0.0 < phase_bin_width <= 2.0 * np.pi:
        raise ValueError("phase_bin_width must lie in (0, 2*pi]")
    if not 0.0 <= transient_fraction < 1.0:
        raise ValueError("transient_fraction must lie in [0, 1)")

    rng_local = np.random.default_rng(seed)
    initial_phase = rng_local.uniform(-np.pi, np.pi, n_samples)
    normals = rng_local.normal(size=(n_samples, n_time))
    normals -= np.mean(normals, axis=0, keepdims=True)
    normals /= np.maximum(np.std(normals, axis=0, keepdims=True), 1e-12)

    zero_drive = np.zeros((n_samples, n_time))
    _, _, initial_flux, _ = _phase_twotime_paths(
        zero_drive,
        initial_phase,
        I,
        alpha,
        beta,
        dt,
        phase_bin_width,
    )
    centered_flux = initial_flux - np.mean(initial_flux, axis=0, keepdims=True)
    covariance = sigma**2 * centered_flux.T @ centered_flux / n_samples

    residual_history = []
    converged = sigma == 0.0
    for iteration in range(int(max_iter)):
        drive = _sample_gaussian_kernel(covariance, normals)
        phases, density, filtered_flux, velocity_derivative = _phase_twotime_paths(
            drive,
            initial_phase,
            I,
            alpha,
            beta,
            dt,
            phase_bin_width,
        )
        centered_flux = filtered_flux - np.mean(
            filtered_flux, axis=0, keepdims=True
        )
        proposed = sigma**2 * centered_flux.T @ centered_flux / n_samples
        proposed = 0.5 * (proposed + proposed.T)
        residual = float(
            np.linalg.norm(proposed - covariance)
            / max(np.linalg.norm(covariance), 1e-12)
        )
        residual_history.append(residual)
        covariance = (1.0 - mixing) * covariance + mixing * proposed
        if residual < tolerance:
            converged = True
            break

    drive = _sample_gaussian_kernel(covariance, normals)
    phases, density, filtered_flux, velocity_derivative = _phase_twotime_paths(
        drive,
        initial_phase,
        I,
        alpha,
        beta,
        dt,
        phase_bin_width,
    )
    drive_centered = drive - np.mean(drive, axis=0, keepdims=True)
    density_centered = density - np.mean(density, axis=0, keepdims=True)
    C11 = drive_centered.T @ drive_centered / n_samples
    C13 = drive_centered.T @ density_centered / n_samples
    C31 = C13.T
    C33 = density_centered.T @ density_centered / n_samples

    response = {}
    causal_mask = np.tril(np.ones((n_time, n_time), dtype=bool), k=-1)
    for mode in response_modes:
        mode = int(mode)
        phase_mode = np.exp(-1j * mode * phases)
        phase_velocity_cross = phase_mode.T @ velocity_derivative / n_samples
        response[mode] = np.where(
            causal_mask,
            -1j * mode * dt * phase_velocity_cross,
            0.0,
        )

    start = int(transient_fraction * n_time)
    available = n_time - start
    max_lag = min(int(np.ceil(tau_max / dt)) + 1, available)
    internal_tau = np.arange(max_lag) * dt
    C11_lag = _lag_average(C11, max_lag, row_later=True, start=start)
    C13_lag = _lag_average(C13, max_lag, row_later=True, start=start)
    C31_lag = _lag_average(C31, max_lag, row_later=True, start=start)
    C33_lag = _lag_average(C33, max_lag, row_later=True, start=start)
    n_lag = max(1, int(tau_max / dtau))
    tau = np.arange(n_lag) * dtau
    result = np.interp(tau, internal_tau, C11_lag)
    sigma_c = _phase_sigma_c(I, alpha)
    if not return_diagnostics:
        return tau, result, sigma_c

    diagnostics = dict(
        converged=converged,
        iterations=iteration + 1,
        residual_history=np.asarray(residual_history),
        internal_dt=dt,
        covariance_kernel=C11,
        C13_kernel=C13,
        C31_kernel=C31,
        C33_kernel=C33,
        C13_lag=C13_lag,
        C31_lag=C31_lag,
        phase_density_covariance=C33_lag,
        phase_response_modes=response,
        phases=phases,
        filtered_flux=filtered_flux,
    )
    return tau, result, sigma_c, diagnostics


# Backward-compatible name used by earlier development scripts.
theory_phase_twotime_gaussian = theory_phase_twotime_dmft


def theory_phase_fixed_q_gaussian(
    I=1.0,
    alpha=1.0,
    sigma=1.0,
    beta=1.0,
    tau_max=20.0,
    dtau=0.05,
    internal_dt=None,
    n_time=None,
    n_phase=65,
    max_iter=400,
    mixing=0.15,
    tolerance=1e-7,
    phase_bin_width=0.25,
    return_diagnostics=False,
):
    """Run the incomplete fixed-bilocal-kernel Gaussian diagnostic.

    This is not a full 2PI solution: Q and Qhat propagators are omitted, and
    the exact event diagonal is combined with an external Wick closure.  The
    result can violate covariance positivity and is retained for auditing.
    """
    if not np.isclose(alpha, 1.0):
        raise NotImplementedError(
            "the fixed-Q Gaussian diagnostic currently supports alpha=1"
        )
    F0 = float(I)
    if F0 <= 0.0:
        raise ValueError("the fixed-Q Gaussian diagnostic requires I > 0")
    period = 2.0 * np.pi / F0
    if internal_dt is None:
        internal_dt = period / 128.0
    internal_dt = float(internal_dt)
    if n_time is None:
        n_time = int(np.ceil((tau_max + 2.0 * period) / internal_dt)) + 1
    solution = solve_uniform_phase_fixed_q_gaussian(
        n_time=int(n_time),
        n_phase=int(n_phase),
        dt=internal_dt,
        beta=beta,
        sigma=sigma,
        F0=F0,
        F1=1.0,
        max_iter=max_iter,
        mixing=mixing,
        tolerance=tolerance,
    )
    start = min(int(np.ceil(period / internal_dt)), solution.C11.shape[0] // 2)
    max_lag = min(
        int(np.ceil(tau_max / internal_dt)) + 1,
        solution.C11.shape[0] - start,
    )
    internal_tau = np.arange(max_lag) * internal_dt
    C11_lag = _lag_average(
        solution.C11, max_lag, row_later=True, start=start
    )
    flux_C33_lag = _lag_average(
        solution.C33_threshold, max_lag, row_later=True, start=start
    )
    displacement = np.mod(F0 * internal_tau, 2.0 * np.pi)
    width = float(phase_bin_width)
    if not 0.0 < width <= 2.0 * np.pi:
        raise ValueError("phase_bin_width must lie in (0, 2*pi]")
    overlap = np.maximum(0.0, width - displacement)
    overlap += np.maximum(0.0, width - (2.0 * np.pi - displacement))
    C33_lag = overlap / (2.0 * np.pi * width**2) - 1.0 / (2.0 * np.pi) ** 2
    tau = np.arange(max(1, int(tau_max / dtau))) * dtau
    covariance = np.interp(tau, internal_tau, C11_lag)
    sigma_c = _phase_sigma_c(I, alpha)
    if not return_diagnostics:
        return tau, covariance, sigma_c
    diagnostics = dict(
        converged=solution.converged,
        residual_history=solution.residual_history,
        C11_kernel=solution.C11,
        C13_kernel=solution.C13,
        C31_kernel=solution.C31,
        C33_kernel=solution.C33_threshold,
        phase_density_covariance=C33_lag,
        flux_density_covariance=flux_C33_lag,
        flux_kernel=solution.flux_kernel,
        mean_rate=solution.mean_rate,
        internal_dt=solution.dt,
        phase_dv=solution.dv,
        phase_bin_width=width,
    )
    covariance_kernel = 0.5 * (solution.C11 + solution.C11.T)
    flux_kernel = 0.5 * (solution.flux_kernel + solution.flux_kernel.T)
    min_covariance_eigenvalue = float(np.linalg.eigvalsh(covariance_kernel)[0])
    min_flux_eigenvalue = float(np.linalg.eigvalsh(flux_kernel)[0])
    diagonal = np.diag(covariance_kernel)
    denominator = np.sqrt(np.maximum(diagonal[:, None] * diagonal[None, :], 0.0))
    valid_denominator = denominator > 1e-14
    normalized = np.zeros_like(covariance_kernel)
    normalized[valid_denominator] = (
        np.abs(covariance_kernel[valid_denominator])
        / denominator[valid_denominator]
    )
    max_normalized_covariance = float(np.max(normalized))
    scale = max(float(np.max(np.abs(diagonal))), 1.0)
    physical = bool(
        min_covariance_eigenvalue >= -1e-10 * scale
        and min_flux_eigenvalue >= -1e-10 * scale
        and max_normalized_covariance <= 1.0 + 1e-8
    )
    diagnostics.update(
        min_covariance_eigenvalue=min_covariance_eigenvalue,
        min_flux_eigenvalue=min_flux_eigenvalue,
        max_normalized_covariance=max_normalized_covariance,
        physical=physical,
    )
    return tau, covariance, sigma_c, diagnostics


def theory_phase_gaussian_2pi(
    I=1.0,
    alpha=1.0,
    sigma=1.0,
    beta=1.0,
    tau_max=20.0,
    dtau=0.05,
    internal_dt=None,
    n_time=None,
    n_phase=129,
    max_iter=1000,
    mixing=1.0,
    tolerance=1e-9,
    phase_bin_width=0.25,
    return_diagnostics=False,
):
    """Solve the legacy homogeneous Hartree/Wick approximation."""
    if not np.isclose(alpha, 1.0):
        raise NotImplementedError(
            "the Hartree/Wick solver currently supports alpha=1"
        )
    F0 = float(I)
    if F0 <= 0.0:
        raise ValueError("the Hartree/Wick solver requires I > 0")
    if int(n_phase) % 2 == 0:
        raise ValueError("the Hartree/Wick spectral grid requires odd n_phase")
    period = 2.0 * np.pi / F0
    matched_dt = period / int(n_phase)
    if internal_dt is None:
        internal_dt = matched_dt
    elif not np.isclose(internal_dt, matched_dt, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "Hartree/Wick requires internal_dt = period / n_phase so the "
            "threshold phase cell and event time bin are the same observable"
        )
    internal_dt = float(internal_dt)
    if n_time is None:
        n_time = int(np.ceil((tau_max + 2.0 * period) / internal_dt)) + 1
    solution = solve_uniform_phase_gaussian_2pi(
        n_time=int(n_time),
        n_phase=int(n_phase),
        dt=internal_dt,
        beta=beta,
        sigma=sigma,
        F0=F0,
        F1=1.0,
        max_iter=max_iter,
        mixing=mixing,
        tolerance=tolerance,
    )
    start = min(int(np.ceil(period / internal_dt)), int(n_time) // 2)
    max_lag = min(
        int(np.ceil(tau_max / internal_dt)) + 1,
        int(n_time) - start,
    )
    internal_tau = np.arange(max_lag) * internal_dt
    if solution.stable:
        C11_lag = _lag_average(
            solution.C11, max_lag, row_later=True, start=start
        )
    else:
        C11_lag = np.full(max_lag, np.nan)
    flux_C33_lag = _lag_average(
        solution.C33_threshold, max_lag, row_later=True, start=start
    )
    displacement = np.mod(F0 * internal_tau, 2.0 * np.pi)
    width = float(phase_bin_width)
    if not 0.0 < width <= 2.0 * np.pi:
        raise ValueError("phase_bin_width must lie in (0, 2*pi]")
    overlap = np.maximum(0.0, width - displacement)
    overlap += np.maximum(0.0, width - (2.0 * np.pi - displacement))
    C33_lag = overlap / (2.0 * np.pi * width**2) - 1.0 / (2.0 * np.pi) ** 2
    tau = np.arange(max(1, int(tau_max / dtau))) * dtau
    covariance = np.interp(tau, internal_tau, C11_lag)
    sigma_c = _phase_sigma_c(I, alpha)
    if not return_diagnostics:
        return tau, covariance, sigma_c
    diagnostics = dict(
        converged=solution.converged,
        stable=solution.stable,
        physical=solution.physical,
        residual_history=solution.residual_history,
        feedback_eigenvalue=solution.feedback_eigenvalue,
        gaussian_critical_sigma=(
            np.inf
            if solution.feedback_eigenvalue <= 0.0 or sigma == 0.0
            else abs(sigma) / np.sqrt(solution.feedback_eigenvalue)
        ),
        min_covariance_eigenvalue=solution.min_covariance_eigenvalue,
        min_flux_eigenvalue=solution.min_flux_eigenvalue,
        max_normalized_covariance=solution.max_normalized_covariance,
        C11_kernel=solution.C11,
        C13_kernel=solution.C13,
        C31_kernel=solution.C31,
        C33_kernel=solution.C33_threshold,
        phase_density_covariance=C33_lag,
        flux_density_covariance=flux_C33_lag,
        flux_kernel=solution.flux_kernel,
        mean_rate=solution.mean_rate,
        internal_dt=solution.dt,
        phase_dv=solution.dv,
        phase_bin_width=width,
    )
    return tau, covariance, sigma_c, diagnostics


def theory_phase_autocorr(*args, solver="density", **kwargs):
    """Dispatch to the event-DMFT, Gaussian 2PI, or scalar closure."""
    solver_key = str(solver).lower()
    if solver_key in ("gaussian_2pi", "gaussian-2pi"):
        return theory_phase_gaussian_2pi(*args, **kwargs)
    if solver_key in ("fixed_q", "fixed-q", "fixed_q_gaussian"):
        return theory_phase_fixed_q_gaussian(*args, **kwargs)
    if solver_key in (
        "twotime",
        "two-time",
        "twotime_dmft",
        "two-time-dmft",
        "twotime_gaussian",
    ):
        return theory_phase_twotime_dmft(*args, **kwargs)
    if solver_key in ("density", "dmft", "event_dmft", "phase_density", "phase-density"):
        return theory_phase_density_autocorr(*args, **kwargs)
    if solver_key in ("cusp", "scalar", "smooth"):
        return _theory_phase_scalar_autocorr(*args, solver="cusp", **kwargs)
    raise ValueError(
        "phase theory solver must be 'gaussian_2pi', 'fixed_q', 'twotime', "
        "'density', or 'cusp'"
    )


def plot_phase_operational_criticality(
    beta=1.0,
    alpha_vals=(1.0, 1.25, 1.5, 2.0, 3.0),
    I_vals=(0.25, 0.5, 1.0, 1.5, 2.0),
    theory_kwargs=None,
    plot_dir=None,
    **_unused,
):
    """Plot the smooth transition and cusp covariance at that transition.

    The cusp term generates a finite covariance on both sides of the smooth
    instability, so branch existence is not used as a transition criterion.
    """
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    kwargs = {} if theory_kwargs is None else dict(theory_kwargs)
    kwargs.setdefault("solver", "cusp")
    kwargs.setdefault("q_method", "hermite")
    kwargs.setdefault("n_quad", 48)
    kwargs.setdefault("hermite_order", 32)
    kwargs.setdefault("tau_max", 0.2)
    kwargs.setdefault("dtau", 0.1)
    kwargs.setdefault("warn_on_no_branch", False)

    alphas = np.asarray(alpha_vals, dtype=float)
    inputs = np.asarray(I_vals, dtype=float)
    sigma_smooth = np.array(
        [[_phase_sigma_c(input_value, alpha) for input_value in inputs] for alpha in alphas]
    )
    C0_at_transition = np.full_like(sigma_smooth, np.nan)

    for ia, alpha in enumerate(alphas):
        for ii, (input_value, sigma_c) in enumerate(zip(inputs, sigma_smooth[ia])):
            print(
                f"  cusp covariance at transition: I={input_value:g}, "
                f"alpha={alpha:g}, sigma_c={sigma_c:.4g}"
            )
            _tau, C, _ = theory_phase_autocorr(
                I=input_value,
                alpha=alpha,
                sigma=sigma_c,
                beta=beta,
                **kwargs,
            )
            if len(C) and np.isfinite(C[0]):
                C0_at_transition[ia, ii] = C[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(alphas)))
    for alpha, values, color in zip(alphas, sigma_smooth, colors):
        axes[0].plot(inputs, values, "o-", color=color, lw=1.8, label=fr"$\alpha={alpha:g}$")
    axes[0].set(
        xlabel=r"$I$",
        ylabel=r"estimated critical $\sigma_c$",
        title=r"Smooth-feedback estimate: $\sigma_c=[\rho F'(0)]^{-1}$",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    for alpha, values, color in zip(alphas, C0_at_transition, colors):
        axes[1].plot(
            inputs,
            values,
            "o-",
            color=color,
            lw=1.8,
            label=fr"$\alpha={alpha:g}$",
        )
    axes[1].set(
        xlabel=r"$I$",
        ylabel=r"scalar-cusp $C_{11}(0)$ at estimated $\sigma_c$",
        title=fr"Scalar-cusp covariance at estimate ($\beta={beta:g}$)",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_operational_criticality.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")
    return dict(
        alpha=alphas,
        I=inputs,
        sigma_smooth=sigma_smooth,
        beta=float(beta),
        C0_at_transition=C0_at_transition,
    )


def plot_phase_theory_comparison(
    I=1.0,
    alpha=1.0,
    beta=1.0,
    sigma=None,
    N=256,
    T=900.0,
    dt=0.02,
    dtau=0.1,
    tau_max=35.0,
    burn=300.0,
    sim_reps=2,
    n_probe=None,
    seed=161803,
    plot_dir=None,
    theory_variants=None,
    scale_slow_beta_time=True,
):
    """Overlay simulation with several approximate phase-network theories."""
    import os

    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    sigma_c = _phase_sigma_c(I, alpha)
    if sigma is None:
        sigma = 1.3 * sigma_c

    if theory_variants is None:
        theory_variants = [
            dict(
                label="stationary event-DMFT closure",
                kwargs=dict(solver="density"),
                style=dict(color="C3", ls="--", lw=2.2),
            ),
        ]

    C_runs = []
    tau_s = None
    local_rng = np.random.default_rng(seed)
    if n_probe is None:
        n_probe = N
    slow_factor = _beta_time_factor(beta) if scale_slow_beta_time else 1.0
    for _ in range(max(1, int(sim_reps))):
        tau_run, C_run = sim_phase_network(
            N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
            T=T * slow_factor, dt=dt, burn=burn * slow_factor,
            tau_max=tau_max, n_probe=min(int(n_probe), N), rng=local_rng,
        )
        tau_s = tau_run
        C_runs.append(C_run)
    C_s = np.mean(C_runs, axis=0)
    C_s_norm = C_s / C_s[0] if C_s[0] > 0 else C_s

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        tau_s,
        C_s_norm,
        color="k",
        lw=2.0,
        zorder=5,
        label=f"simulation ({sim_reps} runs)",
    )

    for variant in theory_variants:
        kwargs = dict(variant.get("kwargs", {}))
        style = dict(variant.get("style", {}))
        label = variant.get("label", kwargs.get("solver", "theory"))
        try:
            tau_th, C_th, _ = theory_phase_autocorr(
                I=I, alpha=alpha, sigma=sigma, beta=beta,
                tau_max=tau_max, dtau=dtau, **kwargs,
            )
            norm = max(abs(C_th[0]), 1e-12)
            ax.plot(tau_th, C_th / norm, label=label, **style)
        except Exception as err:
            print(f"  theory variant failed ({label}): {err}")

    ax.axhline(0, color="k", lw=0.5)
    ax.set(
        xlabel=r"$\tau$",
        ylabel=r"$C_{uu}(\tau) / C_{uu}(0)$",
        title=fr"Phase theory comparison: $\sigma={sigma:.2f}$, $g={sigma/sigma_c:.2f}$",
        xlim=(0, tau_max),
        ylim=(-0.35, 1.1),
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_theory_comparison.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


def plot_phase_theory_examples(
    examples=None,
    N=192,
    T=700.0,
    dt=0.02,
    dtau=0.12,
    tau_max=30.0,
    burn=300.0,
    sim_reps=1,
    n_probe=None,
    seed=314159,
    plot_dir=None,
    theory_variants=None,
    scale_slow_beta_time=True,
):
    """Grid of phase-network examples comparing simulation to theory variants."""
    import os

    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    if examples is None:
        examples = []
        for g_val in (1.1, 1.3, 1.5):
            sc = _phase_sigma_c(1.0, 1.0)
            examples.append(dict(I=1.0, alpha=1.0, beta=1.0, sigma=g_val * sc,
                                 label=fr"$\alpha=1,\ \beta=1,\ g={g_val:.1f}$"))
        sc = _phase_sigma_c(1.0, 2.0)
        examples.append(dict(I=1.0, alpha=2.0, beta=1.0, sigma=1.3 * sc,
                             label=fr"$\alpha=2,\ \beta=1,\ g=1.3$"))
        sc = _phase_sigma_c(1.0, 0.5)
        examples.append(dict(I=1.0, alpha=0.5, beta=1.0, sigma=1.3 * sc,
                             label=fr"$\alpha=0.5,\ \beta=1,\ g=1.3$"))
        sc = _phase_sigma_c(1.0, 1.0)
        examples.append(dict(I=1.0, alpha=1.0, beta=0.5, sigma=1.3 * sc,
                             label=fr"$\alpha=1,\ \beta=0.5,\ g=1.3$"))

    if theory_variants is None:
        theory_variants = [
            dict(
                label="stationary event-DMFT closure",
                kwargs=dict(solver="density"),
                style=dict(color="C3", ls="--", lw=1.8),
            ),
        ]

    n = len(examples)
    ncols = 2 if n == 4 else min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows))
    axes_flat = np.array(axes).reshape(-1)
    local_rng = np.random.default_rng(seed)
    if n_probe is None:
        n_probe = N

    for ax, ex in zip(axes_flat, examples):
        I = ex.get("I", 1.0)
        alpha = ex.get("alpha", 1.0)
        beta = ex.get("beta", 1.0)
        sigma = ex["sigma"]
        sigma_c = _phase_sigma_c(I, alpha)
        label = ex.get("label", fr"$g={sigma/sigma_c:.2f}$")
        print(f"  example {label}: sigma={sigma:.3f}, sigma/sigma_c={sigma/sigma_c:.2f}")

        C_runs = []
        tau_s = None
        slow_factor = _beta_time_factor(beta) if scale_slow_beta_time else 1.0
        for _ in range(max(1, int(sim_reps))):
            tau_run, C_run = sim_phase_network(
                N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
                T=T * slow_factor, dt=dt, burn=burn * slow_factor,
                tau_max=tau_max, n_probe=min(int(n_probe), N), rng=local_rng,
            )
            tau_s = tau_run
            C_runs.append(C_run)
        C_s = np.mean(C_runs, axis=0)
        C_s_norm = C_s / C_s[0] if C_s[0] > 0 else C_s
        ax.plot(
            tau_s,
            C_s_norm,
            color="k",
            lw=1.8,
            zorder=5,
            label="simulation",
        )

        for variant in theory_variants:
            kwargs = dict(variant.get("kwargs", {}))
            style = dict(variant.get("style", {}))
            vlabel = variant.get("label", kwargs.get("solver", "theory"))
            try:
                tau_th, C_th, _ = theory_phase_autocorr(
                    I=I, alpha=alpha, sigma=sigma, beta=beta,
                    tau_max=tau_max, dtau=dtau, **kwargs,
                )
                norm = max(abs(C_th[0]), 1e-12)
                ax.plot(tau_th, C_th / norm, label=vlabel, **style)
            except Exception as err:
                print(f"    theory variant failed ({vlabel}): {err}")

        ax.axhline(0, color="k", lw=0.5)
        ax.set(
            xlabel=r"$\tau$",
            ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
            title=label,
            xlim=(0, tau_max),
            ylim=(-0.35, 1.1),
        )
        ax.legend(fontsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle(
        "Phase network: two-time and stationary event-DMFT",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_theory_examples.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


def plot_phase_beta_scaling_diagnostic(
    beta_vals=(0.5, 1.0, 2.0),
    I=1.0,
    alpha=1.0,
    g_val=1.3,
    N=192,
    T=700.0,
    dt=0.02,
    dtau=0.12,
    tau_max=25.0,
    burn=300.0,
    min_T=400.0,
    min_burn=400.0,
    sim_reps=1,
    n_probe=None,
    seed=141421,
    plot_dir=None,
    theory_kwargs=None,
    theory_variants=None,
    scale_slow_beta_time=True,
):
    """Check phase theory and simulation on the scaled time axis beta*tau.

    Here tau_max is interpreted as the maximum beta*tau shown in every panel,
    so the raw simulated/theory lag window is tau_max/beta for each beta.
    """
    import os

    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    if theory_variants is None:
        if theory_kwargs is None:
            theory_kwargs = dict(solver="density")
        theory_variants = [
            dict(
                label="stationary event-DMFT",
                kwargs=dict(theory_kwargs),
            )
        ]

    sigma_c = _phase_sigma_c(I, alpha)
    sigma = g_val * sigma_c

    beta_vals = tuple(float(b) for b in beta_vals)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    ax_raw, ax_scaled = axes
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(beta_vals)))
    max_raw_tau = tau_max / max(min(beta_vals), 1e-12)
    local_rng = np.random.default_rng(seed)
    if n_probe is None:
        n_probe = N

    for beta, color in zip(beta_vals, colors):
        print(f"  beta scaling: beta={beta:g}, sigma={sigma:.3f}, g={g_val:.2f}")
        C_runs = []
        tau_s = None
        slow_factor = _beta_time_factor(beta) if scale_slow_beta_time else 1.0
        raw_tau_max = tau_max / max(float(beta), 1e-12)
        T_run = max(T * slow_factor, float(min_T))
        burn_run = max(burn * slow_factor, float(min_burn))
        for _ in range(max(1, int(sim_reps))):
            tau_run, C_run = sim_phase_network(
                N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
                T=T_run, dt=dt, burn=burn_run,
                tau_max=raw_tau_max,
                n_probe=min(int(n_probe), int(N)),
                rng=local_rng,
            )
            tau_s = tau_run
            C_runs.append(C_run)
        C_s = np.mean(C_runs, axis=0)
        C_s_norm = C_s / max(abs(C_s[0]), 1e-12)
        ax_raw.plot(tau_s, C_s_norm, color=color, lw=1.8,
                    label=fr"sim $\beta={beta:g}$")
        ax_scaled.plot(beta * tau_s, C_s_norm, color=color, lw=1.8,
                       label=fr"sim $\beta={beta:g}$")

        for variant in theory_variants:
            kwargs = dict(variant.get("kwargs", {}))
            label = variant.get("label", kwargs.get("solver", "theory"))
            solver = str(kwargs.get("solver", "")).lower()
            linestyle = "--" if solver == "twotime_dmft" else ":"
            linewidth = 2.3 if solver == "twotime_dmft" else 1.8
            try:
                tau_th, C_th, _ = theory_phase_autocorr(
                    I=I, alpha=alpha, sigma=sigma, beta=beta,
                    tau_max=raw_tau_max, dtau=dtau, **kwargs,
                )
                C_th_norm = C_th / max(abs(C_th[0]), 1e-12)
                ax_raw.plot(
                    tau_th,
                    C_th_norm,
                    color=color,
                    ls=linestyle,
                    lw=linewidth,
                    alpha=0.95,
                )
                ax_scaled.plot(
                    beta * tau_th,
                    C_th_norm,
                    color=color,
                    ls=linestyle,
                    lw=linewidth,
                    alpha=0.95,
                )
            except Exception as err:
                print(f"    theory failed ({label}, beta={beta:g}): {err}")

    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([0], [0], color="k", ls="-", lw=1.8, label="simulation")
    ]
    for variant in theory_variants:
        kwargs = dict(variant.get("kwargs", {}))
        solver = str(kwargs.get("solver", "")).lower()
        method_handles.append(
            Line2D(
                [0],
                [0],
                color="k",
                ls="--" if solver == "twotime_dmft" else ":",
                lw=2.3 if solver == "twotime_dmft" else 1.8,
                label=variant.get("label", kwargs.get("solver", "theory")),
            )
        )
    beta_handles = [
        Line2D([0], [0], color=color, lw=2.5, label=fr"$\beta={beta:g}$")
        for beta, color in zip(beta_vals, colors)
    ]
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
        ax.set(ylim=(-0.35, 1.1))
        ax.legend(handles=method_handles + beta_handles, fontsize=8, ncol=2)

    ax_raw.set(
        xlabel=r"$\tau$",
        ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
        title="Raw time",
        xlim=(0, max_raw_tau),
    )
    ax_scaled.set(
        xlabel=r"$\beta\tau$",
        title="Scaled time",
        xlim=(0, tau_max),
    )

    plt.suptitle(
        fr"Phase beta diagnostic: raw vs scaled time, $\alpha={alpha}$, $g={g_val}$",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_beta_scaling_diagnostic.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


def plot_phase_network_N_convergence(
    N_vals=(64, 128, 256, 512),
    I=1.0,
    alpha=1.0,
    beta=1.0,
    g_val=1.3,
    T=900.0,
    dt=0.02,
    dtau=0.06,
    tau_max=20.0,
    burn=250.0,
    sim_reps=3,
    theory_kwargs=None,
    theory_variants=None,
    seed=2718,
    plot_dir=None,
):
    """Show finite-N convergence of phase-network covariance to DMFT."""
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    if theory_variants is None:
        if theory_kwargs is None:
            theory_kwargs = dict(solver="density")
        theory_variants = [
            dict(
                label="stationary event-DMFT",
                kwargs=dict(theory_kwargs),
                style=dict(color="C3", ls="--", lw=2.0),
            )
        ]

    sigma_c = _phase_sigma_c(I, alpha)
    sigma = g_val * sigma_c
    theories = []
    for variant in theory_variants:
        kwargs = dict(variant.get("kwargs", {}))
        tau_th, C_th, _ = theory_phase_autocorr(
            I=I,
            alpha=alpha,
            sigma=sigma,
            beta=beta,
            tau_max=tau_max,
            dtau=dtau,
            **kwargs,
        )
        theories.append(
            dict(
                label=variant.get("label", kwargs.get("solver", "theory")),
                style=dict(variant.get("style", {})),
                tau=tau_th,
                covariance=C_th,
                normalized=C_th / max(abs(C_th[0]), 1e-12),
            )
        )

    N_vals = np.asarray(N_vals, dtype=int)
    if np.any(N_vals < 2):
        raise ValueError("all network sizes must be at least 2")
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(N_vals)))
    local_rng = np.random.default_rng(seed)
    curves = []
    variance_mean = []
    variance_sem = []

    for N in N_vals:
        print(f"  phase N convergence: N={N}, reps={sim_reps}")
        run_curves = []
        run_variance = []
        tau_sim = None
        for _ in range(max(1, int(sim_reps))):
            tau_sim, covariance = sim_phase_network(
                N=int(N),
                I=I,
                alpha=alpha,
                sigma=sigma,
                beta=beta,
                T=T,
                dt=dt,
                burn=burn,
                tau_max=tau_max,
                n_probe=int(N),
                rng=local_rng,
            )
            normalized = covariance / max(abs(covariance[0]), 1e-12)
            run_curves.append(normalized)
            run_variance.append(float(covariance[0]))

        run_curves = np.asarray(run_curves)
        run_variance = np.asarray(run_variance)
        curves.append(np.mean(run_curves, axis=0))
        variance_mean.append(float(np.mean(run_variance)))
        if len(run_variance) > 1:
            variance_sem.append(
                float(np.std(run_variance, ddof=1) / np.sqrt(len(run_variance)))
            )
        else:
            variance_sem.append(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    ax_cov, ax_error = axes
    for theory in theories:
        ax_cov.plot(
            theory["tau"],
            theory["normalized"],
            label=theory["label"],
            **theory["style"],
        )
    for N, color, covariance in zip(N_vals, colors, curves):
        ax_cov.plot(tau_sim, covariance, color=color, lw=1.6, label=fr"$N={N}$")
    ax_cov.axhline(0.0, color="k", lw=0.5)
    ax_cov.set(
        xlabel=r"$\tau$",
        ylabel=r"$C_{11}(\tau)/C_{11}(0)$",
        title="Drive autocorrelation",
        xlim=(0.0, tau_max),
        ylim=(-0.3, 1.05),
    )
    ax_cov.legend(fontsize=8)

    variance_mean = np.asarray(variance_mean)
    variance_sem = np.asarray(variance_sem)
    ax_error.errorbar(
        N_vals,
        variance_mean,
        yerr=variance_sem,
        fmt="o-",
        color="C3",
        lw=1.8,
        capsize=3,
        label="simulation",
    )
    for theory in theories:
        ax_error.axhline(
            theory["covariance"][0],
            label=theory["label"],
            **theory["style"],
        )
    ax_error.set_xscale("log", base=2)
    ax_error.set_xticks(N_vals, labels=[str(N) for N in N_vals])
    ax_error.set(
        xlabel=r"network size $N$",
        ylabel=r"equal-time variance $C_{11}(0)$",
        title="Finite-size covariance amplitude",
    )
    ax_error.grid(alpha=0.25)
    ax_error.legend(fontsize=8)

    plt.suptitle(
        fr"Deterministic phase network: finite-size convergence "
        fr"($\alpha={alpha:g}$, $\beta={beta:g}$, $g={g_val:g}$)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_network_N_convergence.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")
    return dict(
        N=N_vals,
        variance=variance_mean,
        variance_sem=variance_sem,
        tau_theory=theories[0]["tau"],
        covariance_theory=theories[0]["normalized"],
        theories=theories,
    )


def plot_phase_density_correlation(
    g_vals=(0.25, 0.5, 1.3),
    I=1.0,
    alpha=1.0,
    beta=1.0,
    N=256,
    T=900.0,
    dt=0.02,
    dtau=0.05,
    tau_max=15.0,
    burn=250.0,
    n_probe=192,
    sim_reps=2,
    phase_bin_width=0.25,
    theory_kwargs=None,
    theory_variants=None,
    seed=31415,
    plot_dir=None,
):
    """Compare threshold phase-density C33 in simulations and DMFT."""
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)
    if theory_variants is None:
        if theory_kwargs is None:
            theory_kwargs = dict(solver="density")
        theory_variants = [
            dict(
                label="stationary event-DMFT",
                kwargs=dict(theory_kwargs),
                style=dict(color="C3", ls="--", lw=2.0),
            )
        ]

    sigma_c = _phase_sigma_c(I, alpha)
    g_vals = tuple(float(value) for value in g_vals)
    local_rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(g_vals), figsize=(5.0 * len(g_vals), 4.2), sharey=True)
    if len(g_vals) == 1:
        axes = [axes]

    diagnostics_out = []
    for ax, g_val in zip(axes, g_vals):
        sigma = g_val * sigma_c
        print(f"  phase C33: g={g_val:g}, sigma={sigma:.4g}")
        theory_curves = []
        for variant in theory_variants:
            kwargs = dict(variant.get("kwargs", {}))
            tau_th, _C11, _, diagnostics = theory_phase_autocorr(
                I=I,
                alpha=alpha,
                sigma=sigma,
                beta=beta,
                tau_max=tau_max,
                dtau=dtau,
                phase_bin_width=phase_bin_width,
                return_diagnostics=True,
                **kwargs,
            )
            internal_dt = diagnostics["internal_dt"]
            C33_th_full = diagnostics["phase_density_covariance"]
            tau_internal = np.arange(len(C33_th_full)) * internal_dt
            C33_th = np.interp(tau_th, tau_internal, C33_th_full)
            theory_curves.append(
                dict(
                    tau=tau_th,
                    normalized=C33_th / max(abs(C33_th[0]), 1e-12),
                    label=variant.get(
                        "label", kwargs.get("solver", "theory")
                    ),
                    style=dict(variant.get("style", {})),
                    diagnostics=diagnostics,
                )
            )

        runs = []
        tau_sim = None
        for _ in range(max(1, int(sim_reps))):
            tau_sim, _C11_sim, C33_sim = sim_phase_network(
                N=N,
                I=I,
                alpha=alpha,
                sigma=sigma,
                beta=beta,
                T=T,
                dt=dt,
                burn=burn,
                tau_max=tau_max,
                n_probe=min(int(n_probe), int(N)),
                return_phase_density=True,
                phase_bin_width=phase_bin_width,
                rng=local_rng,
            )
            runs.append(C33_sim / max(abs(C33_sim[0]), 1e-12))
        C33_sim_norm = np.mean(runs, axis=0)

        ax.plot(
            tau_sim,
            C33_sim_norm,
            color="k",
            lw=1.8,
            zorder=5,
            label="simulation",
        )
        for theory in theory_curves:
            ax.plot(
                theory["tau"],
                theory["normalized"],
                label=theory["label"],
                **theory["style"],
            )
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set(
            xlabel=r"$\tau$",
            title=fr"$g={g_val:g}$",
            xlim=(0.0, tau_max),
            ylim=(-0.12, 1.05),
        )
        ax.legend(fontsize=8)
        diagnostics = dict(theory_curves[0]["diagnostics"])
        diagnostics["theory_comparison"] = {
            theory["label"]: theory["diagnostics"] for theory in theory_curves
        }
        diagnostics_out.append(diagnostics)

    axes[0].set_ylabel(
        r"$C_{33}(v_T,v_T,\tau)/C_{33}(v_T,v_T,0)$"
    )
    plt.suptitle(
        fr"Threshold phase-density correlation ($\Delta v={phase_bin_width:g}$)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_density_correlation.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")
    return diagnostics_out


def plot_phase_fixed_q_diagnostic(
    g_vals=(0.25, 0.5, 1.3),
    I=1.0,
    alpha=1.0,
    beta=1.0,
    N=256,
    T=700.0,
    dt=0.02,
    dtau=0.06,
    tau_max=15.0,
    burn=250.0,
    n_probe=192,
    sim_reps=2,
    seed=424242,
    plot_dir=None,
):
    """Plot the rejected fixed-Q diagnostic for development audits only."""
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)
    sigma_c = _phase_sigma_c(I, alpha)
    local_rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(
        2, len(g_vals), figsize=(5.0 * len(g_vals), 7.8), sharex="col"
    )
    if len(g_vals) == 1:
        axes = np.asarray(axes)[:, None]

    diagnostics_out = []
    for column, g_val in enumerate(g_vals):
        sigma = float(g_val) * sigma_c
        simulation_C11 = []
        simulation_C33 = []
        for _rep in range(int(sim_reps)):
            tau_sim, C11_sim, C33_sim = sim_phase_network(
                N=N,
                I=I,
                alpha=alpha,
                sigma=sigma,
                beta=beta,
                T=T,
                dt=dt,
                burn=burn,
                tau_max=tau_max,
                n_probe=min(n_probe, N),
                return_phase_density=True,
                rng=local_rng,
            )
            simulation_C11.append(C11_sim)
            simulation_C33.append(C33_sim)
        C11_sim = np.mean(simulation_C11, axis=0)
        C33_sim = np.mean(simulation_C33, axis=0)

        tau_2pi, C11_2pi, _, diagnostic_2pi = theory_phase_autocorr(
            solver="fixed_q",
            I=I,
            alpha=alpha,
            sigma=sigma,
            beta=beta,
            tau_max=tau_max,
            dtau=dtau,
            max_iter=500,
            mixing=0.12,
            tolerance=2e-6,
            return_diagnostics=True,
        )
        tau_stationary, C11_stationary, _, diagnostic_stationary = (
            theory_phase_autocorr(
                solver="density",
                I=I,
                alpha=alpha,
                sigma=sigma,
                beta=beta,
                tau_max=tau_max,
                dtau=dtau,
                internal_dt=0.02,
                n_time=8192,
                n_samples=64,
                max_iter=50,
                mixing=0.18,
                tolerance=0.03,
                return_diagnostics=True,
                seed=1729 + column,
            )
        )

        ax_C11 = axes[0, column]
        ax_C33 = axes[1, column]
        ax_C11.plot(
            tau_sim,
            C11_sim / max(abs(C11_sim[0]), 1e-12),
            color="C0",
            lw=1.7,
            label="simulation",
        )
        if diagnostic_2pi["converged"] and diagnostic_2pi["physical"]:
            ax_C11.plot(
                tau_2pi,
                C11_2pi / max(abs(C11_2pi[0]), 1e-12),
                color="k",
                lw=2.1,
                label="fixed-Q diagnostic",
            )
        else:
            ax_C11.text(
                0.97,
                0.92,
                "fixed-Q result inadmissible",
                transform=ax_C11.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="k",
            )
        ax_C11.plot(
            tau_stationary,
            C11_stationary / max(abs(C11_stationary[0]), 1e-12),
            color="C3",
            ls="--",
            lw=1.8,
            label="stationary approximation",
        )
        ax_C11.axhline(0.0, color="0.3", lw=0.5)
        ax_C11.set(title=fr"$g={g_val:g}$", ylabel=r"$C_{11}(\tau)/C_{11}(0)$")

        C33_2pi = diagnostic_2pi["phase_density_covariance"]
        tau_2pi_C33 = (
            np.arange(len(C33_2pi)) * diagnostic_2pi["internal_dt"]
        )
        C33_stationary = diagnostic_stationary["phase_density_covariance"]
        tau_stationary_C33 = (
            np.arange(len(C33_stationary))
            * diagnostic_stationary["internal_dt"]
        )
        ax_C33.plot(
            tau_sim,
            C33_sim / max(abs(C33_sim[0]), 1e-12),
            color="C0",
            lw=1.7,
            label="simulation",
        )
        if diagnostic_2pi["converged"] and diagnostic_2pi["physical"]:
            ax_C33.plot(
                tau_2pi_C33,
                C33_2pi / max(abs(C33_2pi[0]), 1e-12),
                color="k",
                lw=2.1,
                label="fixed-Q diagnostic",
            )
        ax_C33.plot(
            tau_stationary_C33,
            C33_stationary / max(abs(C33_stationary[0]), 1e-12),
            color="C3",
            ls="--",
            lw=1.8,
            label="stationary approximation",
        )
        ax_C33.axhline(0.0, color="0.3", lw=0.5)
        ax_C33.set(
            xlabel=r"$\tau$",
            ylabel=r"$C_{33}(v_T,v_T,\tau)/C_{33}(v_T,v_T,0)$",
            xlim=(0.0, tau_max),
        )
        diagnostics_out.append((diagnostic_2pi, diagnostic_stationary))

    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    plt.suptitle(
        "Phase network: fixed-Q diagnostic and Gaussian approximations",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_fixed_q_diagnostic.png")
    plt.savefig(outpath, dpi=180)
    print(f"Saved to {outpath}")
    plt.close(fig)
    return diagnostics_out


def plot_phase_gaussian_2pi_comparison(
    g_vals=(0.25, 0.5, 1.3),
    I=1.0,
    alpha=1.0,
    beta=1.0,
    N=256,
    T=700.0,
    dt=0.02,
    dtau=0.05,
    tau_max=15.0,
    burn=250.0,
    n_probe=192,
    sim_reps=2,
    phase_bin_width=0.25,
    seed=424242,
    n_phase=129,
    stationary_kwargs=None,
    twotime_kwargs=None,
    sim_cache_path=None,
    plot_dir=None,
):
    """Compare the admissible homogeneous Hartree branch with simulation."""
    import csv

    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)
    sigma_c = _phase_sigma_c(I, alpha)
    g_vals = np.asarray(g_vals, dtype=float)
    local_rng = np.random.default_rng(seed)

    cached = None
    if sim_cache_path is not None and os.path.exists(sim_cache_path):
        candidate = np.load(sim_cache_path)
        expected_cache_parameters = dict(
            N=N,
            T=T,
            dt=dt,
            burn=burn,
            sim_reps=sim_reps,
            phase_bin_width=phase_bin_width,
        )
        cache_matches = np.array_equal(candidate["g_vals"], g_vals)
        for name, expected in expected_cache_parameters.items():
            cache_matches = cache_matches and name in candidate
            if cache_matches:
                cache_matches = bool(
                    np.isclose(float(candidate[name]), float(expected))
                )
        if cache_matches:
            cached = candidate
    if cached is None:
        simulation_C11 = []
        simulation_C33 = []
        tau_sim = None
        for g_val in g_vals:
            C11_runs = []
            C33_runs = []
            for _rep in range(int(sim_reps)):
                tau_run, C11_run, C33_run = sim_phase_network(
                    N=N,
                    I=I,
                    alpha=alpha,
                    sigma=float(g_val) * sigma_c,
                    beta=beta,
                    T=T,
                    dt=dt,
                    burn=burn,
                    tau_max=tau_max,
                    n_probe=min(n_probe, N),
                    return_phase_density=True,
                    phase_bin_width=phase_bin_width,
                    rng=local_rng,
                )
                tau_sim = tau_run
                C11_runs.append(C11_run)
                C33_runs.append(C33_run)
            simulation_C11.append(np.mean(C11_runs, axis=0))
            simulation_C33.append(np.mean(C33_runs, axis=0))
        simulation_C11 = np.asarray(simulation_C11)
        simulation_C33 = np.asarray(simulation_C33)
        if sim_cache_path is not None:
            np.savez_compressed(
                sim_cache_path,
                g_vals=g_vals,
                tau=tau_sim,
                C11=simulation_C11,
                C33=simulation_C33,
                N=N,
                T=T,
                dt=dt,
                burn=burn,
                sim_reps=sim_reps,
                phase_bin_width=phase_bin_width,
            )
    else:
        tau_sim = cached["tau"]
        simulation_C11 = cached["C11"]
        simulation_C33 = cached["C33"]

    stationary_options = dict(
        internal_dt=0.02,
        n_time=8192,
        n_samples=64,
        max_iter=50,
        mixing=0.18,
        tolerance=0.03,
    )
    if stationary_kwargs is not None:
        stationary_options.update(stationary_kwargs)
    twotime_options = dict(
        internal_dt=0.05,
        n_time=800,
        n_samples=1024,
        max_iter=140,
        mixing=0.04,
        tolerance=0.04,
        transient_fraction=0.55,
    )
    if twotime_kwargs is not None:
        twotime_options.update(twotime_kwargs)

    fig, axes = plt.subplots(
        2, len(g_vals), figsize=(5.0 * len(g_vals), 8.2), sharex="col"
    )
    if len(g_vals) == 1:
        axes = np.asarray(axes)[:, None]
    metric_rows = []
    diagnostics_out = []

    def normalized(values):
        return values / max(abs(values[0]), 1e-12)

    def normalized_rmse(reference, prediction):
        difference = normalized(reference) - normalized(prediction)
        return float(np.sqrt(np.mean(difference**2)))

    for column, g_val in enumerate(g_vals):
        sigma = float(g_val) * sigma_c
        tau_2pi, C11_2pi, _, diagnostic_2pi = theory_phase_autocorr(
            solver="gaussian_2pi",
            I=I,
            alpha=alpha,
            sigma=sigma,
            beta=beta,
            tau_max=tau_max,
            dtau=dtau,
            n_phase=n_phase,
            return_diagnostics=True,
        )
        tau_stationary, C11_stationary, _, diagnostic_stationary = (
            theory_phase_autocorr(
                solver="density",
                I=I,
                alpha=alpha,
                sigma=sigma,
                beta=beta,
                tau_max=tau_max,
                dtau=dtau,
                phase_bin_width=phase_bin_width,
                return_diagnostics=True,
                seed=1729 + column,
                **stationary_options,
            )
        )
        tau_twotime, C11_twotime, _, diagnostic_twotime = (
            theory_phase_autocorr(
                solver="twotime_dmft",
                I=I,
                alpha=alpha,
                sigma=sigma,
                beta=beta,
                tau_max=tau_max,
                dtau=dtau,
                phase_bin_width=phase_bin_width,
                return_diagnostics=True,
                seed=271828 + column,
                **twotime_options,
            )
        )
        C11_sim = simulation_C11[column]
        C33_sim = simulation_C33[column]
        C33_stationary = diagnostic_stationary["phase_density_covariance"]
        tau_C33_stationary = (
            np.arange(len(C33_stationary))
            * diagnostic_stationary["internal_dt"]
        )
        C33_2pi = diagnostic_2pi["phase_density_covariance"]
        tau_C33_2pi = (
            np.arange(len(C33_2pi)) * diagnostic_2pi["internal_dt"]
        )
        C33_twotime = diagnostic_twotime["phase_density_covariance"]
        tau_C33_twotime = (
            np.arange(len(C33_twotime))
            * diagnostic_twotime["internal_dt"]
        )

        ax_C11 = axes[0, column]
        ax_C33 = axes[1, column]
        marker_step = max(1, len(tau_sim) // 36)
        for axis, values in ((ax_C11, C11_sim), (ax_C33, C33_sim)):
            axis.plot(
                tau_sim,
                normalized(values),
                color="k",
                lw=1.0,
                alpha=0.65,
                zorder=1,
            )
            axis.plot(
                tau_sim,
                normalized(values),
                color="k",
                ls="none",
                marker="o",
                ms=2.8,
                markevery=marker_step,
                label="simulation",
                zorder=5,
            )
        if diagnostic_2pi["stable"]:
            ax_C11.plot(
                tau_2pi,
                normalized(C11_2pi),
                color="0.65",
                ls=":",
                lw=1.5,
                label="Hartree/Wick",
            )
            ax_C33.plot(
                tau_C33_2pi,
                normalized(C33_2pi),
                color="0.65",
                ls=":",
                lw=1.5,
                label="Hartree/Wick",
            )
        else:
            ax_C11.text(
                0.96,
                0.82,
                "Hartree branch unstable",
                transform=ax_C11.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
        ax_C11.plot(
            tau_stationary,
            normalized(C11_stationary),
            color="C3",
            ls="--",
            lw=1.8,
            label="stationary event-DMFT",
        )
        ax_C33.plot(
            tau_C33_stationary,
            normalized(C33_stationary),
            color="C3",
            ls="--",
            lw=1.8,
            label="stationary event-DMFT",
        )
        ax_C11.plot(
            tau_twotime,
            normalized(C11_twotime),
            color="C0",
            ls="-",
            lw=2.1,
            label="two-time event-DMFT",
            zorder=3,
        )
        ax_C33.plot(
            tau_C33_twotime,
            normalized(C33_twotime),
            color="C0",
            ls="-",
            lw=2.1,
            label="two-time event-DMFT",
            zorder=3,
        )
        ax_C11.axhline(0.0, color="0.3", lw=0.5)
        ax_C33.axhline(0.0, color="0.3", lw=0.5)
        ax_C11.set(title=fr"$g={g_val:g}$", ylabel=r"$C_{11}(\tau)/C_{11}(0)$")
        ax_C33.set(
            xlabel=r"$\tau$",
            ylabel=r"$C_{33}(v_T,v_T,\tau)/C_{33}(v_T,v_T,0)$",
            xlim=(0.0, tau_max),
        )

        C11_stationary_on_sim = np.interp(tau_sim, tau_stationary, C11_stationary)
        C33_stationary_on_sim = np.interp(tau_sim, tau_C33_stationary, C33_stationary)
        C11_twotime_on_sim = np.interp(tau_sim, tau_twotime, C11_twotime)
        C33_twotime_on_sim = np.interp(
            tau_sim, tau_C33_twotime, C33_twotime
        )
        row = dict(
            g=float(g_val),
            gaussian_stable=bool(diagnostic_2pi["stable"]),
            gaussian_feedback_eigenvalue=float(diagnostic_2pi["feedback_eigenvalue"]),
            gaussian_critical_sigma=float(diagnostic_2pi["gaussian_critical_sigma"]),
            stationary_C11_rmse=normalized_rmse(C11_sim, C11_stationary_on_sim),
            stationary_C33_rmse=normalized_rmse(C33_sim, C33_stationary_on_sim),
            stationary_C11_variance_error=float(
                abs(C11_stationary[0] - C11_sim[0]) / max(abs(C11_sim[0]), 1e-12)
            ),
            twotime_converged=bool(diagnostic_twotime["converged"]),
            twotime_residual=float(
                diagnostic_twotime["residual_history"][-1]
            ),
            twotime_C11_rmse=normalized_rmse(C11_sim, C11_twotime_on_sim),
            twotime_C33_rmse=normalized_rmse(C33_sim, C33_twotime_on_sim),
            twotime_C11_variance_error=float(
                abs(C11_twotime[0] - C11_sim[0])
                / max(abs(C11_sim[0]), 1e-12)
            ),
            gaussian_C11_rmse=np.nan,
            gaussian_C33_rmse=np.nan,
            gaussian_C11_variance_error=np.nan,
        )
        if diagnostic_2pi["stable"]:
            C11_2pi_on_sim = np.interp(tau_sim, tau_2pi, C11_2pi)
            C33_2pi_on_sim = np.interp(tau_sim, tau_C33_2pi, C33_2pi)
            row.update(
                gaussian_C11_rmse=normalized_rmse(C11_sim, C11_2pi_on_sim),
                gaussian_C33_rmse=normalized_rmse(C33_sim, C33_2pi_on_sim),
                gaussian_C11_variance_error=float(
                    abs(C11_2pi[0] - C11_sim[0]) / max(abs(C11_sim[0]), 1e-12)
                ),
            )
        annotation_style = dict(
            transform=None,
            ha="center",
            va="top",
            fontsize=7.5,
            bbox=dict(
                facecolor="white",
                edgecolor="0.8",
                alpha=0.88,
                pad=2.0,
            ),
        )
        ax_C11.text(
            0.65,
            0.95,
            "RMSE: stationary / two-time\n"
            f"{row['stationary_C11_rmse']:.3f} / "
            f"{row['twotime_C11_rmse']:.3f}",
            **{**annotation_style, "transform": ax_C11.transAxes},
        )
        ax_C33.text(
            0.65,
            0.95,
            "RMSE: stationary / two-time\n"
            f"{row['stationary_C33_rmse']:.3f} / "
            f"{row['twotime_C33_rmse']:.3f}",
            **{**annotation_style, "transform": ax_C33.transAxes},
        )
        metric_rows.append(row)
        diagnostics_out.append(
            (diagnostic_2pi, diagnostic_stationary, diagnostic_twotime)
        )

    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    plt.suptitle(
        "Deterministic phase network: hierarchy of 2PI/DMFT closures",
        fontsize=13,
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_gaussian_2pi_comparison.png")
    plt.savefig(outpath, dpi=180)
    plt.close(fig)

    metrics_path = os.path.join(plot_dir, "phase_gaussian_2pi_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Saved to {outpath}")
    print(f"Saved to {metrics_path}")
    return metric_rows, diagnostics_out


def estimate_phase_C33_validity(
    g_vals=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.3),
    N_vals=(64, 128, 256, 512),
    I=1.0,
    alpha=1.0,
    beta=1.0,
    T=700.0,
    dt=0.02,
    burn=250.0,
    n_probe=192,
    sim_reps=3,
    phase_bin_width=0.25,
    theory_kwargs=None,
    seed=8675309,
    output_path=None,
):
    """Estimate finite-N and closure errors in the first C33 return peak."""
    if theory_kwargs is None:
        theory_kwargs = dict(solver="density")
    else:
        theory_kwargs = dict(theory_kwargs)
    g_vals = np.asarray(g_vals, dtype=float)
    N_vals = np.asarray(N_vals, dtype=int)
    sigma_c = _phase_sigma_c(I, alpha)
    baseline_velocity = alpha * max(float(I), 1e-12) ** (1.0 / alpha)
    baseline_period = 2.0 * np.pi / baseline_velocity
    tau_max = 1.3 * baseline_period
    return_window = (0.7 * baseline_period, 1.2 * baseline_period)
    local_rng = np.random.default_rng(seed)

    peak_samples = np.zeros((len(g_vals), len(N_vals), int(sim_reps)))
    variance_samples = np.zeros_like(peak_samples)
    theory_peak = np.zeros(len(g_vals))
    theory_variance = np.zeros(len(g_vals))

    for ig, g_val in enumerate(g_vals):
        sigma = g_val * sigma_c
        for iN, N in enumerate(N_vals):
            print(f"  C33 validity: g={g_val:g}, N={N}, reps={sim_reps}")
            for rep in range(int(sim_reps)):
                tau, C11, C33 = sim_phase_network(
                    N=int(N),
                    I=I,
                    alpha=alpha,
                    sigma=sigma,
                    beta=beta,
                    T=T,
                    dt=dt,
                    burn=burn,
                    tau_max=tau_max,
                    n_probe=min(int(n_probe), int(N)),
                    return_phase_density=True,
                    phase_bin_width=phase_bin_width,
                    rng=local_rng,
                )
                normalized = C33 / max(abs(C33[0]), 1e-12)
                mask = (tau >= return_window[0]) & (tau <= return_window[1])
                peak_samples[ig, iN, rep] = np.max(normalized[mask])
                variance_samples[ig, iN, rep] = C11[0]

        tau_th, C11_th, _, diagnostics = theory_phase_autocorr(
            I=I,
            alpha=alpha,
            sigma=sigma,
            beta=beta,
            tau_max=tau_max,
            dtau=dt,
            phase_bin_width=phase_bin_width,
            return_diagnostics=True,
            **theory_kwargs,
        )
        C33_th = diagnostics["phase_density_covariance"]
        C33_th = C33_th / max(abs(C33_th[0]), 1e-12)
        tau_density = np.arange(len(C33_th)) * diagnostics["internal_dt"]
        mask = (
            (tau_density >= return_window[0])
            & (tau_density <= return_window[1])
        )
        theory_peak[ig] = np.max(C33_th[mask])
        theory_variance[ig] = C11_th[0]

    inverse_sqrt_N = 1.0 / np.sqrt(N_vals)
    peak_mean = np.mean(peak_samples, axis=2)
    variance_mean = np.mean(variance_samples, axis=2)
    peak_infinite = np.array(
        [np.polyfit(inverse_sqrt_N, row, 1)[1] for row in peak_mean]
    )
    variance_infinite = np.array(
        [np.polyfit(inverse_sqrt_N, row, 1)[1] for row in variance_mean]
    )
    peak_absolute_error = np.abs(theory_peak - peak_infinite)
    variance_relative_error = np.divide(
        np.abs(theory_variance - variance_infinite),
        np.maximum(np.abs(theory_variance), 1e-12),
    )

    table = np.column_stack(
        (
            g_vals,
            peak_infinite,
            theory_peak,
            peak_absolute_error,
            variance_infinite,
            theory_variance,
            variance_relative_error,
        )
    )
    if output_path is not None:
        np.savetxt(
            output_path,
            table,
            delimiter=",",
            header=(
                "g,C33_first_return_Ninf,C33_first_return_theory,"
                "C33_absolute_error,C11_zero_Ninf,C11_zero_theory,"
                "C11_relative_error"
            ),
            comments="",
        )
        print(f"Saved to {output_path}")
    return dict(
        table=table,
        N=N_vals,
        peak_samples=peak_samples,
        variance_samples=variance_samples,
    )


# -----------------------------------------------------------------------------
# Phase model: timeseries and raster helpers
# -----------------------------------------------------------------------------

def _sim_phase_timeseries(
    N=256,
    I=1.0,
    alpha=1.0,
    sigma=7.0,
    beta=1.0,
    T=500.0,
    dt=0.02,
    n_show=20,
    burn=200.0,
    synapse_update="exact",
    max_recorded_spikes_per_bin=None,
    rng_=None,
):
    """Simulate phase network and return raw u timeseries and per-neuron spike times.

    Returns
    -------
    t         : (nt,) time array
    U         : (nt, n_show) u-values for the first n_show neurons
    spk_times : list of n_show arrays, each holding spike times for that neuron
    """
    if rng_ is None:
        rng_ = np.random.default_rng(42)
    n_show = min(n_show, N)
    W = make_weights(N, sigma, lam=1, rng=rng_)
    phi = rng_.uniform(-np.pi, np.pi, N)
    u = np.zeros(N)
    synapse_update = str(synapse_update).lower()
    if synapse_update not in ("exact", "euler"):
        raise ValueError("synapse_update must be 'exact' or 'euler'")

    def F(u_):
        return alpha * np.clip(I + u_, 0.0, 1e12) ** (1.0 / alpha)

    def spike_weights(phi_old, rate, spike_counts):
        if synapse_update == "euler":
            return spike_counts
        weighted = np.zeros_like(spike_counts, dtype=float)
        spiking = np.flatnonzero(spike_counts > 0)
        for j in spiking:
            count = int(spike_counts[j])
            if count <= 0 or rate[j] <= 0.0:
                continue
            thresholds = np.pi + 2.0 * np.pi * np.arange(count)
            crossing_times = (thresholds - phi_old[j]) / rate[j]
            crossing_times = np.clip(crossing_times, 0.0, dt)
            weighted[j] = np.sum(np.exp(-beta * (dt - crossing_times)))
        return weighted

    def update_u(u_, drive_):
        if synapse_update == "exact":
            return np.exp(-beta * dt) * u_ + beta * drive_
        return u_ + (-beta * u_) * dt + beta * drive_

    nb = int(burn / dt)
    for _ in range(nb):
        phi_old = phi.copy()
        rate = F(u)
        phi = phi_old + rate * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        drive = W @ spike_weights(phi_old, rate, spike_counts)
        u = update_u(u, drive)
        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)

    nt = int(T / dt)
    t = np.arange(nt) * dt
    idx = np.arange(n_show)
    U = np.zeros((nt, n_show))
    spk_times = [[] for _ in range(n_show)]

    for step in range(nt):
        phi_old = phi.copy()
        rate = F(u)
        phi = phi_old + rate * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        drive = W @ spike_weights(phi_old, rate, spike_counts)
        u = update_u(u, drive)
        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
        U[step] = u[idx]
        for k in range(n_show):
            count = int(spike_counts[idx[k]])
            if count:
                if max_recorded_spikes_per_bin is not None:
                    count = min(count, int(max_recorded_spikes_per_bin))
                spk_times[k].extend([t[step]] * count)

    return t, U, [np.array(st) for st in spk_times]


def _phase_sigma_c(I, alpha):
    rho = 1.0 / (2.0 * np.pi)
    Fprime0 = float(np.maximum(I, 1e-10) ** (1.0 / alpha - 1.0))
    return 1.0 / (rho * Fprime0)


def _beta_time_factor(beta):
    """Simulation-time multiplier for slow synapses.

    The u-filter relaxation time is O(1/beta).  Comparisons at beta < 1 need
    proportionally longer burn-in and recording windows to estimate the same
    number of effective correlation times.
    """
    return max(1.0, 1.0 / max(float(beta), 1e-12))


# -----------------------------------------------------------------------------
# 1. u(t) time series
# -----------------------------------------------------------------------------

def plot_u_timeseries(
    sigma_vals=None,
    I=1.0,
    alpha=1.0,
    beta=1.0,
    N=256,
    T=200.0,
    dt=0.02,
    n_show=8,
    burn=100.0,
    synapse_update="exact",
    max_recorded_spikes_per_bin=1,
    plot_dir=None,
):
    """Plot u(t) traces for several neurons across sigma values.

    One column per sigma value; each panel shows n_show overlaid traces.
    """
    import os
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    sigma_c = _phase_sigma_c(I, alpha)
    if sigma_vals is None:
        sigma_vals = [0.5 * sigma_c, 1.0 * sigma_c, 2.0 * sigma_c]

    ncols = len(sigma_vals)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, sigma in zip(axes, sigma_vals):
        g = sigma / sigma_c
        print(f"  u timeseries: sigma={sigma:.2f} (g={g:.2f})")
        t, U, _ = _sim_phase_timeseries(
            N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
            T=T, dt=dt, n_show=n_show, burn=burn,
            synapse_update=synapse_update,
            max_recorded_spikes_per_bin=max_recorded_spikes_per_bin,
        )
        for k in range(n_show):
            ax.plot(t, U[:, k], lw=0.8, alpha=0.7)
        ax.set(
            xlabel="time",
            ylabel=r"$u_i(t)$",
            title=fr"$\sigma={sigma:.2f}$, $g=\sigma/\sigma_c={g:.2f}$",
        )

    plt.suptitle(
        fr"Phase network: $u(t)$ time series  ($I={I}$, $\alpha={alpha}$, $\sigma_c={sigma_c:.2f}$)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_u_timeseries.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


# -----------------------------------------------------------------------------
# 2. Spike raster + population rate
# -----------------------------------------------------------------------------

def plot_phase_raster(
    sigma_vals=None,
    I=1.0,
    alpha=1.0,
    beta=1.0,
    N=256,
    T=500.0,
    dt=0.02,
    burn=100.0,
    synapse_update="exact",
    max_recorded_spikes_per_bin=1,
    plot_dir=None,
):
    """Spike raster (neuron index vs time) + population rate for several sigma.

    Two rows per sigma: raster (top), smoothed population firing rate (bottom).
    """
    import os
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    sigma_c = _phase_sigma_c(I, alpha)
    if sigma_vals is None:
        sigma_vals = [0.5 * sigma_c, 1.0 * sigma_c, 2.0 * sigma_c]

    ncols = len(sigma_vals)
    fig, axes = plt.subplots(
        2, ncols, figsize=(5 * ncols, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for col, sigma in enumerate(sigma_vals):
        g = sigma / sigma_c
        print(f"  raster: sigma={sigma:.2f} (g={g:.2f})")
        t, _, spk_times = _sim_phase_timeseries(
            N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
            T=T, dt=dt, n_show=N, burn=burn,
            synapse_update=synapse_update,
            max_recorded_spikes_per_bin=max_recorded_spikes_per_bin,
        )
        ax_raster = axes[0, col]
        ax_rate   = axes[1, col]

        # Raster
        for k, st in enumerate(spk_times):
            if len(st):
                ax_raster.scatter(st, np.full(len(st), k), s=0.5, c="k", linewidths=0)
        ax_raster.set(
            xlim=(0, T), ylim=(-1, N),
            ylabel="neuron" if col == 0 else "",
            title=fr"$\sigma={sigma:.2f}$, $g={g:.2f}$",
        )
        ax_raster.tick_params(labelbottom=False)

        # Population rate (smoothed)
        nt = len(t)
        pop_rate = np.zeros(nt)
        for st in spk_times:
            for ts in st:
                idx = int(ts / dt)
                if idx < nt:
                    pop_rate[idx] += 1.0
        pop_rate /= N * dt
        win = max(1, int(5.0 / dt))
        pop_rate_sm = np.convolve(pop_rate, np.ones(win) / win, mode="same")
        ax_rate.plot(t, pop_rate_sm, lw=1.0, color="steelblue")
        ax_rate.set(
            xlabel="time",
            ylabel="rate" if col == 0 else "",
            xlim=(0, T),
        )

    plt.suptitle(
        fr"Phase network: spike raster  ($I={I}$, $\alpha={alpha}$, $\sigma_c={sigma_c:.2f}$)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_raster.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


# -----------------------------------------------------------------------------
# Parallel simulation helper (module-level so ProcessPoolExecutor can pickle it)
# -----------------------------------------------------------------------------
