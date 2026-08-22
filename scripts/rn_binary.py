"""Binary-neuron simulations, affine benchmark, and sigmoid DMFT."""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from rn_core import autocorr, default_results_dir, make_weights, rng


def sigmoid_rate(u, rate_max=1.0, theta=0.0, delta=0.25):
    """Bounded positive activation rate used by theory and simulation."""
    if rate_max <= 0.0:
        raise ValueError("rate_max must be positive")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    return rate_max * expit((np.asarray(u) - theta) / delta)


def sigmoid_tangent_parameters(rate_max=1.0, theta=0.0, delta=0.25):
    """Return the intercept and slope of the sigmoid at zero drive."""
    p0 = float(expit(-theta / delta))
    return rate_max * p0, rate_max * p0 * (1.0 - p0) / delta


def sim_binary_network(
    N=1000,
    sigma=0.8,
    beta=1.0,
    mu=1.0,
    rate_max=1.0,
    theta=0.0,
    delta=0.25,
    T=1000.0,
    dt=0.02,
    lam=1,
    burn=200.0,
    n_probe=64,
    tau_max=30.0,
    return_spike=False,
    rng=rng,
):
    """Simulate the sigmoid binary network and return connected covariances.

    The state is held fixed over each interval and the synaptic filter is
    advanced exactly. Binary transitions use the exact constant-rate
    transition probability over the interval.
    """
    if dt <= 0.0 or T <= 0.0:
        raise ValueError("dt and T must be positive")
    if beta <= 0.0 or mu <= 0.0:
        raise ValueError("beta and mu must be positive")

    W = make_weights(N, sigma, lam, rng)
    n = rng.integers(0, 2, N).astype(float)
    u = np.zeros(N)
    drive = W @ n
    decay_u = np.exp(-beta * dt)

    def step(record_spikes=False):
        nonlocal n, u, drive
        rate = sigmoid_rate(u, rate_max=rate_max, theta=theta, delta=delta)
        gamma = rate + mu
        relax = 1.0 - np.exp(-gamma * dt)
        p_on = (rate / gamma) * relax
        p_off = (mu / gamma) * relax
        draws = rng.random(N)
        old_n = n.copy()
        n = np.where(old_n > 0.5, draws >= p_off, draws < p_on).astype(float)
        delta_n = n - old_n
        changed = np.flatnonzero(delta_n)
        if len(changed):
            drive += W[:, changed] @ delta_n[changed]
        u = decay_u * u + (1.0 - decay_u) * drive
        if record_spikes:
            return (old_n < 0.5) & (n > 0.5)
        return None

    for _ in range(int(burn / dt)):
        step()

    nt = int(T / dt)
    n_probe = int(max(1, min(N, n_probe)))
    states = np.empty((nt, n_probe), dtype=np.float32)
    drives = np.empty((nt, n_probe), dtype=np.float32)
    spikes = np.empty((nt, n_probe), dtype=np.float32) if return_spike else None
    for index in range(nt):
        events = step(record_spikes=return_spike)
        states[index] = n[:n_probe]
        drives[index] = u[:n_probe]
        if return_spike:
            spikes[index] = events[:n_probe]

    max_lag = min(int(tau_max / dt), nt - 1)
    Cnn = np.mean([autocorr(states[:, i], max_lag) for i in range(n_probe)], axis=0)
    Cuu = np.mean([autocorr(drives[:, i], max_lag) for i in range(n_probe)], axis=0)
    tau = np.arange(len(Cnn)) * dt
    if not return_spike:
        return tau, Cnn, Cuu
    Cspk = np.mean(
        [autocorr(spikes[:, i] / dt, max_lag) for i in range(n_probe)], axis=0
    )
    return tau, Cnn, Cuu, Cspk


def theory_binary_autocorr(sigma, beta, mu, f0, f1, tau_max=30, dtau=0.001):
    """Closed Gaussian 2PI result for the formal affine tangent model."""
    gamma = mu + f0
    n_bar = f0 / gamma
    c1 = f1 * mu / gamma
    D0 = 2.0 * n_bar * (1.0 - n_bar) * gamma
    g = c1 * sigma / gamma

    disc = (gamma**2 - beta**2) ** 2 + 4.0 * c1**2 * beta**2 * sigma**2
    kp2 = 0.5 * ((gamma**2 + beta**2) + np.sqrt(disc))
    km2 = 0.5 * ((gamma**2 + beta**2) - np.sqrt(disc))
    tau = np.arange(0.0, tau_max, dtau)

    if km2 > 0.0:
        kp, km = np.sqrt(kp2), np.sqrt(km2)
        Ap = D0 * 0.5 * (beta**2 - kp2) / (kp * (km2 - kp2))
        Am = D0 * 0.5 * (beta**2 - km2) / (km * (kp2 - km2))
        Cnn = Ap * np.exp(-kp * tau) + Am * np.exp(-km * tau)

        def filtered_term(A, k):
            denominator = beta**2 - k**2
            if abs(denominator) < 1e-10:
                return A * sigma**2 * 0.5 * beta * (1.0 + beta * tau) * np.exp(-beta * tau)
            return A * beta**2 * sigma**2 / denominator * (
                np.exp(-k * tau) - (k / beta) * np.exp(-beta * tau)
            )

        Cuu = filtered_term(Ap, kp) + filtered_term(Am, km)
    else:
        # At and above the tangent instability the zero-frequency denominator
        # is nonpositive, so no stationary positive-semidefinite covariance
        # exists within the affine Gaussian theory.
        Cnn = np.full_like(tau, np.nan)
        Cuu = np.full_like(tau, np.nan)

    return tau, Cnn, Cuu, g


def theory_binary_sigmoid_tangent(
    sigma,
    beta=1.0,
    mu=1.0,
    rate_max=1.0,
    theta=0.0,
    delta=0.25,
    tau_max=30.0,
    dtau=0.01,
):
    """Affine 2PI benchmark obtained from the sigmoid tangent at zero drive."""
    f0, f1 = sigmoid_tangent_parameters(rate_max, theta, delta)
    return theory_binary_autocorr(sigma, beta, mu, f0, f1, tau_max, dtau)


def _sample_stationary_gaussian(spectrum, normals, n_time):
    spectrum = np.maximum(np.asarray(spectrum, dtype=float), 0.0)
    coefficients = normals * np.sqrt(n_time * spectrum)[None, :]
    return np.fft.irfft(coefficients, n=n_time, axis=1)


def _periodic_binary_probability(rates, mu, dt):
    """Return the periodic conditional active-state probability.

    The rate is held constant on each time interval.  The resulting affine
    probability update is the exact matrix exponential of the two-state
    generator, and the initial probability is chosen to be periodic.
    """
    gamma = rates + mu
    decay = np.exp(-gamma * dt)
    relax = 1.0 - decay
    p_on = rates / gamma * relax

    # Solve the affine probability map over one periodic drive realization.
    accumulated = np.zeros(rates.shape[0])
    multiplier = np.ones(rates.shape[0])
    for index in range(rates.shape[1]):
        accumulated = decay[:, index] * accumulated + p_on[:, index]
        multiplier *= decay[:, index]
    p_initial = accumulated / np.maximum(1.0 - multiplier, 1e-15)

    probability = p_initial
    probabilities = np.empty(rates.shape, dtype=float)
    for index in range(rates.shape[1]):
        probabilities[:, index] = probability
        probability = decay[:, index] * probability + p_on[:, index]
    return probabilities, decay


def _conditional_binary_spectrum(rates, mu, dt, survival_tolerance=1e-10):
    """Evaluate the conditional two-state covariance without jump sampling.

    For a fixed periodic drive path, the two-state propagator has memory
    ``prod(decay)``.  The total covariance is the sum of the covariance of the
    conditional means and the conditional Bernoulli covariance.  Averaging
    these terms over paths is a Rao-Blackwellized evaluation of Q_bin.
    """
    probabilities, decay = _periodic_binary_probability(rates, mu, dt)
    n_time = rates.shape[1]

    centered = probabilities - np.mean(probabilities)
    transformed = np.fft.rfft(centered, axis=1)
    mean_spectrum = np.mean(np.abs(transformed) ** 2, axis=0) / n_time

    local_covariance = np.zeros(n_time, dtype=float)
    survival = probabilities * (1.0 - probabilities)
    local_covariance[0] = float(np.mean(survival))
    reference = max(local_covariance[0], 1e-30)
    max_lag = n_time // 2
    for lag in range(1, max_lag + 1):
        survival *= np.roll(decay, -(lag - 1), axis=1)
        value = float(np.mean(survival))
        local_covariance[lag] = value
        if lag != n_time - lag:
            local_covariance[-lag] = value
        if value <= survival_tolerance * reference:
            break

    local_spectrum = np.real(np.fft.rfft(local_covariance))
    spectrum = np.maximum(mean_spectrum + local_spectrum, 0.0)
    return spectrum, probabilities, local_covariance


def theory_binary_sigmoid_dmft(
    sigma,
    beta=1.0,
    mu=1.0,
    rate_max=1.0,
    theta=0.0,
    delta=0.25,
    tau_max=30.0,
    dtau=0.05,
    internal_dt=0.05,
    n_time=4096,
    n_samples=64,
    max_iter=40,
    mixing=0.2,
    tolerance=0.02,
    seed=2718,
    return_diagnostics=False,
):
    """Solve the stationary large-N sigmoid DMFT saddle.

    Gaussian disorder averaging gives a single-site synaptic drive whose
    covariance is ``sigma**2 Cnn`` filtered by the synapse. For every sampled
    Gaussian drive, the conditional active-state probability and two-time
    propagator are evaluated with the exact constant-rate transition matrix.
    Iterating the resulting conditional state spectrum closes the disorder
    saddle without sampling jump paths, introducing effective rates, or
    inserting a telegraph covariance.
    """
    if internal_dt <= 0.0 or dtau <= 0.0:
        raise ValueError("internal_dt and dtau must be positive")
    if beta <= 0.0 or mu <= 0.0:
        raise ValueError("beta and mu must be positive")
    n_time = int(max(256, n_time))
    n_samples = int(max(4, n_samples))
    dt = float(internal_dt)

    logistic0 = float(expit(-theta / delta))
    rate0 = rate_max * logistic0
    gamma0 = rate0 + mu
    mean0 = rate0 / gamma0
    slope0 = rate_max * logistic0 * (1.0 - logistic0) / delta
    susceptibility0 = mu * slope0 / gamma0**2
    sigma_critical = np.inf if susceptibility0 <= 0.0 else 1.0 / susceptibility0
    output_tau = np.arange(0.0, tau_max, dtau)

    if sigma == 0.0:
        Cnn = mean0 * (1.0 - mean0) * np.exp(-gamma0 * output_tau)
        Cuu = np.zeros_like(Cnn)
        diagnostics = dict(
            converged=True,
            iterations=0,
            residual_history=np.array([], dtype=float),
            final_residual=0.0,
            mean_activity=mean0,
            sigma_critical_tangent=sigma_critical,
            internal_dt=dt,
            n_time=n_time,
            n_samples=0,
            conditional_method="exact_master_equation",
        )
        if return_diagnostics:
            return output_tau, Cnn, Cuu, diagnostics
        return output_tau, Cnn, Cuu, sigma_critical

    local_rng = np.random.default_rng(seed)
    n_freq = n_time // 2 + 1
    normals = (
        local_rng.normal(size=(n_samples, n_freq))
        + 1j * local_rng.normal(size=(n_samples, n_freq))
    ) / np.sqrt(2.0)
    normals[:, 0] = local_rng.normal(size=n_samples)
    if n_time % 2 == 0:
        normals[:, -1] = local_rng.normal(size=n_samples)
    circular_lag = np.minimum(np.arange(n_time), n_time - np.arange(n_time)) * dt
    initial_covariance = mean0 * (1.0 - mean0) * np.exp(-gamma0 * circular_lag)
    state_spectrum = np.maximum(np.real(np.fft.rfft(initial_covariance)), 0.0)

    omega_dt = 2.0 * np.pi * np.fft.rfftfreq(n_time)
    synaptic_decay = np.exp(-beta * dt)
    synaptic_filter = (1.0 - synaptic_decay) ** 2 / np.maximum(
        1.0 + synaptic_decay**2
        - 2.0 * synaptic_decay * np.cos(omega_dt),
        1e-30,
    )

    history = []
    converged = False
    mean_activity = mean0
    probabilities = None
    intrinsic_covariance = None
    drive_spectrum = np.zeros_like(state_spectrum)
    for iteration in range(int(max_iter)):
        drive_spectrum = sigma**2 * synaptic_filter * state_spectrum
        drive = _sample_stationary_gaussian(drive_spectrum, normals, n_time)
        rates = sigmoid_rate(drive, rate_max, theta, delta)
        proposed, probabilities, intrinsic_covariance = (
            _conditional_binary_spectrum(rates, mu, dt)
        )
        mean_activity = float(np.mean(probabilities))
        scale = max(float(np.linalg.norm(state_spectrum)), 1e-12)
        residual = float(np.linalg.norm(proposed - state_spectrum) / scale)
        history.append(residual)
        state_spectrum = (1.0 - mixing) * state_spectrum + mixing * proposed
        if residual < tolerance:
            converged = True
            break

    state_covariance = np.fft.irfft(state_spectrum, n=n_time)
    drive_spectrum = sigma**2 * synaptic_filter * state_spectrum
    drive_covariance = np.fft.irfft(drive_spectrum, n=n_time)
    max_internal_lag = min(int(np.ceil(tau_max / dt)) + 1, n_time // 2)
    internal_tau = np.arange(max_internal_lag) * dt
    Cnn = np.interp(output_tau, internal_tau, state_covariance[:max_internal_lag])
    Cuu = np.interp(output_tau, internal_tau, drive_covariance[:max_internal_lag])

    diagnostics = dict(
        converged=converged,
        iterations=iteration + 1,
        residual_history=np.asarray(history),
        final_residual=float(history[-1]),
        mean_activity=mean_activity,
        sigma_critical_tangent=sigma_critical,
        internal_dt=dt,
        n_time=n_time,
        n_samples=n_samples,
        state_spectrum=state_spectrum,
        drive_spectrum=drive_spectrum,
        sample_probabilities=probabilities,
        intrinsic_covariance=intrinsic_covariance,
        conditional_method="exact_master_equation",
    )
    if return_diagnostics:
        return output_tau, Cnn, Cuu, diagnostics
    return output_tau, Cnn, Cuu, sigma_critical


def _normalized(covariance):
    covariance = np.asarray(covariance, dtype=float)
    return covariance / covariance[0] if abs(covariance[0]) > 1e-12 else covariance


def _binary_simulation_average(
    sigma, reps, seed, *, N, beta, mu, rate_max, theta, delta, T, burn, dt, tau_max
):
    runs = [
        sim_binary_network(
            N=N,
            sigma=sigma,
            beta=beta,
            mu=mu,
            rate_max=rate_max,
            theta=theta,
            delta=delta,
            T=T,
            burn=burn,
            dt=dt,
            tau_max=tau_max,
            rng=np.random.default_rng(seed + rep),
        )
        for rep in range(int(reps))
    ]
    return runs[0][0], np.mean([run[1] for run in runs], axis=0), np.mean(
        [run[2] for run in runs], axis=0
    )


def plot_binary_network(
    sigma_vals=(1.0, 2.0, 3.0),
    N=800,
    beta=1.0,
    mu=1.0,
    rate_max=1.0,
    theta=0.0,
    delta=0.25,
    T=1500.0,
    burn=250.0,
    dt=0.02,
    tau_max=20.0,
    sim_reps=2,
    theory_kwargs=None,
    plot_dir=None,
):
    """Compare the sigmoid DMFT prediction with matched simulations."""
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)
    theory_kwargs = {} if theory_kwargs is None else dict(theory_kwargs)
    fig, axes = plt.subplots(2, len(sigma_vals), figsize=(5.0 * len(sigma_vals), 7.5))
    if len(sigma_vals) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column, sigma in enumerate(sigma_vals):
        tau_th, Cnn_th, Cuu_th, diagnostics = theory_binary_sigmoid_dmft(
            sigma=sigma,
            beta=beta,
            mu=mu,
            rate_max=rate_max,
            theta=theta,
            delta=delta,
            tau_max=tau_max,
            return_diagnostics=True,
            **theory_kwargs,
        )
        tau_sim, Cnn_sim, Cuu_sim = _binary_simulation_average(
            sigma,
            sim_reps,
            1000 + 31 * column,
            N=N,
            beta=beta,
            mu=mu,
            rate_max=rate_max,
            theta=theta,
            delta=delta,
            T=T,
            burn=burn,
            dt=dt,
            tau_max=tau_max,
        )
        g0 = sigma / diagnostics["sigma_critical_tangent"]
        axes[0, column].plot(
            tau_sim, _normalized(Cnn_sim), color="k", lw=1.8, label="simulation"
        )
        axes[0, column].plot(
            tau_th, _normalized(Cnn_th), color="C3", lw=2.2, label="dynamic DMFT"
        )
        axes[0, column].set(
            title=fr"$\sigma={sigma:g},\ g_0={g0:.2f}$",
            ylabel=r"$C_{nn}(\tau)/C_{nn}(0)$",
            xlim=(0, tau_max),
        )
        axes[0, column].legend(fontsize=8)
        axes[1, column].plot(
            tau_sim, _normalized(Cuu_sim), color="k", lw=1.8, label="simulation"
        )
        axes[1, column].plot(
            tau_th, _normalized(Cuu_th), color="C3", lw=2.2, label="dynamic DMFT"
        )
        axes[1, column].set(
            xlabel=r"$\tau$",
            ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
            xlim=(0, tau_max),
        )
        axes[1, column].legend(fontsize=8)

    fig.suptitle("Sigmoid binary network: dynamic 2PI-DMFT", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output = os.path.join(plot_dir, "binary_network_test.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved to {output}")


def plot_binary_network_N_convergence(
    sigma=2.0,
    N_vals=(128, 300, 800, 1600),
    beta=1.0,
    mu=1.0,
    rate_max=1.0,
    theta=0.0,
    delta=0.25,
    T=1200.0,
    burn=250.0,
    dt=0.02,
    tau_max=20.0,
    theory_kwargs=None,
    plot_dir=None,
):
    """Compare finite networks with the same large-N sigmoid DMFT saddle."""
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)
    theory_kwargs = {} if theory_kwargs is None else dict(theory_kwargs)
    tau_th, Cnn_th, Cuu_th, diagnostics = theory_binary_sigmoid_dmft(
        sigma=sigma,
        beta=beta,
        mu=mu,
        rate_max=rate_max,
        theta=theta,
        delta=delta,
        tau_max=tau_max,
        return_diagnostics=True,
        **theory_kwargs,
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_vals)))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(tau_th, _normalized(Cnn_th), color="C3", lw=2.8, label="dynamic DMFT")
    axes[1].plot(tau_th, _normalized(Cuu_th), color="C3", lw=2.8, label="dynamic DMFT")
    for index, (N, color) in enumerate(zip(N_vals, colors)):
        tau, Cnn, Cuu = sim_binary_network(
            N=N,
            sigma=sigma,
            beta=beta,
            mu=mu,
            rate_max=rate_max,
            theta=theta,
            delta=delta,
            T=T,
            burn=burn,
            dt=dt,
            tau_max=tau_max,
            rng=np.random.default_rng(5000 + index),
        )
        axes[0].plot(tau, _normalized(Cnn), color=color, lw=1.4, label=fr"$N={N}$")
        axes[1].plot(tau, _normalized(Cuu), color=color, lw=1.4, label=fr"$N={N}$")
    axes[0].set(
        xlabel=r"$\tau$", ylabel=r"$C_{nn}(\tau)/C_{nn}(0)$",
        title="state covariance", xlim=(0, tau_max)
    )
    axes[1].set(
        xlabel=r"$\tau$", ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
        title="drive covariance", xlim=(0, tau_max)
    )
    for axis in axes:
        axis.legend(fontsize=8)
    g0 = sigma / diagnostics["sigma_critical_tangent"]
    fig.suptitle(
        fr"Sigmoid binary network: finite-size convergence ($\sigma={sigma:g},\ g_0={g0:.2f}$)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    output = os.path.join(plot_dir, "binary_network_N_convergence.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved to {output}")


def plot_binary_theory_hierarchy(
    sigma_vals=(1.0, 2.0, 3.0),
    N=800,
    beta=1.0,
    mu=1.0,
    rate_max=1.0,
    theta=0.0,
    delta=0.25,
    T=1500.0,
    burn=250.0,
    dt=0.02,
    tau_max=20.0,
    theory_kwargs=None,
    plot_dir=None,
):
    """Compare nonlinear DMFT with its controlled sigmoid-tangent limit."""
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)
    theory_kwargs = {} if theory_kwargs is None else dict(theory_kwargs)
    fig, axes = plt.subplots(1, len(sigma_vals), figsize=(5.0 * len(sigma_vals), 4.2))
    if len(sigma_vals) == 1:
        axes = [axes]
    for index, (axis, sigma) in enumerate(zip(axes, sigma_vals)):
        tau_dmft, Cnn_dmft, _Cuu, diagnostics = theory_binary_sigmoid_dmft(
            sigma=sigma,
            beta=beta,
            mu=mu,
            rate_max=rate_max,
            theta=theta,
            delta=delta,
            tau_max=tau_max,
            return_diagnostics=True,
            **theory_kwargs,
        )
        tau_affine, Cnn_affine, _Cuu_affine, _g = theory_binary_sigmoid_tangent(
            sigma=sigma,
            beta=beta,
            mu=mu,
            rate_max=rate_max,
            theta=theta,
            delta=delta,
            tau_max=tau_max,
            dtau=max(dt, 0.05),
        )
        tau_sim, Cnn_sim, _Cuu_sim = _binary_simulation_average(
            sigma,
            1,
            8000 + index,
            N=N,
            beta=beta,
            mu=mu,
            rate_max=rate_max,
            theta=theta,
            delta=delta,
            T=T,
            burn=burn,
            dt=dt,
            tau_max=tau_max,
        )
        axis.plot(tau_sim, _normalized(Cnn_sim), color="k", lw=1.8, label="simulation")
        axis.plot(tau_dmft, _normalized(Cnn_dmft), color="C3", lw=2.3, label="nonlinear DMFT")
        if np.all(np.isfinite(Cnn_affine)):
            axis.plot(
                tau_affine, _normalized(Cnn_affine), color="0.65", ls=":",
                lw=2.0, label="sigmoid tangent"
            )
        g0 = sigma / diagnostics["sigma_critical_tangent"]
        axis.set(
            xlabel=r"$\tau$", ylabel=r"$C_{nn}(\tau)/C_{nn}(0)$",
            title=fr"$\sigma={sigma:g},\ g_0={g0:.2f}$", xlim=(0, tau_max)
        )
        axis.legend(fontsize=8)
    fig.suptitle("Sigmoid binary network: controlled theory hierarchy", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output = os.path.join(plot_dir, "binary_theory_hierarchy.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved to {output}")
