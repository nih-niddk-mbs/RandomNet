"""Rate-network simulations and DMFT calculations."""

import os

import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.special import expit

from rn_core import autocorr, default_results_dir, make_weights, rng


def sigmoid_output(u, theta=0.0, delta=0.5):
    """Bounded rate output used by the nonlinear rate model."""
    return expit((np.asarray(u) - theta) / delta)


def _stationary_spectrum(values, smoothing_bins=1):
    """Average circular connected spectrum over independent paths."""
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values, axis=1, keepdims=True)
    transformed = np.fft.rfft(centered, axis=1)
    spectrum = np.mean(np.abs(transformed) ** 2, axis=0) / values.shape[1]
    smoothing_bins = int(max(1, smoothing_bins))
    if smoothing_bins > 1:
        kernel = np.ones(smoothing_bins, dtype=float) / smoothing_bins
        numerator = np.convolve(spectrum, kernel, mode="same")
        denominator = np.convolve(np.ones_like(spectrum), kernel, mode="same")
        spectrum = numerator / denominator
    if spectrum.size > 1:
        spectrum[0] = spectrum[1]
    return spectrum


def _stationary_covariance(values, max_lag):
    """Connected temporal covariance, averaged over sampled units."""
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values, axis=1, keepdims=True)
    n_time = values.shape[1]
    transformed = np.fft.rfft(centered, n=2 * n_time, axis=1)
    correlations = np.fft.irfft(
        np.abs(transformed) ** 2, n=2 * n_time, axis=1
    )[:, :max_lag]
    return np.mean(correlations, axis=0) / (n_time - np.arange(max_lag))


def _sample_stationary_gaussian(spectrum, normals, n_time):
    spectrum = np.maximum(np.asarray(spectrum, dtype=float), 0.0)
    coefficients = normals * np.sqrt(n_time * spectrum)[None, :]
    return np.fft.irfft(coefficients, n=n_time, axis=1)


def _nonlinear_rate_step(u, s, drive, dt, tau_u, tau_s, kappa, E, alpha, J0, mean_output):
    """Advance the nonlinear representative or network state by Euler's method."""
    du = (-u - kappa * u**3 + (E - u) * s) / tau_u
    ds = (-s + alpha * (1.0 - s) * (J0 * mean_output + drive)) / tau_s
    return u + dt * du, s + dt * ds


def sim_nonlinear_rate_network(
    N=256,
    sigma=1.3,
    tau_u=1.0,
    tau_s=1.0,
    kappa=0.2,
    E=4.0,
    alpha=0.5,
    J0=0.0,
    theta=0.0,
    delta=0.5,
    T=500.0,
    burn=150.0,
    dt=0.02,
    n_probe=128,
    tau_max=20.0,
    lam=1,
    return_paths=False,
    rng=rng,
):
    """Simulate the nonlinear conductance/saturation rate network.

    The microscopic equations are Eqs. (4.1)--(4.2) of the manuscript.  The
    returned covariances are connected temporal covariances, averaged over a
    fixed subset of units.
    """
    W = make_weights(int(N), float(sigma), lam, rng)
    u = rng.normal(0.0, 0.05, int(N))
    s = rng.normal(0.0, 0.02, int(N))
    n_burn = int(round(burn / dt))
    n_time = int(round(T / dt))
    n_probe = int(max(1, min(N, n_probe)))
    probe = np.arange(n_probe)

    def advance(u_value, s_value):
        output = sigmoid_output(u_value, theta, delta)
        centered = output - np.mean(output)
        recurrent = W @ centered
        return _nonlinear_rate_step(
            u_value,
            s_value,
            recurrent,
            dt,
            tau_u,
            tau_s,
            kappa,
            E,
            alpha,
            J0,
            float(np.mean(output)),
        )

    for _ in range(n_burn):
        u, s = advance(u, s)
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(s)):
            raise FloatingPointError("nonlinear rate network left the finite branch")

    U = np.empty((n_probe, n_time), dtype=float)
    S = np.empty_like(U)
    P = np.empty_like(U)
    for index in range(n_time):
        u, s = advance(u, s)
        U[:, index] = u[probe]
        S[:, index] = s[probe]
        P[:, index] = sigmoid_output(u[probe], theta, delta)

    max_lag = min(int(round(tau_max / dt)) + 1, n_time)
    tau = np.arange(max_lag) * dt
    covariances = {
        "Cuu": _stationary_covariance(U, max_lag),
        "Css": _stationary_covariance(S, max_lag),
        "Q": _stationary_covariance(P, max_lag),
    }
    diagnostics = {
        "mean_u": float(np.mean(U)),
        "mean_s": float(np.mean(S)),
        "mean_output": float(np.mean(P)),
        "N": int(N),
        "sigma": float(sigma),
    }
    if return_paths:
        diagnostics["paths_u"] = U
        diagnostics["paths_s"] = S
        diagnostics["paths_output"] = P
    return tau, covariances, diagnostics


def nonlinear_rate_fixed_q_response(
    output_paths,
    sigma=1.3,
    tau_u=1.0,
    tau_s=1.0,
    kappa=0.2,
    E=4.0,
    alpha=0.5,
    J0=0.0,
    theta=0.0,
    delta=0.5,
    dt=0.02,
    n_samples=256,
    warmup_cycles=2,
    tau_max=20.0,
    seed=314159,
):
    """Sample the representative process at a measured, non-iterated Q.

    ``output_paths`` are finite-network samples of Phi(u).  Holding their
    spectrum fixed isolates the Gaussian-input reduction from errors in the
    self-consistent Q iteration.
    """
    output_paths = np.asarray(output_paths, dtype=float)
    if output_paths.ndim != 2 or output_paths.shape[1] < 256:
        raise ValueError("output_paths must have shape (samples, time) with at least 256 times")
    n_time = output_paths.shape[1]
    n_samples = int(max(4, n_samples))
    q_spectrum = _stationary_spectrum(output_paths, smoothing_bins=1)
    local_rng = np.random.default_rng(seed)
    n_freq = n_time // 2 + 1
    normals = (
        local_rng.normal(size=(n_samples, n_freq))
        + 1j * local_rng.normal(size=(n_samples, n_freq))
    ) / np.sqrt(2.0)
    normals[:, 0] = local_rng.normal(size=n_samples)
    if n_time % 2 == 0:
        normals[:, -1] = local_rng.normal(size=n_samples)
    eta = _sample_stationary_gaussian(q_spectrum, normals, n_time)

    u = np.zeros(n_samples, dtype=float)
    s = np.zeros(n_samples, dtype=float)
    paths_u = np.empty((n_samples, n_time), dtype=float)
    paths_s = np.empty_like(paths_u)
    for cycle in range(int(max(1, warmup_cycles)) + 1):
        for index in range(n_time):
            u, s = _nonlinear_rate_step(
                u,
                s,
                sigma * eta[:, index],
                dt,
                tau_u,
                tau_s,
                kappa,
                E,
                alpha,
                J0,
                0.5,
            )
            if not np.all(np.isfinite(u)) or not np.all(np.isfinite(s)):
                raise FloatingPointError("fixed-Q representative paths left the finite branch")
            if cycle == int(max(1, warmup_cycles)):
                paths_u[:, index] = u
                paths_s[:, index] = s

    paths_output = sigmoid_output(paths_u, theta, delta)
    n_lag = min(int(round(tau_max / dt)) + 1, n_time)
    tau = np.arange(n_lag) * dt
    covariances = {
        "Cuu": _stationary_covariance(paths_u, n_lag),
        "Css": _stationary_covariance(paths_s, n_lag),
        "Q_in": _stationary_covariance(output_paths, n_lag),
        "Q_out": _stationary_covariance(paths_output, n_lag),
    }
    diagnostics = {
        "mean_u": float(np.mean(paths_u)),
        "mean_s": float(np.mean(paths_s)),
        "mean_output": float(np.mean(paths_output)),
        "n_samples": n_samples,
        "n_time": n_time,
    }
    return tau, covariances, diagnostics


def plot_nonlinear_rate_monte_carlo_q(
    sigma=1.6,
    N=768,
    sim_reps=4,
    representative_reps=4,
    representative_samples=1024,
    picard_reps=4,
    picard_samples=768,
    picard_n_time=8192,
    T=400.0,
    burn=200.0,
    dt=0.025,
    n_probe=384,
    tau_max=12.0,
    plot_dir=None,
):
    """Compare a network with the representative process at measured Q_N."""
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)
    network_runs = []
    output_paths = []
    for rep in range(int(max(1, sim_reps))):
        run = sim_nonlinear_rate_network(
            N=N,
            sigma=sigma,
            T=T,
            burn=burn,
            dt=dt,
            n_probe=min(N, n_probe),
            tau_max=tau_max,
            return_paths=True,
            rng=np.random.default_rng(5100 + rep),
        )
        network_runs.append(run)
        output_paths.append(run[2]["paths_output"])
    measured_outputs = np.concatenate(output_paths, axis=0)

    fixed_q_runs = [
        nonlinear_rate_fixed_q_response(
            measured_outputs,
            sigma=sigma,
            dt=dt,
            n_samples=representative_samples,
            warmup_cycles=2,
            tau_max=tau_max,
            seed=77 + 104729 * rep,
        )
        for rep in range(int(max(1, representative_reps)))
    ]
    picard_runs = [
        theory_nonlinear_rate_dmft(
            sigma=sigma,
            internal_dt=0.02,
            n_time=picard_n_time,
            n_samples=picard_samples,
            warmup_cycles=2,
            max_iter=300,
            mixing=0.15,
            tolerance=0.008,
            spectral_smoothing=1,
            tau_max=tau_max,
            seed=271828 + 104729 * rep,
            return_diagnostics=True,
        )
        for rep in range(int(max(1, picard_reps)))
    ]
    tau_network = network_runs[0][0]
    tau_fixed = fixed_q_runs[0][0]
    tau_picard = picard_runs[0][0]
    network = {
        name: np.mean([run[1][name] for run in network_runs], axis=0)
        for name in ("Cuu", "Q")
    }
    network_sem = {
        name: np.std([run[1][name] for run in network_runs], axis=0)
        / np.sqrt(len(network_runs))
        for name in ("Cuu", "Q")
    }
    fixed_q = {
        name: np.mean([run[1][name] for run in fixed_q_runs], axis=0)
        for name in ("Cuu", "Q_in", "Q_out")
    }
    fixed_q_sem = {
        name: np.std([run[1][name] for run in fixed_q_runs], axis=0)
        / np.sqrt(len(fixed_q_runs))
        for name in ("Cuu", "Q_out")
    }
    picard = {
        name: np.mean([run[1][name] for run in picard_runs], axis=0)
        for name in ("Cuu", "Q")
    }
    picard_sem = {
        name: np.std([run[1][name] for run in picard_runs], axis=0)
        / np.sqrt(len(picard_runs))
        for name in ("Cuu", "Q")
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    comparisons = (
        (axes[0, 0], network["Cuu"], fixed_q["Cuu"], picard["Cuu"], network_sem["Cuu"], fixed_q_sem["Cuu"], picard_sem["Cuu"], True, r"$C_{uu}(\tau)/C_{uu}(0)$"),
        (axes[0, 1], fixed_q["Q_in"], fixed_q["Q_out"], picard["Q"], network_sem["Q"], fixed_q_sem["Q_out"], picard_sem["Q"], True, r"$Q_\Phi(\tau)/Q_\Phi(0)$"),
        (axes[1, 0], network["Cuu"], fixed_q["Cuu"], picard["Cuu"], network_sem["Cuu"], fixed_q_sem["Cuu"], picard_sem["Cuu"], False, r"$C_{uu}(\tau)$"),
        (axes[1, 1], fixed_q["Q_in"], fixed_q["Q_out"], picard["Q"], network_sem["Q"], fixed_q_sem["Q_out"], picard_sem["Q"], False, r"$Q_\Phi(\tau)$"),
    )
    for axis, target, response, approximation, target_sem, response_sem, approximation_sem, normalize, ylabel in comparisons:
        target_scale = target[0] if normalize else 1.0
        response_scale = response[0] if normalize else 1.0
        axis.plot(tau_network, target / target_scale, color="k", lw=1.9, label=fr"network MC, $N={N}$")
        axis.fill_between(
            tau_network,
            (target - target_sem) / target_scale,
            (target + target_sem) / target_scale,
            color="k",
            alpha=0.12,
            linewidth=0,
        )
        approximation_scale = approximation[0] if normalize else 1.0
        axis.plot(
            tau_picard,
            approximation / approximation_scale,
            color="#6a3d9a",
            lw=1.9,
            ls="--",
            label="self-consistent Picard",
        )
        axis.fill_between(
            tau_picard,
            (approximation - approximation_sem) / approximation_scale,
            (approximation + approximation_sem) / approximation_scale,
            color="#6a3d9a",
            alpha=0.10,
            linewidth=0,
        )
        axis.plot(tau_fixed, response / response_scale, color="C3", lw=2.4, label=r"single site at MC $Q_N$")
        axis.fill_between(
            tau_fixed,
            (response - response_sem) / response_scale,
            (response + response_sem) / response_scale,
            color="C3",
            alpha=0.16,
            linewidth=0,
        )
        axis.axhline(0.0, color="0.82", lw=0.7)
        axis.set(xlabel=r"$\tau$", ylabel=ylabel, xlim=(0, tau_max))
        axis.legend(fontsize=8)
    fig.suptitle(fr"Nonlinear conductance-based rate network, $\sigma={sigma:g}$")
    fig.tight_layout()

    figure_path = os.path.join(plot_dir, "nonlinear_rate_network.png")
    data_path = os.path.join(plot_dir, "nonlinear_rate_network.npz")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    np.savez_compressed(
        data_path,
        tau_network=tau_network,
        tau_fixed=tau_fixed,
        tau_picard=tau_picard,
        Cuu_network=network["Cuu"],
        Q_network=network["Q"],
        Cuu_network_sem=network_sem["Cuu"],
        Q_network_sem=network_sem["Q"],
        Cuu_fixed_q=fixed_q["Cuu"],
        Q_in=fixed_q["Q_in"],
        Q_out=fixed_q["Q_out"],
        Cuu_fixed_q_sem=fixed_q_sem["Cuu"],
        Q_out_sem=fixed_q_sem["Q_out"],
        Cuu_picard=picard["Cuu"],
        Q_picard=picard["Q"],
        Cuu_picard_sem=picard_sem["Cuu"],
        Q_picard_sem=picard_sem["Q"],
        picard_residuals=np.asarray([run[2]["final_residual"] for run in picard_runs]),
        N=int(N),
        sigma=float(sigma),
    )
    print(f"Saved Monte Carlo-Q comparison to {figure_path}")
    return figure_path


def theory_nonlinear_rate_dmft(
    sigma=1.3,
    tau_u=1.0,
    tau_s=1.0,
    kappa=0.2,
    E=4.0,
    alpha=0.5,
    J0=0.0,
    theta=0.0,
    delta=0.5,
    internal_dt=0.02,
    n_time=4096,
    n_samples=64,
    warmup_cycles=2,
    max_iter=60,
    mixing=0.15,
    tolerance=0.03,
    spectral_smoothing=1,
    tau_max=20.0,
    seed=271828,
    return_diagnostics=False,
):
    """Solve the nonlinear rate DMFT by representative-path iteration of Q."""
    if min(tau_u, tau_s, delta, internal_dt) <= 0.0:
        raise ValueError("time constants, delta, and internal_dt must be positive")
    n_time = int(max(256, n_time))
    n_samples = int(max(4, n_samples))
    local_rng = np.random.default_rng(seed)
    n_freq = n_time // 2 + 1
    normals = (
        local_rng.normal(size=(n_samples, n_freq))
        + 1j * local_rng.normal(size=(n_samples, n_freq))
    ) / np.sqrt(2.0)
    normals[:, 0] = local_rng.normal(size=n_samples)
    if n_time % 2 == 0:
        normals[:, -1] = local_rng.normal(size=n_samples)

    circular_lag = np.minimum(np.arange(n_time), n_time - np.arange(n_time))
    circular_lag = circular_lag * internal_dt
    initial_variance = 0.02
    q_spectrum = np.maximum(
        np.real(np.fft.rfft(initial_variance * np.exp(-circular_lag))), 0.0
    )
    residual_history = []
    converged = False
    final_paths = None

    for iteration in range(int(max_iter)):
        eta = _sample_stationary_gaussian(q_spectrum, normals, n_time)
        u = np.zeros(n_samples, dtype=float)
        s = np.zeros(n_samples, dtype=float)
        mean_output = 0.5
        paths_u = np.empty((n_samples, n_time), dtype=float)
        paths_s = np.empty_like(paths_u)

        for cycle in range(int(max(1, warmup_cycles)) + 1):
            for index in range(n_time):
                u, s = _nonlinear_rate_step(
                    u,
                    s,
                    sigma * eta[:, index],
                    internal_dt,
                    tau_u,
                    tau_s,
                    kappa,
                    E,
                    alpha,
                    J0,
                    mean_output,
                )
                if cycle == int(max(1, warmup_cycles)):
                    paths_u[:, index] = u
                    paths_s[:, index] = s
            mean_output = float(np.mean(sigmoid_output(u, theta, delta)))

        paths_output = sigmoid_output(paths_u, theta, delta)
        proposed = _stationary_spectrum(
            paths_output, smoothing_bins=spectral_smoothing
        )
        scale = max(float(np.linalg.norm(q_spectrum)), 1e-12)
        residual = float(np.linalg.norm(proposed - q_spectrum) / scale)
        residual_history.append(residual)
        q_spectrum = (1.0 - mixing) * q_spectrum + mixing * proposed
        final_paths = (paths_u, paths_s, paths_output)
        if residual < tolerance:
            converged = True
            break

    paths_u, paths_s, paths_output = final_paths
    spectra = {
        "Cuu": _stationary_spectrum(paths_u, smoothing_bins=spectral_smoothing),
        "Css": _stationary_spectrum(paths_s, smoothing_bins=spectral_smoothing),
        "Q": q_spectrum,
    }
    circular_covariances = {
        name: np.fft.irfft(spectrum, n=n_time) for name, spectrum in spectra.items()
    }
    n_lag = min(int(round(tau_max / internal_dt)) + 1, n_time // 2)
    tau = np.arange(n_lag) * internal_dt
    covariances = {name: values[:n_lag] for name, values in circular_covariances.items()}
    diagnostics = {
        "converged": converged,
        "iterations": iteration + 1,
        "final_residual": float(residual_history[-1]),
        "residual_history": np.asarray(residual_history),
        "mean_u": float(np.mean(paths_u)),
        "mean_s": float(np.mean(paths_s)),
        "mean_output": float(np.mean(paths_output)),
        "spectral_smoothing": int(spectral_smoothing),
        "sigma_critical_linear": float(
            1.0 / (E * alpha * (0.25 / delta))
        ),
        "n_samples": n_samples,
        "n_time": n_time,
    }
    if return_diagnostics:
        return tau, covariances, diagnostics
    return tau, covariances


def _normalized(covariance):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.size == 0 or abs(covariance[0]) < 1e-14:
        return covariance
    return covariance / covariance[0]


def plot_nonlinear_rate_network(
    sigma=1.3,
    N_vals=(192, 384, 768),
    sim_reps=2,
    T=500.0,
    burn=200.0,
    dt=0.025,
    tau_max=12.0,
    theory_reps=1,
    theory_kwargs=None,
    plot_dir=None,
):
    """Compare nonlinear rate DMFT with finite networks and save the data."""
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)
    theory_kwargs = {} if theory_kwargs is None else dict(theory_kwargs)
    theory_runs = []
    base_seed = int(theory_kwargs.pop("seed", 271828))
    for rep in range(int(max(1, theory_reps))):
        theory_runs.append(
            theory_nonlinear_rate_dmft(
                sigma=sigma,
                tau_max=tau_max,
                seed=base_seed + 104729 * rep,
                return_diagnostics=True,
                **theory_kwargs,
            )
        )
    tau_theory = theory_runs[0][0]
    theory = {
        name: np.mean([run[1][name] for run in theory_runs], axis=0)
        for name in ("Cuu", "Css", "Q")
    }
    theory_std = {
        name: np.std([run[1][name] for run in theory_runs], axis=0)
        for name in ("Cuu", "Css", "Q")
    }
    theory_diagnostics = [run[2] for run in theory_runs]

    simulations = []
    for size_index, N in enumerate(N_vals):
        runs = []
        for rep in range(int(sim_reps)):
            runs.append(
                sim_nonlinear_rate_network(
                    N=N,
                    sigma=sigma,
                    T=T,
                    burn=burn,
                    dt=dt,
                    n_probe=min(N, 384),
                    tau_max=tau_max,
                    rng=np.random.default_rng(4100 + 101 * size_index + rep),
                )
            )
        simulations.append(
            {
                "N": int(N),
                "tau": runs[0][0],
                "Cuu": np.mean([run[1]["Cuu"] for run in runs], axis=0),
                "Q": np.mean([run[1]["Q"] for run in runs], axis=0),
                "Cuu_std": np.std([run[1]["Cuu"] for run in runs], axis=0),
                "Q_std": np.std([run[1]["Q"] for run in runs], axis=0),
                "Cuu0_runs": np.asarray([run[1]["Cuu"][0] for run in runs]),
                "Q0_runs": np.asarray([run[1]["Q"][0] for run in runs]),
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    result = simulations[-1]
    label = fr"network $N={result['N']}$"
    normalized_cuu = _normalized(result["Cuu"])
    normalized_q = _normalized(result["Q"])
    sim_scale = np.sqrt(max(1, int(sim_reps)))
    axes[0, 0].plot(result["tau"], normalized_cuu, color="k", lw=1.8, label=label)
    axes[0, 1].plot(result["tau"], normalized_q, color="k", lw=1.8, label=label)
    axes[1, 0].plot(result["tau"], result["Cuu"], color="k", lw=1.8, label=label)
    axes[0, 0].fill_between(
        result["tau"],
        normalized_cuu - result["Cuu_std"] / (result["Cuu"][0] * sim_scale),
        normalized_cuu + result["Cuu_std"] / (result["Cuu"][0] * sim_scale),
        color="k",
        alpha=0.12,
        linewidth=0,
    )
    axes[0, 1].fill_between(
        result["tau"],
        normalized_q - result["Q_std"] / (result["Q"][0] * sim_scale),
        normalized_q + result["Q_std"] / (result["Q"][0] * sim_scale),
        color="k",
        alpha=0.12,
        linewidth=0,
    )

    axes[0, 0].plot(tau_theory, _normalized(theory["Cuu"]), color="C3", lw=2.6, label="DMFT")
    axes[0, 1].plot(tau_theory, _normalized(theory["Q"]), color="C3", lw=2.6, label="DMFT")
    axes[1, 0].plot(tau_theory, theory["Cuu"], color="C3", lw=2.6, label="DMFT")
    theory_scale = np.sqrt(max(1, int(theory_reps)))
    axes[0, 0].fill_between(
        tau_theory,
        _normalized(theory["Cuu"]) - theory_std["Cuu"] / (theory["Cuu"][0] * theory_scale),
        _normalized(theory["Cuu"]) + theory_std["Cuu"] / (theory["Cuu"][0] * theory_scale),
        color="C3",
        alpha=0.16,
        linewidth=0,
    )
    axes[0, 1].fill_between(
        tau_theory,
        _normalized(theory["Q"]) - theory_std["Q"] / (theory["Q"][0] * theory_scale),
        _normalized(theory["Q"]) + theory_std["Q"] / (theory["Q"][0] * theory_scale),
        color="C3",
        alpha=0.16,
        linewidth=0,
    )
    axes[0, 0].set(xlabel=r"$\tau$", ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$", xlim=(0, tau_max))
    axes[0, 1].set(xlabel=r"$\tau$", ylabel=r"$Q_\Phi(\tau)/Q_\Phi(0)$", xlim=(0, tau_max))
    axes[1, 0].set(xlabel=r"$\tau$", ylabel=r"$C_{uu}(\tau)$", xlim=(0, tau_max))
    for axis in axes.flat[:3]:
        axis.axhline(0.0, color="0.8", lw=0.7)
        axis.legend(fontsize=8)

    inverse_sizes = 1.0 / np.asarray([result["N"] for result in simulations])
    cuu_means = np.asarray([np.mean(result["Cuu0_runs"]) for result in simulations])
    cuu_errors = np.asarray([np.std(result["Cuu0_runs"]) for result in simulations])
    q_means = np.asarray([np.mean(result["Q0_runs"]) for result in simulations])
    q_errors = np.asarray([np.std(result["Q0_runs"]) for result in simulations])
    axes[1, 1].errorbar(inverse_sizes, cuu_means, yerr=cuu_errors / sim_scale, fmt="o-", color="k", label=r"network $C_{uu}(0)$")
    axes[1, 1].errorbar(inverse_sizes, q_means, yerr=q_errors / sim_scale, fmt="s-", color="C0", label=r"network $Q_\Phi(0)$")
    axes[1, 1].axhline(theory["Cuu"][0], color="C3", lw=2.2, label=r"DMFT $C_{uu}(0)$")
    axes[1, 1].axhline(theory["Q"][0], color="C1", lw=2.2, ls="--", label=r"DMFT $Q_\Phi(0)$")
    axes[1, 1].set(xlabel=r"$1/N$", ylabel="equal-time covariance")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(fr"Nonlinear conductance-based rate network, $\sigma={sigma:g}$")
    fig.tight_layout()

    figure_path = os.path.join(plot_dir, "nonlinear_rate_network.png")
    data_path = os.path.join(plot_dir, "nonlinear_rate_network.npz")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    arrays = {
        "tau_theory": tau_theory,
        "Cuu_theory": theory["Cuu"],
        "Css_theory": theory["Css"],
        "Q_theory": theory["Q"],
        "N_vals": np.asarray(N_vals),
        "Cuu_theory_std": theory_std["Cuu"],
        "Q_theory_std": theory_std["Q"],
        "theory_final_residuals": np.asarray(
            [item["final_residual"] for item in theory_diagnostics]
        ),
    }
    for result in simulations:
        suffix = str(result["N"])
        arrays[f"tau_N{suffix}"] = result["tau"]
        arrays[f"Cuu_N{suffix}"] = result["Cuu"]
        arrays[f"Q_N{suffix}"] = result["Q"]
        arrays[f"Cuu_std_N{suffix}"] = result["Cuu_std"]
        arrays[f"Q_std_N{suffix}"] = result["Q_std"]
        arrays[f"Cuu0_runs_N{suffix}"] = result["Cuu0_runs"]
        arrays[f"Q0_runs_N{suffix}"] = result["Q0_runs"]
    np.savez_compressed(data_path, **arrays)
    print(
        f"Saved to {figure_path}; max DMFT residual="
        f"{max(item['final_residual'] for item in theory_diagnostics):.4g}, "
        f"Cuu(0)={theory['Cuu'][0]:.4g}, Q(0)={theory['Q'][0]:.4g}"
    )
    return figure_path

# -----------------------------------------------------------------------------
# 1. RATE NEURON NETWORK
#    Model: du_i/dt = -u_i + sum_j W_ij f(u_j)
#    Theory: C''(tau) = C(tau) - sigma^2 Q(tau),
#            Q(tau) = <f(u(0))f(u(tau))>
# -----------------------------------------------------------------------------

def sim_rate_network(
    N=512,
    sigma=1.5,
    T=2000.0,
    dt=0.05,
    f=np.tanh,
    lam=1,
    burn=200,
    n_probe=64,
    tau_max=50.0,
    rng=rng,
):
    """
    Euler-Maruyama integration of the rate network.
    Returns time-lagged autocorrelation of u averaged over neurons.

    Parameters
    ----------
    N     : network size
    sigma : weight std  (chaos when sigma > 1 for tanh)
    T     : total simulation time
    dt    : time step
    f     : gain function
    lam   : 1=row-sum corrected, 0=plain
    burn  : burn-in time (discarded)
    """
    W = make_weights(N, sigma, lam, rng)
    nt = int(T / dt)
    nb = int(burn / dt)
    u = rng.normal(0, 0.1, N)

    # burn in
    for _ in range(nb):
        u += dt * (-u + W @ f(u))

    n_probe = int(max(1, min(N, n_probe)))
    probe_idx = np.arange(n_probe)
    U = np.zeros((nt, n_probe))
    for t in range(nt):
        u += dt * (-u + W @ f(u))
        U[t] = u[probe_idx]

    # Average autocorrelation over a subset of neurons.
    # Using more probes reduces large-lag variance in finite simulations.
    max_lag = int(tau_max / dt)
    C = np.mean([autocorr(U[:, i], max_lag) for i in range(n_probe)], axis=0)
    tau = np.arange(len(C)) * dt
    return tau, C


def theory_rate_autocorr(
    C0=None,
    sigma=1.5,
    tau_max=50,
    dtau=0.01,
    f=np.tanh,
    n_quad=20,
    C0_bounds=(0.05, 5.0),
    mu_f=None,
):
    """
         Solve the SCS equation:
             C''(tau) = C(tau) - sigma^2 Q(tau)
       Q(tau) = <f(u(0))f(u(tau))>_Gaussian
    with u ~ N(0, C(0)).

     We use deterministic Gauss-Hermite quadrature to evaluate
         Q(tau) = E[f(x) f(y)] with (x,y) jointly Gaussian.
     This keeps the theory curve reproducible and removes RNG dependence.

    The physical solution is the monotone branch that satisfies the energy
    condition V(C(0)) = V(0) = 0, where V'(C) = -C + sigma^2 Q(C).
    If C0 is None, we solve for that self-consistent C0 by bisection.
    
    If mu_f is provided (a scalar or callable), Q is computed as centered covariance:
        Q_centered(tau) = E[f(x)f(y)] - mu_f(C0)^2
    This is needed for activation functions with nonzero mean (e.g., ReLU-like gains).
    """
    ntau = int(tau_max / dtau)
    tau = np.arange(ntau) * dtau
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    gh_w2 = np.outer(gh_w, gh_w)

    def Q_func(C_tau, C0_val):
        """Deterministic quadrature for E[f(x)f(y)] with correlated Gaussian x,y,
        optionally centered by subtracting mu_f^2."""
        if C0_val <= 0:
            return 0.0
        rho = float(np.clip(C_tau / C0_val, -0.999999, 0.999999))
        scale = np.sqrt(2.0 * C0_val)
        x = scale * gh_x[:, None]
        y = scale * (rho * gh_x[:, None] + np.sqrt(1.0 - rho**2) * gh_x[None, :])
        vals = f(x) * f(y)
        Q_raw = float(np.sum(gh_w2 * vals) / np.pi)
        
        # Subtract mu^2 if centering is requested (for nonzero-mean activation functions)
        if mu_f is not None:
            mu = mu_f(C0_val) if callable(mu_f) else float(mu_f)
            Q_centered = Q_raw - mu**2
            return Q_centered
        return Q_raw

    def energy_endpoint(C0_val, n_grid=256):
        """Return H(C0) = C0^2 - 2∫_0^{C0} Q(C; C0) dC."""
        if C0_val <= 0:
            return np.nan
        C_grid = np.linspace(0.0, float(C0_val), n_grid)
        Q_grid = np.array([sigma**2 * Q_func(c, C0_val) for c in C_grid])
        integral_Q = np.zeros_like(C_grid)
        integral_Q[1:] = np.cumsum(0.5 * (Q_grid[1:] + Q_grid[:-1]) * np.diff(C_grid))
        return float(C0_val**2 - 2.0 * integral_Q[-1])

    def solve_c0(initial_guess=None):
        lo, hi = C0_bounds
        candidates = np.linspace(lo, hi, 24)
        if initial_guess is not None:
            candidates = np.unique(np.sort(np.append(candidates, float(initial_guess))))
        values = np.array([energy_endpoint(c0) for c0 in candidates])

        finite = np.isfinite(values)
        candidates = candidates[finite]
        values = values[finite]
        if len(candidates) == 0:
            return float(initial_guess if initial_guess is not None else 0.65)

        # Look for a NEGATIVE-to-POSITIVE sign change only.
        # +→- crossings are numerical artifacts near C0=0 and must be ignored;
        # the physical SCS fixed point is where H goes from negative to positive.
        for i in range(len(candidates) - 1):
            if values[i] == 0 and values[i] < values[i + 1]:
                return float(candidates[i])
            if values[i] < 0 and values[i + 1] > 0:          # physical -→+ crossing
                a, b = float(candidates[i]), float(candidates[i + 1])
                fa, fb = float(values[i]), float(values[i + 1])
                for _ in range(40):
                    m = 0.5 * (a + b)
                    fm = energy_endpoint(m)
                    if not np.isfinite(fm):
                        break
                    if abs(fm) < 1e-8:
                        return float(m)
                    if fa * fm <= 0:
                        b, fb = m, fm
                    else:
                        a, fa = m, fm
                return float(0.5 * (a + b))

        # No physical -→+ crossing found: no SCS fixed point exists.
        return float(lo)

    def monotone_solution(C0_val):
        C0_val = float(C0_val)
        n_grid = max(512, int(300 * max(C0_val, 1.0)))
        C_grid = np.linspace(0.0, C0_val, n_grid)
        Q_grid = np.array([sigma**2 * Q_func(c, C0_val) for c in C_grid])
        integral_Q = np.zeros_like(C_grid)
        integral_Q[1:] = np.cumsum(0.5 * (Q_grid[1:] + Q_grid[:-1]) * np.diff(C_grid))
        H_grid = np.maximum(C_grid**2 - 2.0 * integral_Q, 1e-14)

        # Since C decays monotonically, integrate d tau / dC = 1 / sqrt(H(C)).
        C_desc = C_grid[::-1]
        H_desc = H_grid[::-1]
        speed = np.sqrt(H_desc)
        dC = -np.diff(C_desc)
        seg_speed = 0.5 * (speed[:-1] + speed[1:])
        tau_desc = np.concatenate([[0.0], np.cumsum(dC / np.maximum(seg_speed, 1e-14))])
        tau_out = np.arange(ntau) * dtau
        C_out = np.interp(tau_out, tau_desc, C_desc, left=C0_val, right=0.0)
        return C_out

    if C0 is None:
        C0 = solve_c0(initial_guess=None)
    else:
        # Treat the supplied value as a guess, not as a hard constraint.
        C0 = solve_c0(initial_guess=C0)

    C = monotone_solution(C0)
    return tau, C


def plot_rate_network(
    sigma=2.2,
    N=1536,
    C0_guess=0.8,
    T=1800.0,
    dt=0.05,
    burn=400.0,
    n_probe=768,
    sim_reps=5,
    seeds=(301, 302, 303, 304, 305),
    tau_max=50.0,
    n_quad=48,
    plot_dir=None,
):
    """Compare simulation vs theory for rate network."""
    import os
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Simulating rate network: N={N}, sigma={sigma} ...")
    runs = []
    tau_sim = None
    for rep in range(int(sim_reps)):
        seed = int(seeds[rep]) if rep < len(seeds) else int(seeds[-1]) + rep
        tau_sim, covariance = sim_rate_network(
            N=N,
            sigma=sigma,
            T=T,
            dt=dt,
            burn=burn,
            n_probe=n_probe,
            tau_max=tau_max,
            rng=np.random.default_rng(seed),
        )
        runs.append(covariance / covariance[0])
    C_sim = np.mean(runs, axis=0)

    print("Computing theory ...")
    tau_th, C_th = theory_rate_autocorr(
        C0=C0_guess,
        sigma=sigma,
        tau_max=tau_max,
        n_quad=n_quad,
    )
    C_th_norm = C_th / C_th[0]
    comparison = np.interp(tau_sim, tau_th, C_th_norm)
    mask = tau_sim <= min(20.0, tau_max)
    rmse = float(np.sqrt(np.mean((C_sim[mask] - comparison[mask]) ** 2)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(tau_sim, C_sim, "b", lw=1.5, label="Simulation")
    ax.plot(tau_th, C_th_norm, "r--", lw=2, label="SCS theory")
    ax.set(
        xlabel=r"$\tau$",
        ylabel=r"$C_{11}(\tau)/C_{11}(0)$",
        title=fr"Rate network $\sigma={sigma:g}$ (RMSE={rmse:.4f})",
        xlim=(0, 20),
    )
    ax.legend()

    # power spectrum
    ax = axes[1]
    n = len(C_sim)
    fq = fftfreq(2 * n, d=tau_sim[1] - tau_sim[0])[:n]
    Sw = np.abs(fft(np.concatenate([C_sim, C_sim[::-1]]))[:n])
    ax.semilogy(fq, Sw, "b", lw=1.5, label="Simulation")
    ax.set(
        xlabel=r"$\omega / 2\pi$",
        ylabel="Power spectrum",
        title="Power spectrum",
        xlim=(0, 2),
    )
    ax.legend()

    plt.suptitle(
        f"Rate-network SCS calibration ({sim_reps} finite-network runs)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "rate_network_test.png"), dpi=150)
    print(f"Saved to {os.path.join(plot_dir, 'rate_network_test.png')}")
    plt.close("all")
