"""Chaos diagnostics and figures for general random spiking networks."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rn_phase import (
    _sim_phase_timeseries,
    maximal_lyapunov_phase_network,
    phase_replica_stability_dmft,
    sim_phase_network,
    theory_phase_density_autocorr,
    theory_phase_twotime_dmft,
)


def _lyapunov_job(arguments):
    N, sigma, repetition, kwargs = arguments
    return maximal_lyapunov_phase_network(
        N=N,
        sigma=sigma,
        seed=int(kwargs["seed"]) + 1009 * repetition + 17 * N,
        **{key: value for key, value in kwargs.items() if key != "seed"},
    )


def scan_network_lyapunov(
    sigma_values,
    N_values=(64, 128, 256),
    repetitions=3,
    jobs=1,
    **kwargs,
):
    """Scan finite-network Lyapunov exponents with reproducible disorder."""
    sigma_values = np.asarray(sigma_values, dtype=float)
    N_values = np.asarray(N_values, dtype=int)
    parameters = dict(
        I=1.0,
        alpha=1.0,
        beta=1.0,
        T=300.0,
        dt=0.01,
        burn=150.0,
        renormalization_time=0.25,
        perturbation=1e-6,
        lam=1,
        phase_model="theta",
        seed=20260822,
    )
    parameters.update(kwargs)
    tasks = [
        (int(N), float(sigma), repetition, parameters)
        for N in N_values
        for sigma in sigma_values
        for repetition in range(int(repetitions))
    ]
    if int(jobs) > 1:
        with ProcessPoolExecutor(max_workers=int(jobs)) as executor:
            flat = list(executor.map(_lyapunov_job, tasks))
    else:
        flat = [_lyapunov_job(task) for task in tasks]
    samples = np.asarray(flat).reshape(
        len(N_values), len(sigma_values), int(repetitions)
    )
    return dict(
        sigma=sigma_values,
        N=N_values,
        samples=samples,
        mean=np.mean(samples, axis=2),
        standard_error=np.std(samples, axis=2, ddof=1)
        / np.sqrt(max(int(repetitions), 1))
        if int(repetitions) > 1
        else np.zeros(samples.shape[:2]),
        parameters=parameters,
    )


def _replica_job(arguments):
    sigma, n_time, kwargs = arguments
    multiplier, diagnostics = phase_replica_stability_dmft(
        sigma=sigma,
        n_time=n_time,
        seed=int(kwargs["seed"]) + int(n_time) + int(round(100 * sigma)),
        return_diagnostics=True,
        **{key: value for key, value in kwargs.items() if key != "seed"},
    )
    return multiplier, diagnostics["converged"]


def scan_replica_stability(
    sigma_values,
    n_time_values=(1024, 2048, 4096),
    jobs=1,
    **kwargs,
):
    """Scan the tangent two-replica DMFT multiplier over observation windows."""
    sigma_values = np.asarray(sigma_values, dtype=float)
    n_time_values = np.asarray(n_time_values, dtype=int)
    parameters = dict(
        I=1.0,
        alpha=1.0,
        beta=1.0,
        internal_dt=0.02,
        n_samples=96,
        fixed_point_iterations=50,
        fixed_point_mixing=0.18,
        fixed_point_tolerance=0.03,
        power_iterations=8,
        covariance_perturbation=1e-3,
        phase_model="theta",
        seed=8675309,
    )
    parameters.update(kwargs)
    tasks = [
        (float(sigma), int(n_time), parameters)
        for n_time in n_time_values
        for sigma in sigma_values
    ]
    if int(jobs) > 1:
        with ProcessPoolExecutor(max_workers=int(jobs)) as executor:
            flat = list(executor.map(_replica_job, tasks))
    else:
        flat = [_replica_job(task) for task in tasks]
    multipliers = np.asarray([item[0] for item in flat]).reshape(
        len(n_time_values), len(sigma_values)
    )
    converged = np.asarray([item[1] for item in flat]).reshape(
        len(n_time_values), len(sigma_values)
    )
    critical_sigma = np.full(len(n_time_values), np.nan)
    for row, values in enumerate(multipliers):
        log_values = np.log(np.maximum(values, 1e-300))
        crossings = np.flatnonzero(log_values >= 0.0)
        if len(crossings) and crossings[0] > 0:
            upper = int(crossings[0])
            lower = upper - 1
            fraction = -log_values[lower] / (
                log_values[upper] - log_values[lower]
            )
            critical_sigma[row] = sigma_values[lower] + fraction * (
                sigma_values[upper] - sigma_values[lower]
            )
    return dict(
        sigma=sigma_values,
        n_time=n_time_values,
        duration=n_time_values * float(parameters["internal_dt"]),
        multiplier=multipliers,
        critical_sigma=critical_sigma,
        converged=converged,
        parameters=parameters,
    )


def plot_lyapunov_scan(result, output_path):
    """Plot finite-size Lyapunov estimates with uncertainty across disorder."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(result["N"])))
    for N, mean, error, color in zip(
        result["N"], result["mean"], result["standard_error"], colors
    ):
        ax.errorbar(
            result["sigma"],
            mean,
            yerr=error,
            marker="o",
            ms=4,
            lw=1.7,
            capsize=2,
            color=color,
            label=fr"$N={N}$",
        )
    ax.axhline(0.0, color="0.15", lw=1.0)
    ax.set(
        xlabel=r"coupling disorder $\sigma$",
        ylabel=r"maximal Lyapunov exponent $\lambda_{\max}$",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_replica_scan(result, output_path):
    """Plot the two-replica tangent covariance multiplier."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = plt.cm.plasma(np.linspace(0.15, 0.82, len(result["duration"])))
    for duration, values, color in zip(
        result["duration"], result["multiplier"], colors
    ):
        ax.plot(
            result["sigma"],
            values,
            "o-",
            lw=1.7,
            ms=4,
            color=color,
            label=fr"window $T={duration:g}$",
        )
    ax.axhline(1.0, color="0.15", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set(
        xlabel=r"coupling disorder $\sigma$",
        ylabel="leading replica multiplier",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_spiking_activity(
    output_path,
    sigma_values=(0.25, 2.0),
    N=192,
    T=80.0,
    burn=100.0,
    dt=0.01,
    I=1.0,
    phase_model="theta",
):
    """Plot spike rasters and recurrent fields for a built-in neuron model."""
    fig, axes = plt.subplots(2, len(sigma_values), figsize=(10.5, 5.2), sharex=True)
    if len(sigma_values) == 1:
        axes = axes.reshape(2, 1)
    for column, sigma in enumerate(sigma_values):
        time, fields, spike_times = _sim_phase_timeseries(
            N=N,
            I=I,
            sigma=sigma,
            beta=1.0,
            T=T,
            burn=burn,
            dt=dt,
            n_show=N,
            phase_model=phase_model,
            rng_=np.random.default_rng(500 + column),
        )
        for neuron, events in enumerate(spike_times):
            axes[0, column].vlines(events, neuron - 0.45, neuron + 0.45, lw=0.35)
        for neuron in range(min(6, N)):
            axes[1, column].plot(time, fields[:, neuron], lw=0.7, alpha=0.8)
        axes[0, column].set_title(fr"$\sigma={sigma:g}$")
        axes[0, column].set_ylim(-1, N)
        axes[1, column].set_xlabel("time")
    axes[0, 0].set_ylabel("neuron")
    axes[1, 0].set_ylabel(r"$u_i(t)$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_theta_activity(*args, **kwargs):
    """Backward-compatible transformed-theta activity wrapper."""
    kwargs.setdefault("I", 1.0)
    kwargs.setdefault("phase_model", "theta")
    return plot_spiking_activity(*args, **kwargs)


def compute_spiking_covariance_comparison(
    sigma,
    N=256,
    T=600.0,
    burn=200.0,
    dt=0.02,
    tau_max=12.0,
    n_probe=192,
    seed=314159,
    stationary_kwargs=None,
    twotime_kwargs=None,
    include_twotime=True,
    I=1.0,
    phase_model="theta",
    phase_bin_width=0.25,
):
    """Compute C11 and threshold-density C33 from simulation and DMFT."""
    tau_sim, C11_sim, C33_sim = sim_phase_network(
        N=N,
        I=I,
        sigma=sigma,
        beta=1.0,
        T=T,
        burn=burn,
        dt=dt,
        tau_max=tau_max,
        n_probe=n_probe,
        return_phase_density=True,
        phase_bin_width=phase_bin_width,
        phase_model=phase_model,
        rng=np.random.default_rng(seed),
    )
    stationary = dict(
        internal_dt=dt,
        n_time=8192,
        n_samples=96,
        max_iter=50,
        mixing=0.18,
        tolerance=0.03,
    )
    if stationary_kwargs:
        stationary.update(stationary_kwargs)
    tau_stationary, C11_stationary, _, diagnostic_stationary = (
        theory_phase_density_autocorr(
            I=I,
            sigma=sigma,
            beta=1.0,
            tau_max=tau_max,
            dtau=dt,
            seed=seed + 1,
            return_diagnostics=True,
            phase_bin_width=phase_bin_width,
            phase_model=phase_model,
            **stationary,
        )
    )
    result = dict(
        sigma=float(sigma),
        I=float(I),
        phase_model=str(phase_model),
        tau_sim=tau_sim,
        C11_sim=C11_sim,
        C33_sim=C33_sim,
        tau_stationary=tau_stationary,
        C11_stationary=C11_stationary,
        C33_stationary=diagnostic_stationary["phase_density_covariance"],
        tau_C33_stationary=(
            np.arange(len(diagnostic_stationary["phase_density_covariance"]))
            * diagnostic_stationary["internal_dt"]
        ),
        stationary_diagnostics=diagnostic_stationary,
    )
    if include_twotime:
        twotime = dict(
            internal_dt=0.05,
            n_time=640,
            n_samples=768,
            max_iter=80,
            mixing=0.06,
            tolerance=0.05,
            transient_fraction=0.5,
        )
        if twotime_kwargs:
            twotime.update(twotime_kwargs)
        tau_twotime, C11_twotime, _, diagnostic_twotime = (
            theory_phase_twotime_dmft(
                I=I,
                sigma=sigma,
                beta=1.0,
                tau_max=tau_max,
                dtau=dt,
                seed=seed + 2,
                return_diagnostics=True,
                phase_bin_width=phase_bin_width,
                phase_model=phase_model,
                **twotime,
            )
        )
        result.update(
            tau_twotime=tau_twotime,
            C11_twotime=C11_twotime,
            C33_twotime=diagnostic_twotime["phase_density_covariance"],
            tau_C33_twotime=(
                np.arange(len(diagnostic_twotime["phase_density_covariance"]))
                * diagnostic_twotime["internal_dt"]
            ),
            twotime_diagnostics=diagnostic_twotime,
        )
    return result


def compute_theta_covariance_comparison(*args, **kwargs):
    """Backward-compatible transformed-theta covariance wrapper."""
    kwargs.setdefault("I", 1.0)
    kwargs.setdefault("phase_model", "theta")
    return compute_spiking_covariance_comparison(*args, **kwargs)


def plot_spiking_covariance_comparison(results, output_path):
    """Plot C11 and C33 with simulation visually dominant."""
    fig, axes = plt.subplots(2, len(results), figsize=(5.2 * len(results), 7.0))
    if len(results) == 1:
        axes = axes.reshape(2, 1)
    for column, result in enumerate(results):
        axes[0, column].plot(
            result["tau_sim"], result["C11_sim"], color="k", lw=2.4, label="simulation"
        )
        axes[0, column].plot(
            result["tau_stationary"],
            result["C11_stationary"],
            color="C0",
            lw=2.0,
            ls="--",
            label="event-DMFT",
        )
        if "tau_twotime" in result:
            axes[0, column].plot(
                result["tau_twotime"],
                result["C11_twotime"],
                color="C3",
                lw=1.5,
                label="two-time event-DMFT",
            )
        axes[1, column].plot(
            result["tau_sim"], result["C33_sim"], color="k", lw=2.4
        )
        axes[1, column].plot(
            result["tau_C33_stationary"],
            result["C33_stationary"],
            color="C0",
            lw=2.0,
            ls="--",
        )
        if "tau_C33_twotime" in result:
            axes[1, column].plot(
                result["tau_C33_twotime"],
                result["C33_twotime"],
                color="C3",
                lw=1.5,
            )
        axes[0, column].set_title(fr"$\sigma={result['sigma']:g}$")
        axes[1, column].set_xlabel(r"lag $\tau$")
        axes[1, column].set_xlim(0.0, result["tau_sim"][-1])
        axes[0, column].grid(alpha=0.18)
        axes[1, column].grid(alpha=0.18)
    axes[0, 0].set_ylabel(r"$C_{11}(\tau)$")
    axes[1, 0].set_ylabel(r"$C_{33}(\tau)$")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_theta_covariance_comparison(results, output_path):
    """Backward-compatible covariance plotting wrapper."""
    return plot_spiking_covariance_comparison(results, output_path)


def plot_lif_summary(lyapunov, replica, covariance, output_path):
    """Summarize LIF network chaos and cavity-DMFT covariance predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lyapunov["N"])))
    for N, mean, error, color in zip(
        lyapunov["N"],
        lyapunov["mean"],
        lyapunov["standard_error"],
        colors,
    ):
        axes[0, 0].errorbar(
            lyapunov["sigma"],
            mean,
            yerr=error,
            color=color,
            marker="o",
            ms=3.5,
            lw=1.5,
            capsize=2,
            label=fr"$N={N}$",
        )
    axes[0, 0].axhline(0.0, color="0.15", lw=0.9)
    axes[0, 0].set(
        xlabel=r"coupling disorder $\sigma$",
        ylabel=r"$\lambda_{\max}$",
        title="(a) Finite-network stability",
    )
    axes[0, 0].legend(frameon=False)

    replica_colors = plt.cm.plasma(
        np.linspace(0.15, 0.82, len(replica["duration"]))
    )
    for duration, multiplier, color in zip(
        replica["duration"], replica["multiplier"], replica_colors
    ):
        axes[0, 1].plot(
            replica["sigma"],
            multiplier,
            "o-",
            color=color,
            ms=3.5,
            lw=1.5,
            label=fr"$T={duration:g}$",
        )
    axes[0, 1].axhline(1.0, color="0.15", lw=0.9, ls="--")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(
        xlabel=r"coupling disorder $\sigma$",
        ylabel="replica multiplier",
        title="(b) Cavity stability",
    )
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(
        covariance["tau_sim"],
        covariance["C11_sim"],
        color="k",
        lw=2.3,
        label="simulation",
    )
    axes[1, 0].plot(
        covariance["tau_stationary"],
        covariance["C11_stationary"],
        color="C0",
        lw=1.9,
        ls="--",
        label="event-DMFT",
    )
    axes[1, 0].set(
        xlabel=r"lag $\tau$",
        ylabel=r"$C_{11}(\tau)$",
        title=fr"(c) Field covariance, $\sigma={covariance['sigma']:g}$",
    )
    axes[1, 0].set_xlim(0.0, covariance["tau_sim"][-1])
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(
        covariance["tau_sim"],
        covariance["C33_sim"],
        color="k",
        lw=2.3,
        label="simulation",
    )
    axes[1, 1].plot(
        covariance["tau_C33_stationary"],
        covariance["C33_stationary"],
        color="C0",
        lw=1.9,
        ls="--",
        label="event-DMFT",
    )
    axes[1, 1].set(
        xlabel=r"lag $\tau$",
        ylabel=r"$C_{33}(\tau)$",
        title="(d) Threshold-density covariance",
    )
    axes[1, 1].set_xlim(0.0, covariance["tau_sim"][-1])
    for ax in axes.flat:
        ax.grid(alpha=0.18, which="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_scan(path, result):
    """Save array-valued scan entries in a portable NumPy archive."""
    arrays = {
        key: value
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
