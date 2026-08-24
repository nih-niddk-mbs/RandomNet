"""Benchmark LIF Fokker--Planck quadrature against Gaussian drive paths."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


blas_threads = os.environ.get("RANDOMNET_BLAS_THREADS", "1")
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = blas_threads
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rn_lif_fp import (
    compute_lif_ou_return_map,
    solve_lif_ou_first_return_renewal,
    solve_projected_lif_ou_dmft,
    validate_lif_ou_fokker_planck,
)


def run_lif_fp_benchmark(profile="quick", output_dir=None):
    """Run prescribed-drive validation and the one-mode closure diagnostic."""
    quick = profile == "quick"
    output = Path(
        output_dir or os.environ.get("RANDOMNET_RESULTS_DIR", "~/randomnet-results")
    ).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    n_voltage = 96 if quick else 300
    n_drive = 61 if quick else 161
    tau_max = 6.0 if quick else 8.0
    dt = 0.04 if quick else 0.02
    validation = validate_lif_ou_fokker_planck(
        n_time=2048 if quick else 8192,
        n_paths=128 if quick else 768,
        n_voltage=n_voltage,
        n_drive=n_drive,
        tau_max=tau_max,
        dt=dt,
    )
    renewal = solve_lif_ou_first_return_renewal(
        n_voltage=n_voltage,
        n_drive=n_drive,
        tau_max=tau_max,
        dt=dt,
    )
    memory_decays = np.asarray((0.5, 1.0, 2.0, 5.0, 10.0))
    memory_field_errors = []
    memory_spike_errors = []
    memory_return_radii = []
    memory_scalar_estimates = []
    for decay in memory_decays:
        memory_result = solve_lif_ou_first_return_renewal(
            drive_decay=decay,
            n_voltage=96 if quick else 128,
            n_drive=61 if quick else 81,
            tau_max=6.0,
            dt=0.04 if quick else 0.03,
        )
        memory_full = memory_result.full_fokker_planck
        memory_return_map = compute_lif_ou_return_map(memory_full)
        memory_field_errors.append(
            np.linalg.norm(
                memory_result.field_covariance - memory_full.field_covariance
            )
            / max(np.linalg.norm(memory_full.field_covariance), 1e-15)
        )
        memory_spike_errors.append(
            np.linalg.norm(
                memory_result.regular_spike_covariance
                - memory_full.regular_spike_covariance
            )
            / max(
                np.linalg.norm(memory_full.regular_spike_covariance),
                1e-15,
            )
        )
        memory_return_radii.append(memory_return_map.spectral_radius)
        memory_scalar_estimates.append(
            np.trapezoid(
                np.exp(-decay * memory_result.tau)
                * memory_result.first_return_density,
                memory_result.tau,
            )
        )
    memory_field_errors = np.asarray(memory_field_errors)
    memory_spike_errors = np.asarray(memory_spike_errors)
    memory_return_radii = np.asarray(memory_return_radii)
    memory_scalar_estimates = np.asarray(memory_scalar_estimates)
    return_map = compute_lif_ou_return_map(renewal.full_fokker_planck)
    projection = solve_projected_lif_ou_dmft(
        sigma=0.5,
        I=2.0,
        beta=1.0,
        tau_max=6.0,
        dt=0.05 if quick else 0.04,
        max_iter=12 if quick else 20,
        mixing=0.4,
        tolerance=5e-3 if quick else 2e-3,
        n_voltage=80 if quick else 100,
        n_drive=51 if quick else 61,
        fit_tau_max=4.0,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.1))
    axes[0].plot(
        validation.tau,
        validation.path_covariance,
        color="k",
        lw=2.2,
        label="Gaussian paths",
    )
    axes[0].plot(
        validation.tau,
        validation.quadrature_covariance,
        color="C0",
        lw=1.9,
        ls="--",
        label="FP quadrature",
    )
    axes[0].plot(
        renewal.tau,
        renewal.field_covariance,
        color="C3",
        lw=1.7,
        ls=":",
        label="FPT renewal",
    )
    axes[0].set(
        xlabel=r"lag $\tau$",
        ylabel=r"filtered covariance $C_y(\tau)$",
        title="(a) Prescribed OU drive",
    )
    axes[0].legend(frameon=False)

    axes[1].plot(
        projection.tau,
        projection.dmft_covariance,
        color="k",
        lw=2.2,
        label=r"FP output $\sigma^2C_y$",
    )
    axes[1].plot(
        projection.tau,
        projection.projected_covariance,
        color="C3",
        lw=1.9,
        ls="--",
        label="one-mode OU projection",
    )
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].set(
        xlabel=r"lag $\tau$",
        ylabel=r"drive covariance $C_u(\tau)$",
        title="(b) One-mode closure diagnostic",
    )
    axes[1].legend(frameon=False)
    axes[2].plot(
        memory_decays,
        memory_field_errors,
        color="C0",
        marker="o",
        lw=1.9,
        label="filtered field",
    )
    axes[2].plot(
        memory_decays,
        memory_spike_errors,
        color="C3",
        marker="s",
        lw=1.9,
        label="regular spikes",
    )
    axes[2].plot(
        memory_decays,
        memory_return_radii,
        color="0.25",
        marker="^",
        lw=1.7,
        ls="--",
        label=r"return memory $|\lambda_2|$",
    )
    axes[2].plot(
        memory_decays,
        memory_scalar_estimates,
        color="C2",
        marker="x",
        lw=1.5,
        ls=":",
        label=r"$\mathbb{E}[e^{-\gamma T}]$",
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set(
        xlabel=r"OU decay $\gamma$",
        ylabel="dimensionless magnitude",
        title="(c) Return memory and error",
    )
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.tight_layout()
    figure_path = output / "lif_fp_quadrature_benchmark.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    axes[0].plot(
        renewal.tau,
        renewal.first_return_density,
        color="C2",
        lw=2.0,
    )
    axes[0].axvline(
        renewal.mean_interval,
        color="0.25",
        lw=1.2,
        ls="--",
        label=r"$\langle T\rangle$",
    )
    axes[0].set(
        xlabel=r"first-return time $T$",
        ylabel=r"density $f(T)$",
        title="(a) Spike-conditioned first return",
    )
    axes[0].legend(frameon=False)

    full_fp = renewal.full_fokker_planck
    axes[1].plot(
        full_fp.tau,
        full_fp.regular_spike_covariance,
        color="k",
        lw=2.2,
        label="full return operator",
    )
    axes[1].plot(
        renewal.tau,
        renewal.regular_spike_covariance,
        color="C3",
        lw=1.9,
        ls="--",
        label="renewal from FPT",
    )
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].set(
        xlabel=r"lag $\tau$",
        ylabel=r"regular spike covariance",
        title="(b) Cost of independent intervals",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.tight_layout()
    renewal_figure_path = output / "lif_first_return_renewal.png"
    fig.savefig(renewal_figure_path, dpi=180)
    plt.close(fig)

    renewal_field_error = float(
        np.linalg.norm(renewal.field_covariance - full_fp.field_covariance)
        / max(np.linalg.norm(full_fp.field_covariance), 1e-15)
    )
    renewal_regular_error = float(
        np.linalg.norm(
            renewal.regular_spike_covariance
            - full_fp.regular_spike_covariance
        )
        / max(np.linalg.norm(full_fp.regular_spike_covariance), 1e-15)
    )

    np.savez_compressed(
        output / "lif_fp_quadrature_benchmark.npz",
        tau_validation=validation.tau,
        covariance_paths=validation.path_covariance,
        covariance_quadrature=validation.quadrature_covariance,
        spike_covariance_paths=validation.path_spike_covariance,
        spike_covariance_quadrature=validation.quadrature_spike_covariance,
        first_return_density=renewal.first_return_density,
        survival_probability=renewal.survival_probability,
        covariance_renewal=renewal.field_covariance,
        spike_covariance_renewal=renewal.spike_covariance,
        memory_decays=memory_decays,
        memory_field_errors=memory_field_errors,
        memory_spike_errors=memory_spike_errors,
        memory_return_radii=memory_return_radii,
        memory_scalar_estimates=memory_scalar_estimates,
        tau_projection=projection.tau,
        covariance_dmft=projection.dmft_covariance,
        covariance_projected=projection.projected_covariance,
        projection_residuals=projection.parameter_residuals,
        projection_errors=projection.projection_errors,
    )
    summary = dict(
        profile=profile,
        prescribed_drive=dict(
            covariance_relative_error=validation.covariance_relative_error,
            equal_time_relative_error=validation.equal_time_relative_error,
            rate_relative_error=validation.rate_relative_error,
            path_rate=validation.path_rate,
            quadrature_rate=validation.quadrature_rate,
        ),
        first_return_renewal=dict(
            return_probability=renewal.return_probability,
            mean_interval=renewal.mean_interval,
            inverse_rate=1.0 / renewal.mean_rate,
            interval_cv=renewal.interval_cv,
            field_covariance_relative_error=renewal_field_error,
            regular_spike_covariance_relative_error=renewal_regular_error,
            memory_decays=memory_decays.tolist(),
            memory_field_errors=memory_field_errors.tolist(),
            memory_spike_errors=memory_spike_errors.tolist(),
            return_spectral_radius=return_map.spectral_radius,
            return_mixing_spikes=return_map.mixing_spikes,
            return_mixing_time=return_map.mixing_time,
            scalar_ou_memory=float(memory_scalar_estimates[1]),
            memory_return_radii=memory_return_radii.tolist(),
            memory_scalar_estimates=memory_scalar_estimates.tolist(),
            figure=str(renewal_figure_path),
        ),
        one_mode_projection=dict(
            converged=bool(projection.converged),
            drive_variance=float(projection.drive_variance),
            drive_decay=float(projection.drive_decay),
            final_parameter_residual=float(projection.parameter_residuals[-1]),
            final_projection_error=float(projection.projection_errors[-1]),
        ),
    )
    with (output / "lif_fp_quadrature_benchmark.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return figure_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "paper"), default="quick")
    parser.add_argument("--output-dir")
    arguments = parser.parse_args()
    print(run_lif_fp_benchmark(arguments.profile, arguments.output_dir))


if __name__ == "__main__":
    main()
