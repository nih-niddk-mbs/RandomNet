"""Generate the independent ``randnetchaos`` paper calculations and figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

blas_threads = os.environ.get("RANDOMNET_BLAS_THREADS", "1")
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = blas_threads
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rn_chaos import (  # noqa: E402
    compute_spiking_covariance_comparison,
    compute_theta_covariance_comparison,
    plot_lif_summary,
    plot_lyapunov_scan,
    plot_replica_scan,
    plot_theta_activity,
    plot_theta_covariance_comparison,
    save_scan,
    scan_network_lyapunov,
    scan_replica_stability,
)


CHAOS_FIGURE_FILES = (
    "chaos_fig01_activity.png",
    "chaos_fig02_lyapunov.png",
    "chaos_fig03_replica.png",
    "chaos_fig04_covariances.png",
    "chaos_fig05_lif.png",
)


def make_chaos_figures(profile="quick", output_dir=None, jobs=1):
    """Run all calculations without assuming any machine-specific path."""
    quick = profile == "quick"
    output = Path(output_dir or os.environ.get("RANDOMNET_RESULTS_DIR", "~/randomnet-results"))
    output = output.expanduser()
    output.mkdir(parents=True, exist_ok=True)

    plot_theta_activity(
        output / CHAOS_FIGURE_FILES[0],
        N=96 if quick else 256,
        T=30.0 if quick else 120.0,
        burn=30.0 if quick else 150.0,
        dt=0.02 if quick else 0.01,
    )

    lyapunov = scan_network_lyapunov(
        [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
        N_values=(48, 96) if quick else (64, 128, 256),
        repetitions=1 if quick else 4,
        jobs=jobs,
        T=50.0 if quick else 250.0,
        burn=25.0 if quick else 100.0,
        dt=0.02 if quick else 0.01,
    )
    save_scan(output / "chaos_lyapunov_data.npz", lyapunov)
    plot_lyapunov_scan(lyapunov, output / CHAOS_FIGURE_FILES[1])

    if not quick:
        dt_values = np.asarray([0.02, 0.01, 0.005])
        dt_samples = []
        for step in dt_values:
            convergence = scan_network_lyapunov(
                [0.0, 0.1, 0.5, 2.0],
                N_values=(128,),
                repetitions=2,
                jobs=jobs,
                T=150.0,
                burn=75.0,
                dt=float(step),
            )
            dt_samples.append(convergence["samples"][0])
        np.savez_compressed(
            output / "chaos_lyapunov_dt_data.npz",
            dt=dt_values,
            sigma=np.asarray([0.0, 0.1, 0.5, 2.0]),
            samples=np.asarray(dt_samples),
        )

    replica = scan_replica_stability(
        [0.1, 0.25, 0.5, 0.75, 1.0, 1.25],
        n_time_values=(512, 1024) if quick else (1024, 2048, 4096),
        jobs=jobs,
        n_samples=24 if quick else 192,
        fixed_point_iterations=12 if quick else 60,
        power_iterations=4 if quick else 8,
        internal_dt=0.04 if quick else 0.02,
    )
    save_scan(output / "chaos_replica_data.npz", replica)
    plot_replica_scan(replica, output / CHAOS_FIGURE_FILES[2])

    covariance_results = []
    covariance_summary = []
    for index, sigma in enumerate((0.5, 2.0)):
        result = compute_theta_covariance_comparison(
            sigma,
            N=96 if quick else 256,
            T=100.0 if quick else 600.0,
            burn=50.0 if quick else 200.0,
            dt=0.04 if quick else 0.02,
            tau_max=8.0 if quick else 15.0,
            n_probe=64 if quick else 192,
            seed=314159 + index,
            stationary_kwargs=dict(
                n_time=1024 if quick else 16384,
                n_samples=24 if quick else 192,
                max_iter=12 if quick else 150,
                mixing=0.18 if quick else 0.12,
            ),
            include_twotime=False,
        )
        covariance_results.append(result)
        covariance_summary.append(
            dict(
                sigma=float(sigma),
                stationary_converged=bool(
                    result["stationary_diagnostics"]["converged"]
                ),
                stationary_final_residual=float(
                    result["stationary_diagnostics"]["residual_history"][-1]
                ),
            )
        )
        arrays = {
            key: value
            for key, value in result.items()
            if hasattr(value, "shape")
        }
        np.savez_compressed(output / f"chaos_covariance_sigma_{sigma:g}.npz", **arrays)
    plot_theta_covariance_comparison(
        covariance_results, output / CHAOS_FIGURE_FILES[3]
    )

    lif_sigma = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    lif_lyapunov = scan_network_lyapunov(
        lif_sigma,
        N_values=(48, 96) if quick else (64, 128, 256),
        repetitions=1 if quick else 4,
        jobs=jobs,
        I=2.0,
        phase_model="lif",
        T=50.0 if quick else 200.0,
        burn=30.0 if quick else 100.0,
        dt=0.02 if quick else 0.01,
        seed=20260917,
    )
    save_scan(output / "chaos_lif_lyapunov_data.npz", lif_lyapunov)
    lif_replica = scan_replica_stability(
        lif_sigma[1:],
        n_time_values=(512, 1024) if quick else (1024, 2048, 4096),
        jobs=jobs,
        I=2.0,
        phase_model="lif",
        n_samples=24 if quick else 192,
        fixed_point_iterations=80 if quick else 250,
        fixed_point_mixing=0.08 if quick else 0.06,
        fixed_point_tolerance=0.08 if quick else 0.05,
        power_iterations=4 if quick else 8,
        internal_dt=0.04 if quick else 0.02,
        seed=20260918,
    )
    save_scan(output / "chaos_lif_replica_data.npz", lif_replica)
    lif_covariance = compute_spiking_covariance_comparison(
        0.5,
        N=96 if quick else 256,
        T=120.0 if quick else 800.0,
        burn=60.0 if quick else 250.0,
        dt=0.04 if quick else 0.02,
        tau_max=6.0 if quick else 12.0,
        n_probe=64 if quick else 192,
        seed=20260919,
        I=2.0,
        phase_model="lif",
        phase_bin_width=0.1,
        stationary_kwargs=dict(
            n_time=1024 if quick else 4096,
            n_samples=32 if quick else 384,
            max_iter=100 if quick else 350,
            mixing=0.08 if quick else 0.06,
            tolerance=0.08 if quick else 0.05,
        ),
        include_twotime=False,
    )
    np.savez_compressed(
        output / "chaos_lif_covariance_sigma_0.5.npz",
        **{
            key: value
            for key, value in lif_covariance.items()
            if hasattr(value, "shape")
        },
    )
    plot_lif_summary(
        lif_lyapunov,
        lif_replica,
        lif_covariance,
        output / CHAOS_FIGURE_FILES[4],
    )
    summary = dict(
        profile=profile,
        lyapunov=dict(
            sigma=lyapunov["sigma"].tolist(),
            N=lyapunov["N"].tolist(),
            mean=lyapunov["mean"].tolist(),
            standard_error=lyapunov["standard_error"].tolist(),
        ),
        replica=dict(
            sigma=replica["sigma"].tolist(),
            duration=replica["duration"].tolist(),
            multiplier=replica["multiplier"].tolist(),
            critical_sigma=replica["critical_sigma"].tolist(),
            all_fixed_points_converged=bool(np.all(replica["converged"])),
        ),
        covariance=covariance_summary,
        lif=dict(
            lyapunov_mean=lif_lyapunov["mean"].tolist(),
            replica_multiplier=lif_replica["multiplier"].tolist(),
            all_fixed_points_converged=bool(np.all(lif_replica["converged"])),
            covariance_fixed_point_converged=bool(
                lif_covariance["stationary_diagnostics"]["converged"]
            ),
            covariance_final_residual=float(
                lif_covariance["stationary_diagnostics"]["residual_history"][-1]
            ),
        ),
    )
    with (output / "chaos_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return [str(output / name) for name in CHAOS_FIGURE_FILES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "paper"), default="quick")
    parser.add_argument("--output-dir")
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args()
    for path in make_chaos_figures(
        profile=arguments.profile,
        output_dir=arguments.output_dir,
        jobs=arguments.jobs,
    ):
        print(path)


if __name__ == "__main__":
    main()
