"""Generate a curated first-pass figure set for the RandomNet paper.

The script writes to the configured RandomNet results folder by default. It intentionally uses the
active theory paths only:

  * rate SCS
  * phase cusp closure with the exact same-spike contribution
  * binary exact linear theory
  * binary clipped-gain integral/effective closures

Use --quick for a short smoke run and --profile paper for slower, cleaner
figures.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rn_core import default_results_dir  # noqa: E402
from rn_rate import plot_rate_network  # noqa: E402
from rn_phase import (  # noqa: E402
    plot_phase_beta_scaling_diagnostic,
    plot_phase_operational_criticality,
    plot_phase_raster,
    plot_phase_theory_examples,
    plot_phase_theory_comparison,
    plot_u_timeseries,
)
from rn_binary import (  # noqa: E402
    plot_binary_network,
    plot_binary_network_N_convergence,
    plot_clipped_vs_linear,
)


PHASE_THEORY = dict(
    solver="cusp",
    q_method="hermite",
    n_quad=48,
    hermite_order=32,
)

PAPER_FIGURE_FILES = (
    "fig01_rate_scs.png",
    "fig02_phase_theory_examples.png",
    "fig03_phase_theory_comparison.png",
    "fig04_phase_beta_scaling.png",
    "fig06a_phase_u_timeseries.png",
    "fig06b_phase_raster.png",
    "fig07_binary_exact.png",
    "fig08_binary_N_convergence.png",
    "fig09_binary_clipped.png",
    "fig10_phase_criticality.png",
)


def _copy_named(plot_dir: Path, source_name: str, target_name: str, manifest: list[str]) -> None:
    src = plot_dir / source_name
    dst = plot_dir / target_name
    if not src.exists():
        raise FileNotFoundError(f"Expected figure was not produced: {src}")
    if src != dst:
        shutil.copyfile(src, dst)
    manifest.append(target_name)
    print(f"  -> {dst}")


def _phase_variants() -> list[dict]:
    return [
        dict(
            label="cusp closure",
            kwargs=dict(PHASE_THEORY),
            style=dict(color="C3", ls="--", lw=2.0),
        )
    ]


def make_figures(profile: str, figures: set[str], plot_dir: Path, jobs: int) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    quick = profile == "quick"

    rate_N = 384 if quick else 1536
    phase_N = 128 if quick else 256
    binary_N = 300 if quick else 800
    phase_T = 450.0 if quick else 900.0
    phase_reps = 1 if quick else 2
    phase_jobs = 1 if quick else jobs
    binary_T = 800.0 if quick else 2500.0
    phase_examples = [
        dict(I=1.0, alpha=1.0, beta=1.0, sigma=1.1 * 2.0 * 3.141592653589793,
             label=r"$\alpha=1,\ \beta=1,\ g=1.1$"),
        dict(I=1.0, alpha=1.0, beta=1.0, sigma=1.3 * 2.0 * 3.141592653589793,
             label=r"$\alpha=1,\ \beta=1,\ g=1.3$"),
        dict(I=1.0, alpha=2.0, beta=1.0, sigma=1.3 * 2.0 * 3.141592653589793,
             label=r"$\alpha=2,\ \beta=1,\ g=1.3$"),
        dict(I=1.0, alpha=1.0, beta=0.5, sigma=1.3 * 2.0 * 3.141592653589793,
             label=r"$\alpha=1,\ \beta=0.5,\ g=1.3$"),
    ]

    if "rate" in figures:
        print("\n[fig01] rate SCS baseline")
        plot_rate_network(
            sigma=2.2,
            N=rate_N,
            C0_guess=0.8,
            T=350.0 if quick else 1800.0,
            burn=100.0 if quick else 400.0,
            n_probe=192 if quick else 768,
            sim_reps=2 if quick else 5,
            plot_dir=str(plot_dir),
        )
        _copy_named(plot_dir, "rate_network_test.png", "fig01_rate_scs.png", manifest)

    if "phase-theory" in figures:
        print("\n[fig02] phase scalar theory examples")
        plot_phase_theory_examples(
            examples=phase_examples,
            N=phase_N,
            T=phase_T,
            dt=0.02,
            dtau=0.12 if quick else 0.08,
            tau_max=25.0 if quick else 32.0,
            burn=250.0,
            sim_reps=phase_reps,
            plot_dir=str(plot_dir),
            theory_variants=_phase_variants(),
        )
        _copy_named(plot_dir, "phase_theory_examples.png", "fig02_phase_theory_examples.png", manifest)

    if "phase-compare" in figures:
        print("\n[fig03] focused phase theory comparison")
        plot_phase_theory_comparison(
            N=phase_N,
            T=phase_T,
            dt=0.02,
            dtau=0.1 if quick else 0.06,
            tau_max=25.0 if quick else 35.0,
            burn=250.0,
            sim_reps=phase_reps,
            plot_dir=str(plot_dir),
            theory_variants=_phase_variants(),
        )
        _copy_named(plot_dir, "phase_theory_comparison.png", "fig03_phase_theory_comparison.png", manifest)

    if "phase-beta" in figures:
        print("\n[fig04] beta scaling diagnostic")
        plot_phase_beta_scaling_diagnostic(
            N=phase_N,
            T=phase_T,
            dt=0.02,
            dtau=0.12 if quick else 0.08,
            tau_max=18.0 if quick else 25.0,
            burn=250.0,
            sim_reps=phase_reps,
            plot_dir=str(plot_dir),
            theory_kwargs=dict(PHASE_THEORY),
        )
        _copy_named(plot_dir, "phase_beta_scaling_diagnostic.png", "fig04_phase_beta_scaling.png", manifest)

    if "criticality" in figures:
        print("\n[fig10] phase criticality diagnostics")
        plot_phase_operational_criticality(
            alpha_vals=(1.0, 1.25, 1.5, 2.0, 3.0),
            g_bounds=(0.25, 1.8),
            n_scan=10 if quick else 28,
            theory_kwargs=dict(tau_max=1.0, dtau=0.1, **PHASE_THEORY),
            plot_dir=str(plot_dir),
        )
        _copy_named(plot_dir, "phase_operational_criticality.png", "fig10_phase_criticality.png", manifest)

    if "phase-activity" in figures:
        print("\n[fig06] phase activity examples")
        activity_N = 96 if quick else 192
        activity_T = 150.0 if quick else 300.0
        plot_u_timeseries(
            N=activity_N,
            T=activity_T,
            burn=150.0,
            synapse_update="euler",
            plot_dir=str(plot_dir),
        )
        _copy_named(plot_dir, "phase_u_timeseries.png", "fig06a_phase_u_timeseries.png", manifest)
        plot_phase_raster(
            N=activity_N,
            T=activity_T,
            burn=150.0,
            synapse_update="euler",
            plot_dir=str(plot_dir),
        )
        _copy_named(plot_dir, "phase_raster.png", "fig06b_phase_raster.png", manifest)

    if "binary" in figures:
        print("\n[fig07] binary exact linear theory")
        plot_binary_network(
            sigma_vals=(0.5, 0.8, 0.95),
            N=binary_N,
            plot_dir=str(plot_dir),
            sim_method="tau-leap",
        )
        _copy_named(plot_dir, "binary_network_test.png", "fig07_binary_exact.png", manifest)

    if "binary-conv" in figures:
        print("\n[fig08] binary finite-size convergence")
        plot_binary_network_N_convergence(
            sigma=0.8,
            N_vals=(96, 192, 384) if quick else (128, 300, 800, 1600),
            plot_dir=str(plot_dir),
            sim_method="tau-leap",
        )
        _copy_named(plot_dir, "binary_network_N_convergence.png", "fig08_binary_N_convergence.png", manifest)

    if "binary-clipped" in figures:
        print("\n[fig09] binary clipped-gain closures")
        cache_path = plot_dir / "binary_clipped_sim_cache.npz"
        plot_clipped_vs_linear(
            sigma_vals=(0.7, 1.0, 1.3),
            N=binary_N,
            T=binary_T,
            burn=250.0,
            tau_max=16.0 if quick else 20.0,
            plot_dir=str(plot_dir),
            sim_method="tau-leap",
            sim_cache_path=str(cache_path),
            clipped_methods=("effective", "integral-telegraph"),
        )
        _copy_named(plot_dir, "clipped_vs_linear.png", "fig09_binary_clipped.png", manifest)

    inventory = [name for name in PAPER_FIGURE_FILES if (plot_dir / name).exists()]
    readme = plot_dir / "MANIFEST.txt"
    readme.write_text("\n".join(inventory) + "\n", encoding="utf-8")
    print(f"\nWrote manifest: {readme}")
    return manifest


def parse_args() -> argparse.Namespace:
    all_figures = [
        "rate",
        "phase-theory",
        "phase-compare",
        "phase-beta",
        "criticality",
        "phase-activity",
        "binary",
        "binary-conv",
        "binary-clipped",
    ]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("quick", "paper"),
        default="quick",
        help="quick is a smoke run; paper uses larger simulations.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=all_figures + ["all"],
        default=["all"],
        help="Subset of figure groups to generate.",
    )
    parser.add_argument(
        "--plot-dir",
        default=default_results_dir("paper"),
        help="Output directory.",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs for phase simulation helpers.")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
    args = parse_args()
    selected = set(args.figures)
    if "all" in selected:
        selected = {
            "rate",
            "phase-theory",
            "phase-compare",
            "phase-beta",
            "criticality",
            "phase-activity",
            "binary",
            "binary-conv",
            "binary-clipped",
        }
    make_figures(args.profile, selected, Path(args.plot_dir), args.jobs)


if __name__ == "__main__":
    main()
