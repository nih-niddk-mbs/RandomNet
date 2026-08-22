"""Generate the figure set for the RandomNet paper.

Paper runs write to the configured external results folder; quick runs use an
isolated temporary folder. The active theory paths are:

  * rate SCS
  * phase-density DMFT with threshold returns and finite-size convergence
  * binary sigmoid-network DMFT and its controlled affine tangent limit

Use ``--profile quick`` for a transient smoke run and ``--profile paper`` for
the publication calculations.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
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
    plot_phase_network_N_convergence,
    plot_phase_density_correlation,
    plot_phase_gaussian_2pi_comparison,
    estimate_phase_C33_validity,
    plot_phase_operational_criticality,
    plot_phase_raster,
    plot_phase_theory_examples,
    plot_phase_theory_comparison,
    plot_u_timeseries,
)
from rn_binary import (  # noqa: E402
    plot_binary_network,
    plot_binary_network_N_convergence,
    plot_binary_theory_hierarchy,
)


PHASE_SCALAR_THEORY = dict(
    solver="cusp",
    q_method="hermite",
    n_quad=48,
    hermite_order=32,
)

PAPER_FIGURE_FILES = (
    "fig01_rate_scs.png",
    "fig02_binary_sigmoid.png",
    "fig03_binary_N_convergence.png",
    "fig04_binary_hierarchy.png",
    "fig05_phase_closure_comparison.png",
    "fig06_phase_theory_examples.png",
    "fig07_phase_theory_comparison.png",
    "fig08_phase_beta_scaling.png",
    "fig09_phase_N_convergence.png",
    "fig10_phase_C33.png",
    "fig11_phase_criticality.png",
    "fig12_phase_u_timeseries.png",
    "fig13_phase_raster.png",
)

PAPER_FIGURE_GROUPS = (
    "rate",
    "phase-theory",
    "phase-compare",
    "phase-beta",
    "criticality",
    "phase-conv",
    "phase-activity",
    "phase-density",
    "phase-2pi",
    "phase-validity",
    "binary",
    "binary-conv",
    "binary-hierarchy",
)


def expand_figure_groups(groups) -> set[str]:
    """Expand the public ``all`` alias for CLI and external drivers."""
    selected = set(groups)
    if "all" in selected:
        return set(PAPER_FIGURE_GROUPS)
    unknown = selected.difference(PAPER_FIGURE_GROUPS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown paper figure group(s): {names}")
    return selected


def resolve_output_dir(profile: str, requested: str | None) -> Path:
    """Resolve output without letting a smoke run overwrite paper results."""
    if requested is not None:
        return Path(requested).expanduser()
    if profile == "quick":
        return Path(tempfile.gettempdir()) / "randomnet-quick"
    return Path(default_results_dir())


def _copy_named(plot_dir: Path, source_name: str, target_name: str, manifest: list[str]) -> None:
    src = plot_dir / source_name
    dst = plot_dir / target_name
    if not src.exists():
        raise FileNotFoundError(f"Expected figure was not produced: {src}")
    if src != dst:
        shutil.copyfile(src, dst)
        src.unlink()
    manifest.append(target_name)
    print(f"  -> {dst}")


def _phase_variants(
    profile: str,
    include_scalar: bool = False,
    include_twotime: bool = False,
) -> list[dict]:
    quick = profile == "quick"
    variants = []
    if include_twotime:
        variants.append(
            dict(
                label="two-time event-DMFT",
                kwargs=dict(
                    solver="twotime_dmft",
                    internal_dt=0.1 if quick else 0.05,
                    n_time=320 if quick else 800,
                    n_samples=384 if quick else 1024,
                    max_iter=60 if quick else 140,
                    mixing=0.08 if quick else 0.04,
                    tolerance=0.10 if quick else 0.04,
                    transient_fraction=0.55,
                    seed=271830,
                ),
                style=dict(color="C0", ls="-", lw=2.4, zorder=4),
            )
        )
    variants.append(
        dict(
            label="stationary event-DMFT",
            kwargs=dict(
                solver="density",
                internal_dt=0.02,
                n_time=4096 if quick else 16384,
                n_samples=32 if quick else 96,
                max_iter=35 if quick else 60,
                mixing=0.18,
                tolerance=0.05 if quick else 0.03,
                seed=1729,
            ),
            style=dict(color="C3", ls="--", lw=1.9, zorder=3),
        )
    )
    if include_scalar:
        variants.append(
            dict(
                label="smooth scalar closure",
                kwargs=dict(PHASE_SCALAR_THEORY),
                style=dict(color="0.60", ls=":", lw=1.5, zorder=1),
            )
        )
    return variants


def make_figures(profile: str, figures: set[str], plot_dir: Path, jobs: int) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = plot_dir
    manifest: list[str] = []
    quick = profile == "quick"

    rate_N = 384 if quick else 1536
    phase_N = 128 if quick else 256
    binary_N = 128 if quick else 800
    phase_T = 450.0 if quick else 900.0
    phase_reps = 1 if quick else 2
    phase_jobs = 1 if quick else jobs
    binary_T = 250.0 if quick else 2500.0
    binary_theory = dict(
        internal_dt=0.05 if quick else 0.025,
        n_time=1024 if quick else 8192,
        n_samples=16 if quick else 256,
        max_iter=25 if quick else 60,
        mixing=0.22 if quick else 0.18,
        tolerance=0.05 if quick else 0.01,
    )
    phase_examples = [
        dict(I=1.0, alpha=1.0, beta=1.0, sigma=1.1 * 2.0 * 3.141592653589793,
             label=r"$\alpha=1,\ \beta=1,\ g=1.1$"),
        dict(I=1.0, alpha=1.0, beta=1.0, sigma=1.3 * 2.0 * 3.141592653589793,
             label=r"$\alpha=1,\ \beta=1,\ g=1.3$"),
        dict(I=1.0, alpha=2.0, beta=1.0, sigma=1.3 * 2.0 * 3.141592653589793,
             label=r"$\alpha=2,\ \beta=1,\ g=1.3$"),
        dict(I=1.0, alpha=1.0, beta=2.0, sigma=1.3 * 2.0 * 3.141592653589793,
             label=r"$\alpha=1,\ \beta=2,\ g=1.3$"),
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
        print("\n[fig06] two-time and stationary event-DMFT examples")
        plot_phase_theory_examples(
            examples=phase_examples,
            N=phase_N,
            T=phase_T,
            dt=0.02,
            dtau=0.12 if quick else 0.08,
            tau_max=15.0,
            burn=250.0,
            sim_reps=phase_reps,
            n_probe=96 if quick else 192,
            seed=314159,
            plot_dir=str(plot_dir),
            theory_variants=_phase_variants(profile, include_twotime=True),
        )
        _copy_named(plot_dir, "phase_theory_examples.png", "fig06_phase_theory_examples.png", manifest)

    if "phase-compare" in figures:
        print("\n[fig07] focused phase theory comparison")
        plot_phase_theory_comparison(
            N=phase_N,
            T=350.0 if quick else 700.0,
            dt=0.02,
            dtau=0.1 if quick else 0.05,
            tau_max=15.0,
            burn=250.0,
            sim_reps=phase_reps,
            n_probe=96 if quick else 192,
            seed=161803,
            plot_dir=str(plot_dir),
            theory_variants=_phase_variants(
                profile,
                include_scalar=True,
                include_twotime=True,
            ),
        )
        _copy_named(plot_dir, "phase_theory_comparison.png", "fig07_phase_theory_comparison.png", manifest)

    if "phase-beta" in figures:
        print("\n[fig08] beta scaling diagnostic")
        plot_phase_beta_scaling_diagnostic(
            beta_vals=(1.0, 2.0, 3.0),
            N=phase_N,
            T=phase_T,
            dt=0.02,
            dtau=0.12 if quick else 0.08,
            tau_max=14.0 if quick else 15.0,
            burn=250.0,
            sim_reps=phase_reps,
            n_probe=96 if quick else 192,
            seed=141421,
            plot_dir=str(plot_dir),
            theory_variants=_phase_variants(profile, include_twotime=True),
        )
        _copy_named(plot_dir, "phase_beta_scaling_diagnostic.png", "fig08_phase_beta_scaling.png", manifest)

    if "criticality" in figures:
        print("\n[fig11] phase criticality diagnostics")
        plot_phase_operational_criticality(
            alpha_vals=(1.0, 1.25, 1.5, 2.0, 3.0),
            g_bounds=(0.25, 1.8),
            n_scan=10 if quick else 28,
            theory_kwargs=dict(tau_max=1.0, dtau=0.1, **PHASE_SCALAR_THEORY),
            plot_dir=str(plot_dir),
        )
        _copy_named(plot_dir, "phase_operational_criticality.png", "fig11_phase_criticality.png", manifest)

    if "phase-conv" in figures:
        print("\n[fig09] phase finite-size convergence")
        plot_phase_network_N_convergence(
            N_vals=(48, 96, 192) if quick else (64, 128, 256, 512),
            T=450.0 if quick else 900.0,
            dt=0.02,
            dtau=0.1 if quick else 0.06,
            tau_max=14.0 if quick else 15.0,
            burn=250.0,
            sim_reps=1 if quick else 3,
            theory_variants=_phase_variants(profile, include_twotime=True),
            plot_dir=str(plot_dir),
        )
        _copy_named(
            plot_dir,
            "phase_network_N_convergence.png",
            "fig09_phase_N_convergence.png",
            manifest,
        )

    if "phase-activity" in figures:
        print("\n[fig12-13] phase activity examples")
        activity_N = 96 if quick else 192
        activity_T = 150.0 if quick else 300.0
        plot_u_timeseries(
            N=activity_N,
            T=activity_T,
            burn=150.0,
            synapse_update="euler",
            plot_dir=str(plot_dir),
        )
        _copy_named(plot_dir, "phase_u_timeseries.png", "fig12_phase_u_timeseries.png", manifest)
        plot_phase_raster(
            N=activity_N,
            T=activity_T,
            burn=150.0,
            synapse_update="euler",
            plot_dir=str(plot_dir),
        )
        _copy_named(plot_dir, "phase_raster.png", "fig13_phase_raster.png", manifest)

    if "phase-density" in figures:
        print("\n[fig10] threshold phase-density correlation")
        plot_phase_density_correlation(
            g_vals=(0.25, 0.5, 1.3),
            N=128 if quick else 256,
            T=450.0 if quick else 900.0,
            dt=0.02,
            dtau=0.1 if quick else 0.05,
            tau_max=15.0,
            burn=250.0,
            n_probe=96 if quick else 192,
            sim_reps=1 if quick else 2,
            phase_bin_width=0.25,
            theory_variants=_phase_variants(profile, include_twotime=True),
            plot_dir=str(plot_dir),
        )
        _copy_named(
            plot_dir,
            "phase_density_correlation.png",
            "fig10_phase_C33.png",
            manifest,
        )

    if "phase-2pi" in figures:
        print("\n[fig05] phase closure hierarchy")
        diagnostic_dir = plot_dir / ".figure_work" / "phase_2pi"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        plot_phase_gaussian_2pi_comparison(
            g_vals=(0.25, 0.5, 1.3),
            N=128 if quick else 256,
            T=350.0 if quick else 700.0,
            dt=0.02,
            dtau=0.1 if quick else 0.05,
            tau_max=15.0,
            burn=200.0 if quick else 250.0,
            n_probe=96 if quick else 192,
            sim_reps=1 if quick else 2,
            n_phase=65 if quick else 129,
            stationary_kwargs=dict(
                internal_dt=0.02,
                n_time=4096 if quick else 8192,
                n_samples=32 if quick else 64,
                max_iter=35 if quick else 50,
                mixing=0.18,
                tolerance=0.05 if quick else 0.03,
            ),
            twotime_kwargs=dict(
                internal_dt=0.1 if quick else 0.05,
                n_time=320 if quick else 800,
                n_samples=384 if quick else 1024,
                max_iter=60 if quick else 140,
                mixing=0.08 if quick else 0.04,
                tolerance=0.10 if quick else 0.04,
                transient_fraction=0.55,
            ),
            sim_cache_path=str(
                cache_dir / "phase_gaussian_2pi_sim_cache.npz"
            ),
            plot_dir=str(diagnostic_dir),
        )
        _copy_named(
            plot_dir,
            ".figure_work/phase_2pi/phase_gaussian_2pi_comparison.png",
            "fig05_phase_closure_comparison.png",
            manifest,
        )

    if "phase-validity" in figures:
        print("\n[phase validity] finite-N C33 error estimates")
        estimate_phase_C33_validity(
            N_vals=(48, 96, 192) if quick else (64, 128, 256, 512),
            T=400.0 if quick else 700.0,
            sim_reps=1 if quick else 3,
            theory_kwargs=dict(_phase_variants(profile)[0]["kwargs"]),
            output_path=str(plot_dir / "phase_C33_validity.csv"),
        )

    if "binary" in figures:
        print("\n[fig02] binary sigmoid dynamic DMFT")
        plot_binary_network(
            sigma_vals=(1.0, 2.0, 3.0),
            N=binary_N,
            T=binary_T,
            burn=50.0 if quick else 300.0,
            dt=0.05 if quick else 0.02,
            tau_max=12.0 if quick else 20.0,
            sim_reps=1 if quick else 3,
            theory_kwargs=binary_theory,
            plot_dir=str(plot_dir),
        )
        _copy_named(plot_dir, "binary_network_test.png", "fig02_binary_sigmoid.png", manifest)

    if "binary-conv" in figures:
        print("\n[fig03] binary finite-size convergence")
        plot_binary_network_N_convergence(
            sigma=2.0,
            N_vals=(48, 96, 192) if quick else (128, 300, 800, 1600),
            T=binary_T,
            burn=50.0 if quick else 300.0,
            dt=0.05 if quick else 0.02,
            tau_max=12.0 if quick else 20.0,
            theory_kwargs=binary_theory,
            plot_dir=str(plot_dir),
        )
        _copy_named(plot_dir, "binary_network_N_convergence.png", "fig03_binary_N_convergence.png", manifest)

    if "binary-hierarchy" in figures:
        print("\n[fig04] binary sigmoid theory hierarchy")
        plot_binary_theory_hierarchy(
            sigma_vals=(1.0, 2.0, 3.0),
            N=binary_N,
            T=binary_T,
            burn=50.0 if quick else 300.0,
            dt=0.05 if quick else 0.02,
            tau_max=12.0 if quick else 20.0,
            theory_kwargs=binary_theory,
            plot_dir=str(plot_dir),
        )
        _copy_named(plot_dir, "binary_theory_hierarchy.png", "fig04_binary_hierarchy.png", manifest)

    inventory = [name for name in PAPER_FIGURE_FILES if (plot_dir / name).exists()]
    readme = plot_dir / "MANIFEST.txt"
    readme.write_text("\n".join(inventory) + "\n", encoding="utf-8")
    shutil.rmtree(plot_dir / ".figure_work", ignore_errors=True)
    print(f"\nWrote manifest: {readme}")
    return manifest


def parse_args() -> argparse.Namespace:
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
        choices=list(PAPER_FIGURE_GROUPS) + ["all"],
        default=["all"],
        help="Subset of figure groups to generate.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help=(
            "Output directory. Defaults to a temporary directory for quick "
            "runs and RANDOMNET_RESULTS_DIR (or the external results default) "
            "for paper runs."
        ),
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs for phase simulation helpers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = expand_figure_groups(args.figures)
    output_dir = resolve_output_dir(args.profile, args.plot_dir)
    print(f"Output directory: {output_dir}")
    make_figures(args.profile, selected, output_dir, args.jobs)


if __name__ == "__main__":
    main()
