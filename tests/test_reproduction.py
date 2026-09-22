import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_paper_figures  # noqa: E402
import numpy as np
from make_paper_figures import (  # noqa: E402
    PAPER_FIGURE_FILES,
    PAPER_FIGURE_GROUPS,
    expand_figure_groups,
    resolve_output_dir,
)
from rn_core import default_results_dir  # noqa: E402
from rn_rate import (  # noqa: E402
    nonlinear_rate_fixed_q_response,
    sim_nonlinear_rate_network,
    theory_nonlinear_rate_dmft,
)


def test_results_environment_override_is_respected(monkeypatch, tmp_path):
    output_root = tmp_path / "external-results"
    monkeypatch.setenv("RANDOMNET_RESULTS_DIR", str(output_root))

    assert Path(default_results_dir()) == output_root
    assert Path(default_results_dir("table.csv")) == output_root / "table.csv"


def test_results_path_has_portable_home_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("RANDOMNET_RESULTS_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert Path(default_results_dir()) == tmp_path / "randomnet-results"


def test_quick_default_cannot_overwrite_publication_results(monkeypatch, tmp_path):
    monkeypatch.setattr(make_paper_figures.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("RANDOMNET_RESULTS_DIR", str(tmp_path / "paper-results"))

    assert resolve_output_dir("quick", None) == tmp_path / "randomnet-quick"
    assert resolve_output_dir("paper", None) == tmp_path / "paper-results"


def test_explicit_output_directory_has_priority(tmp_path):
    requested = tmp_path / "chosen"

    assert resolve_output_dir("quick", str(requested)) == requested
    assert resolve_output_dir("paper", str(requested)) == requested


def test_figure_names_match_manuscript_order():
    assert PAPER_FIGURE_FILES == (
        "fig01_rate_nonlinear.png",
        "fig02_binary_sigmoid.png",
        "fig03_binary_N_convergence.png",
        "fig04_binary_hierarchy.png",
        "fig04_binary_saturating.png",
        "fig05_phase_closure_comparison.png",
        "fig06_phase_theory_examples.png",
        "fig07_phase_theory_comparison.png",
        "fig08_phase_beta_scaling.png",
        "fig09_phase_N_convergence.png",
        "fig10_phase_C33.png",
        "fig11_phase_criticality.png",
        "fig12_phase_u_timeseries.png",
        "fig13_phase_raster.png",
        "chaos_fig01_activity.png",
        "chaos_fig02_lyapunov.png",
        "chaos_fig03_replica.png",
        "chaos_fig04_covariances.png",
        "chaos_fig05_lif.png",
        "lif_fp_quadrature_benchmark.png",
    )


def test_external_driver_can_expand_figure_groups():
    assert expand_figure_groups(["all"]) == set(PAPER_FIGURE_GROUPS)
    assert expand_figure_groups(["rate", "binary"]) == {"rate", "binary"}


def test_nonlinear_rate_simulation_and_dmft_are_finite():
    tau_sim, simulation, _ = sim_nonlinear_rate_network(
        N=24,
        sigma=1.3,
        T=4.0,
        burn=2.0,
        dt=0.05,
        n_probe=8,
        tau_max=1.0,
        rng=np.random.default_rng(123),
    )
    tau_theory, theory, diagnostics = theory_nonlinear_rate_dmft(
        sigma=1.3,
        internal_dt=0.05,
        n_time=256,
        n_samples=4,
        n_batches=2,
        n_validation_batches=1,
        warmup_cycles=1,
        max_iter=2,
        tau_max=1.0,
        return_diagnostics=True,
    )

    assert tau_sim.shape == simulation["Cuu"].shape
    assert tau_theory.shape == theory["Cuu"].shape
    assert np.all(np.isfinite(simulation["Q"]))
    assert np.all(np.isfinite(theory["Q"]))
    assert diagnostics["sigma_critical_linear"] == 1.0
    assert diagnostics["n_batches"] == 2
    assert diagnostics["n_validation_batches"] == 1
    assert np.isfinite(diagnostics["final_amplitude_residual"])
    assert np.isfinite(diagnostics["validation_residual"])


def test_nonlinear_rate_fixed_q_response_keeps_input_and_output_separate():
    local_rng = np.random.default_rng(321)
    output_paths = 0.5 + 0.05 * local_rng.normal(size=(6, 256))
    tau, covariances, diagnostics = nonlinear_rate_fixed_q_response(
        output_paths,
        sigma=0.2,
        dt=0.05,
        n_samples=6,
        warmup_cycles=1,
        tau_max=0.5,
        seed=123,
    )

    assert tau.shape == covariances["Q_in"].shape
    assert covariances["Q_out"].shape == tau.shape
    assert np.all(np.isfinite(covariances["Cuu"]))
    assert diagnostics["n_samples"] == 6


def test_phase_comparison_variants_put_two_time_theory_first():
    variants = make_paper_figures._phase_variants(
        "quick", include_scalar=True, include_twotime=True
    )

    assert [variant["kwargs"]["solver"] for variant in variants] == [
        "twotime_dmft",
        "density",
        "cusp",
    ]
    assert variants[0]["style"]["zorder"] > variants[1]["style"]["zorder"]
