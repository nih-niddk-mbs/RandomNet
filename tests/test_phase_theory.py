import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rn_phase import theory_phase_autocorr  # noqa: E402


def test_phase_theory_obeys_same_spike_cusp():
    I = 1.0
    alpha = 1.0
    beta = 0.8
    sigma = 1.2 * 2.0 * np.pi
    dtau = 1e-4
    n_quad = 48

    _tau, covariance, _ = theory_phase_autocorr(
        solver="cusp",
        I=I,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        tau_max=5 * dtau,
        dtau=dtau,
        n_quad=n_quad,
        q_method="hermite",
    )

    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    u = np.sqrt(2.0 * covariance[0]) * nodes
    rate = alpha * np.clip(I + u, 0.0, None) ** (1.0 / alpha) / (2.0 * np.pi)
    mean_rate = np.dot(weights, rate) / np.sqrt(np.pi)
    expected_slope = -0.5 * beta**2 * sigma**2 * mean_rate
    measured_slope = (covariance[1] - covariance[0]) / dtau

    assert measured_slope == pytest.approx(expected_slope, rel=2e-3)


def test_inflated_initial_condition_solver_is_removed():
    with pytest.raises(ValueError, match="density.*cusp"):
        theory_phase_autocorr(solver="inflated_ic", tau_max=0.2, dtau=0.1)


def test_incomplete_fixed_q_solver_is_not_exposed_as_full_2pi():
    with pytest.raises(ValueError, match="fixed_q"):
        theory_phase_autocorr(solver="2pi", tau_max=0.2, dtau=0.1)
    with pytest.raises(ValueError, match="fixed_q"):
        theory_phase_autocorr(solver="full", tau_max=0.2, dtau=0.1)


def test_gaussian_2pi_wrapper_requires_matched_phase_and_event_bins():
    with pytest.raises(ValueError, match="internal_dt = period / n_phase"):
        theory_phase_autocorr(
            solver="gaussian_2pi",
            n_phase=33,
            internal_dt=0.1,
            tau_max=0.2,
            dtau=0.1,
        )


def test_fixed_q_diagnostic_returns_four_field_covariance_sectors():
    _tau, covariance, _sigma_c, diagnostics = theory_phase_autocorr(
        solver="fixed_q",
        sigma=0.3,
        tau_max=0.4,
        dtau=0.1,
        internal_dt=0.2,
        n_time=8,
        n_phase=7,
        max_iter=80,
        tolerance=1e-6,
        return_diagnostics=True,
    )
    assert diagnostics["converged"]
    assert np.all(np.isfinite(covariance))
    assert diagnostics["C13_kernel"].shape == (8, 8)
    assert diagnostics["C31_kernel"].shape == (8, 8)
    assert diagnostics["C33_kernel"].shape == (8, 8)
    assert "min_covariance_eigenvalue" in diagnostics
    assert "min_flux_eigenvalue" in diagnostics
    assert "max_normalized_covariance" in diagnostics
    assert isinstance(diagnostics["physical"], bool)


def test_fixed_q_paper_curve_is_rejected_by_covariance_invariants():
    _tau, _covariance, _sigma_c, diagnostics = theory_phase_autocorr(
        solver="fixed_q",
        sigma=np.pi,
        tau_max=2.0,
        dtau=0.1,
        internal_dt=0.1,
        n_time=96,
        n_phase=25,
        max_iter=500,
        mixing=0.12,
        tolerance=2e-6,
        return_diagnostics=True,
    )
    assert diagnostics["converged"]
    assert not diagnostics["physical"]
    assert diagnostics["min_covariance_eigenvalue"] < 0.0
    assert diagnostics["min_flux_eigenvalue"] < 0.0
    assert diagnostics["max_normalized_covariance"] > 1.0


def test_phase_density_theory_contains_threshold_return_peak():
    dt = 0.02
    period = 2.0 * np.pi
    _tau, covariance, _sigma_c, diagnostics = theory_phase_autocorr(
        solver="density",
        I=1.0,
        alpha=1.0,
        sigma=0.0,
        tau_max=8.0,
        dtau=0.05,
        internal_dt=dt,
        n_time=4096,
        n_samples=64,
        max_iter=2,
        return_diagnostics=True,
    )

    off_covariance = diagnostics["off_spike_covariance"]
    return_bin = int(round(period / dt))
    neighborhood = off_covariance[return_bin - 4:return_bin + 5]
    background = np.median(off_covariance[return_bin - 80:return_bin + 80])

    assert np.all(covariance == 0.0)
    assert np.max(neighborhood) > background + 1.0
    density_covariance = diagnostics["phase_density_covariance"]
    density_return = density_covariance[return_bin - 4:return_bin + 5]
    assert np.max(density_return) > 0.95 * density_covariance[0]


def test_phase_density_theory_obeys_same_spike_cusp():
    beta = 0.8
    sigma = 1.1 * 2.0 * np.pi
    dt = 0.02
    _tau, covariance, _sigma_c, diagnostics = theory_phase_autocorr(
        solver="density",
        I=1.0,
        alpha=1.0,
        beta=beta,
        sigma=sigma,
        tau_max=0.1,
        dtau=dt,
        internal_dt=dt,
        n_time=4096,
        n_samples=64,
        max_iter=40,
        mixing=0.18,
        return_diagnostics=True,
    )

    measured_slope = (covariance[1] - covariance[0]) / dt
    expected_slope = -0.5 * beta**2 * sigma**2 * diagnostics["mean_rate"]
    assert measured_slope == pytest.approx(expected_slope, rel=0.04)


def test_twotime_dmft_solver_retains_uncoupled_phase_memory():
    _tau, covariance, _sigma_c, diagnostics = theory_phase_autocorr(
        solver="twotime",
        sigma=0.0,
        tau_max=7.0,
        dtau=0.1,
        internal_dt=0.1,
        n_time=96,
        n_samples=96,
        max_iter=1,
        transient_fraction=0.1,
        return_diagnostics=True,
        seed=11,
    )

    C33 = diagnostics["phase_density_covariance"]
    return_bin = int(round((2.0 * np.pi) / 0.1))
    assert np.all(covariance == 0.0)
    assert np.max(C33[return_bin - 2:return_bin + 3]) > 0.8 * C33[0]
    assert np.allclose(
        diagnostics["C31_kernel"], diagnostics["C13_kernel"].T
    )
    for response in diagnostics["phase_response_modes"].values():
        assert np.all(response[np.triu_indices_from(response)] == 0.0)
