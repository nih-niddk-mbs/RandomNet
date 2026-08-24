import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rn_lif_fp import (  # noqa: E402
    build_lif_ou_generator,
    build_lif_ou_absorbing_generator,
    compute_lif_ou_return_map,
    solve_lif_ou_first_return_renewal,
    solve_lif_ou_fokker_planck,
    stationary_lif_ou_mass,
)


def test_lif_ou_generator_conserves_probability_and_stationary_mass():
    grid = build_lif_ou_generator(
        I=2.0,
        drive_variance=0.02,
        n_voltage=48,
        n_drive=31,
    )
    column_sums = np.asarray(grid.generator.sum(axis=0)).ravel()
    assert np.max(np.abs(column_sums)) < 1e-10

    mass = stationary_lif_ou_mass(grid)
    assert np.sum(mass) == pytest.approx(1.0)
    assert np.min(mass) >= 0.0
    assert np.linalg.norm(grid.generator @ mass, ord=1) < 1e-8


def test_lif_ou_quadrature_recovers_drive_variance_and_return_peak():
    result = solve_lif_ou_fokker_planck(
        I=2.0,
        drive_variance=0.02,
        drive_decay=1.0,
        tau_max=1.2,
        dt=0.03,
        n_voltage=72,
        n_drive=41,
    )
    assert abs(result.drive_mean) < 1e-8
    assert result.drive_variance == pytest.approx(0.02, rel=0.18)
    assert result.mean_rate == pytest.approx(1.0 / np.log(2.0), rel=0.025)

    period_index = int(round(np.log(2.0) / 0.03))
    refractory_index = int(round(0.3 / 0.03))
    assert result.regular_spike_covariance[refractory_index] < 0.0
    assert result.regular_spike_covariance[period_index] > 0.0
    assert np.all(np.isfinite(result.field_covariance))


def test_lif_ou_absorbing_generator_loses_only_threshold_flux():
    grid = build_lif_ou_generator(
        I=2.0,
        drive_variance=0.02,
        n_voltage=48,
        n_drive=31,
    )
    absorbing = build_lif_ou_absorbing_generator(grid)
    column_sums = np.asarray(absorbing.sum(axis=0)).ravel()
    assert column_sums == pytest.approx(-grid.event_hazard, abs=1e-10)


def test_first_return_renewal_has_correct_mean_interval_and_finite_covariance():
    result = solve_lif_ou_first_return_renewal(
        I=2.0,
        drive_variance=0.02,
        drive_decay=1.0,
        tau_max=4.0,
        dt=0.025,
        n_voltage=64,
        n_drive=41,
    )
    assert result.return_probability > 0.999
    assert np.all(np.diff(result.survival_probability) <= 1e-8)
    assert result.mean_interval == pytest.approx(1.0 / result.mean_rate, rel=0.035)
    assert result.interval_cv > 0.0
    assert np.all(np.isfinite(result.field_covariance))
    assert np.max(result.first_return_density) > result.mean_rate

    return_map = compute_lif_ou_return_map(result.full_fokker_planck)
    assert np.sum(return_map.transition_matrix, axis=0) == pytest.approx(1.0)
    assert return_map.column_sum_error < 1e-8
    assert return_map.invariance_error < 1e-8
    assert 0.0 < return_map.spectral_radius < 1.0
    assert return_map.mixing_spikes > 0.0
