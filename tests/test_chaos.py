import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rn_phase import (  # noqa: E402
    _advance_phase,
    _lif_network_tangent_step,
    _phase_network_step,
    _theta_network_tangent_step,
    maximal_lyapunov_phase_network,
    phase_velocity,
    theory_phase_density_autocorr,
)


def test_transformed_theta_velocity_matches_original_theta_equation():
    I = 1.7
    phase = np.linspace(-2.7, 2.7, 17)
    recurrent_field = np.linspace(-0.8, 0.9, len(phase))
    theta = 2.0 * np.arctan(np.sqrt(I) * np.tan(phase / 2.0))
    dtheta_dphase = (
        np.sqrt(I)
        / (np.cos(phase / 2.0) ** 2 + I * np.sin(phase / 2.0) ** 2)
    )
    original_velocity = (
        1.0
        - np.cos(theta)
        + (I + recurrent_field) * (1.0 + np.cos(theta))
    )
    transformed = phase_velocity(
        phase, recurrent_field, I=I, model="theta"
    )
    assert transformed == pytest.approx(original_velocity / dtheta_dphase)


def test_transformed_theta_threshold_velocity_is_input_independent():
    recurrent_field = np.array([-100.0, -1.0, 0.0, 3.0, 100.0])
    velocity = phase_velocity(
        np.full_like(recurrent_field, np.pi),
        recurrent_field,
        I=2.25,
        model="theta",
    )
    assert velocity == pytest.approx(np.full_like(velocity, 3.0))


def test_lif_exact_step_resets_after_the_analytic_period():
    period = np.log(2.0)
    state, counts, _ = _advance_phase(
        np.array([0.0]),
        np.array([0.0]),
        I=2.0,
        alpha=1.0,
        dt=period + 0.1,
        model="lif",
    )
    assert counts[0] == 1
    assert state[0] == pytest.approx(2.0 * (1.0 - np.exp(-0.1)))


def test_theta_tangent_step_matches_directional_finite_difference():
    I = 1.0
    beta = 0.8
    dt = 0.02
    weights = np.array(
        [[0.0, 0.1, -0.1], [-0.2, 0.0, 0.2], [0.15, -0.15, 0.0]]
    )
    state = np.array([np.pi - 0.02, -1.1, 0.4])
    field = np.array([0.2, -0.3, 0.1])
    tangent_state = np.array([0.3, -0.2, 0.1])
    tangent_field = np.array([-0.15, 0.25, 0.05])

    base_state, base_field, jac_state, jac_field, counts = (
        _theta_network_tangent_step(
            state,
            field,
            tangent_state,
            tangent_field,
            weights,
            I,
            beta,
            dt,
        )
    )
    epsilon = 1e-7
    perturbed_state, perturbed_field, perturbed_counts = _phase_network_step(
        state + epsilon * tangent_state,
        field + epsilon * tangent_field,
        weights,
        I,
        1.0,
        beta,
        dt,
        "theta",
    )
    state_difference = (
        (perturbed_state - base_state + np.pi) % (2.0 * np.pi)
    ) - np.pi
    assert np.array_equal(counts, perturbed_counts)
    assert state_difference / epsilon == pytest.approx(
        jac_state, rel=2e-5, abs=2e-6
    )
    assert (perturbed_field - base_field) / epsilon == pytest.approx(
        jac_field, rel=2e-5, abs=2e-6
    )


def test_lif_tangent_step_matches_directional_finite_difference_at_reset():
    I = 2.0
    beta = 0.8
    dt = 0.04
    weights = np.array(
        [[0.0, 0.1, -0.1], [-0.2, 0.0, 0.2], [0.15, -0.15, 0.0]]
    )
    state = np.array([0.96, 0.3, 0.7])
    field = np.array([0.2, -0.3, 0.1])
    tangent_state = np.array([0.3, -0.2, 0.1])
    tangent_field = np.array([-0.15, 0.25, 0.05])

    base_state, base_field, jac_state, jac_field, counts = (
        _lif_network_tangent_step(
            state,
            field,
            tangent_state,
            tangent_field,
            weights,
            I,
            beta,
            dt,
        )
    )
    epsilon = 1e-7
    perturbed_state, perturbed_field, perturbed_counts = _phase_network_step(
        state + epsilon * tangent_state,
        field + epsilon * tangent_field,
        weights,
        I,
        1.0,
        beta,
        dt,
        "lif",
    )
    assert counts[0] == 1
    assert np.array_equal(counts, perturbed_counts)
    assert (perturbed_state - base_state) / epsilon == pytest.approx(
        jac_state, rel=2e-5, abs=2e-6
    )
    assert (perturbed_field - base_field) / epsilon == pytest.approx(
        jac_field, rel=2e-5, abs=2e-6
    )


def test_uncoupled_theta_network_has_neutral_maximal_exponent():
    exponent = maximal_lyapunov_phase_network(
        N=24,
        sigma=0.0,
        T=30.0,
        burn=5.0,
        dt=0.02,
        seed=19,
    )
    assert abs(exponent) < 0.01


def test_theta_dmft_does_not_report_legacy_power_model_transition():
    _tau, _covariance, sigma_critical = theory_phase_density_autocorr(
        I=1.0,
        sigma=0.0,
        tau_max=0.2,
        dtau=0.1,
        internal_dt=0.1,
        n_time=512,
        n_samples=8,
        max_iter=1,
        phase_model="theta",
    )
    assert np.isnan(sigma_critical)
