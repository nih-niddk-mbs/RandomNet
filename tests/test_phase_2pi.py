import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rn_phase_2pi import (
    _filter_two_time,
    _retarded_time_response,
    free_phase_flux_covariance,
    fixed_q_linear_flux_kernel,
    propagator_block,
    solve_stationary_phase_gaussian_advection,
    solve_phase_fixed_q_gaussian,
    solve_phase_fixed_q_propagators,
    solve_uniform_phase_fixed_q_gaussian,
    solve_uniform_phase_gaussian_2pi,
    uniform_phase_gaussian_2pi_stability,
    uniform_phase_fixed_q_stability,
    uniform_phase_covariance,
)


def test_structured_two_time_filter_matches_dense_response_product():
    rng = np.random.default_rng(91)
    kernel = rng.normal(size=(9, 9))
    response = _retarded_time_response(9, 0.07, 1.3)
    expected = response @ kernel @ response.T
    assert np.allclose(_filter_two_time(kernel, 0.07, 1.3), expected)


def test_fixed_q_propagator_inverse_and_causality():
    n_time = 5
    n_phase = 6
    solution = solve_phase_fixed_q_propagators(
        np.eye(n_time),
        beta=0.8,
        sigma=0.7,
        velocity=1.1,
        dt=0.15,
        n_phase=n_phase,
    )

    assert solution.left_residual < 1e-11
    assert solution.right_residual < 1e-11
    C12 = propagator_block(solution, "u", "u_response")
    C21 = propagator_block(solution, "u_response", "u")
    assert np.allclose(C12, np.tril(C12))
    assert np.allclose(C21, np.triu(C21))
    assert np.allclose(
        propagator_block(solution, "u_response", "u_response"), 0.0
    )
    assert np.allclose(
        propagator_block(
            solution, "density_response", "density_response"
        ),
        0.0,
    )


def test_fixed_q_causal_solution_implies_the_2pi_drive_filter_equation():
    rng = np.random.default_rng(17)
    factors = rng.normal(size=(6, 4))
    Q = factors @ factors.T
    beta = 0.9
    sigma = 0.7
    solution = solve_phase_fixed_q_propagators(
        Q,
        beta=beta,
        sigma=sigma,
        velocity=1.0,
        dt=0.12,
        n_phase=5,
    )
    C11 = propagator_block(solution, "u", "u")
    retarded = propagator_block(solution, "u", "u_response")
    advanced = propagator_block(solution, "u_response", "u")
    expected = beta**2 * sigma**2 * retarded @ Q @ advanced

    assert np.allclose(C11, expected, rtol=1e-11, atol=1e-11)


def test_uniform_initial_phase_covariance_is_propagated_exactly():
    n_time = 4
    n_phase = 8
    expected = uniform_phase_covariance(n_phase)
    solution = solve_phase_fixed_q_propagators(
        np.zeros((n_time, n_time)),
        sigma=0.0,
        velocity=1.0,
        dt=0.1,
        n_phase=n_phase,
        initial_phase_covariance=expected,
    )
    C33 = propagator_block(solution, "density", "density")
    assert np.allclose(C33[:n_phase, :n_phase], expected)
    assert np.max(np.abs(C33 @ np.tile(np.ones(n_phase), n_time))) < 1e-10
    assert np.allclose(
        propagator_block(solution, "u", "density"), 0.0
    )


def test_fixed_q_wick_kernel_is_symmetric_with_event_diagonal_replacement():
    n_time = 4
    dt = 0.2
    solution = solve_phase_fixed_q_propagators(
        np.zeros((n_time, n_time)),
        sigma=0.0,
        velocity=1.0,
        dt=dt,
        n_phase=8,
    )
    kernel, rate = fixed_q_linear_flux_kernel(solution, F0=1.0, F1=1.0)
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(np.diag(kernel), rate / dt - rate**2)


def test_small_grid_fixed_q_system_converges_when_uncoupled():
    result = solve_phase_fixed_q_gaussian(
        n_time=4,
        n_phase=6,
        dt=0.2,
        sigma=0.0,
        max_iter=30,
        mixing=0.5,
        tolerance=1e-8,
    )
    assert result.converged
    assert result.residual_history[-1] < 1e-8
    assert result.propagators.left_residual < 1e-11
    assert result.propagators.right_residual < 1e-11


def test_small_grid_fixed_q_system_converges_when_coupled():
    result = solve_phase_fixed_q_gaussian(
        n_time=6,
        n_phase=6,
        dt=0.2,
        sigma=1.0,
        max_iter=80,
        mixing=0.2,
        tolerance=1e-6,
    )
    C11 = propagator_block(result.propagators, "u", "u")
    assert result.converged
    assert result.residual_history[-1] < 1e-6
    assert np.all(np.diag(C11) >= 0.0)


def test_uniform_reduction_matches_dense_fixed_q_system():
    parameters = dict(
        n_time=5,
        n_phase=6,
        dt=0.2,
        beta=0.9,
        sigma=0.8,
        F0=1.0,
        F1=1.0,
        max_iter=120,
        mixing=0.2,
        tolerance=1e-7,
        transport_scheme="upwind",
    )
    dense = solve_phase_fixed_q_gaussian(**parameters)
    reduced = solve_uniform_phase_fixed_q_gaussian(
        **parameters, threshold_regularization="phase_grid"
    )
    dense_C11 = propagator_block(dense.propagators, "u", "u")
    n_phase = parameters["n_phase"]
    threshold = np.arange(parameters["n_time"]) * n_phase + n_phase - 1
    dense_C33 = propagator_block(
        dense.propagators, "density", "density"
    )[np.ix_(threshold, threshold)]

    assert dense.converged and reduced.converged
    assert np.allclose(reduced.C11, dense_C11, atol=2e-7)
    assert np.allclose(reduced.C33_threshold, dense_C33, atol=1e-12)
    assert np.all(reduced.C13 == 0.0)
    assert np.all(reduced.C31 == 0.0)


def test_spectral_transport_preserves_uncoupled_return_amplitude():
    result = solve_uniform_phase_fixed_q_gaussian(
        n_time=80,
        n_phase=25,
        dt=2.0 * np.pi / 72.0,
        sigma=0.0,
        max_iter=2,
        transport_scheme="spectral",
    )
    period_bin = 72
    assert np.isclose(
        result.C33_threshold[period_bin, 0],
        result.C33_threshold[0, 0],
        rtol=1e-11,
        atol=1e-11,
    )


def test_free_flux_regularization_separates_adjacent_time_bins():
    dt = 2.0 * np.pi / 64.0
    kernel = free_phase_flux_covariance(66, dt, velocity=1.0)
    rate = 1.0 / (2.0 * np.pi)
    assert np.isclose(kernel[0, 1], -rate**2, atol=1e-12)
    assert np.isclose(kernel[0, 64], kernel[0, 0], atol=1e-12)


def test_uniform_gaussian_stability_returns_a_converged_positive_threshold():
    stability = uniform_phase_fixed_q_stability(
        n_time=48,
        n_phase=25,
        dt=2.0 * np.pi / 32.0,
        max_iter=300,
        tolerance=1e-8,
    )
    assert stability.residual < 1e-3
    assert stability.unit_sigma_eigenvalue > 0.0
    assert np.isfinite(stability.critical_sigma)
    assert stability.critical_sigma > 0.0


def test_gaussian_2pi_recovers_uncoupled_matched_bin_flux_covariance():
    n_phase = 33
    dt = 2.0 * np.pi / n_phase
    result = solve_uniform_phase_gaussian_2pi(
        n_time=40,
        n_phase=n_phase,
        dt=dt,
        sigma=0.0,
        tolerance=1e-10,
    )
    expected = free_phase_flux_covariance(40, dt)
    assert result.stable and result.converged and result.physical
    assert np.all(result.C11 == 0.0)
    assert np.allclose(result.flux_kernel, expected, atol=1e-11)


def test_gaussian_advection_recovers_uncoupled_phase_returns():
    result = solve_stationary_phase_gaussian_advection(
        n_periods=4,
        n_phase=33,
        sigma=0.0,
        phase_bin_width=0.25,
    )
    period_bin = 33
    assert result.converged and result.physical
    assert np.all(result.C11_lag == 0.0)
    assert result.C33_binned_lag[period_bin] == pytest.approx(
        result.C33_binned_lag[0], rel=1e-12
    )


def test_gaussian_2pi_is_physical_below_and_rejected_above_instability():
    parameters = dict(
        n_time=48,
        n_phase=33,
        dt=2.0 * np.pi / 33.0,
        tolerance=1e-8,
        max_iter=500,
    )
    stability = uniform_phase_gaussian_2pi_stability(**parameters)
    below = solve_uniform_phase_gaussian_2pi(
        sigma=0.5 * stability.critical_sigma,
        **parameters,
    )
    above = solve_uniform_phase_gaussian_2pi(
        sigma=1.1 * stability.critical_sigma,
        **parameters,
    )
    assert stability.residual < 1e-6
    assert below.stable and below.converged and below.physical
    assert below.min_covariance_eigenvalue >= -1e-10
    assert below.min_flux_eigenvalue >= -1e-10
    assert not above.stable
    assert not above.converged
    assert not above.physical
