import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rn_binary import (  # noqa: E402
    _conditional_binary_spectrum,
    _periodic_binary_probability,
    sigmoid_rate,
    sigmoid_tangent_parameters,
    theory_binary_autocorr,
    theory_binary_sigmoid_dmft,
)


def test_conditional_master_equation_probability_is_periodic():
    rates = np.array(
        [[0.2, 0.4, 0.8, 0.3], [1.1, 0.7, 0.1, 0.5]], dtype=float
    )
    mu = 0.9
    dt = 0.1
    probability, decay = _periodic_binary_probability(rates, mu, dt)
    p_on = rates / (rates + mu) * (1.0 - decay)

    propagated = decay * probability + p_on
    assert np.allclose(propagated[:, :-1], probability[:, 1:])
    assert np.allclose(propagated[:, -1], probability[:, 0])


def test_conditional_covariance_matches_constant_rate_process():
    rate = 0.5
    mu = 1.0
    dt = 0.05
    rates = np.full((3, 512), rate)
    _spectrum, probability, intrinsic = _conditional_binary_spectrum(
        rates, mu, dt
    )
    mean = rate / (rate + mu)
    lag = np.arange(40)
    expected = mean * (1.0 - mean) * np.exp(-(rate + mu) * dt * lag)

    assert np.allclose(probability, mean)
    assert np.allclose(intrinsic[:40], expected)


def test_sigmoid_rate_is_positive_bounded_and_has_requested_tangent():
    rate_max = 1.0
    theta = 0.0
    delta = 0.25
    values = sigmoid_rate(np.array([-100.0, 0.0, 100.0]), rate_max, theta, delta)
    f0, f1 = sigmoid_tangent_parameters(rate_max, theta, delta)

    assert np.all(values >= 0.0)
    assert np.all(values <= rate_max)
    assert values[1] == pytest.approx(f0)
    assert f1 == pytest.approx(1.0)


def test_sigmoid_dmft_is_exact_in_uncoupled_limit():
    tau, Cnn, Cuu, diagnostics = theory_binary_sigmoid_dmft(
        sigma=0.0,
        mu=1.0,
        rate_max=1.0,
        theta=0.0,
        delta=0.25,
        tau_max=1.0,
        dtau=0.05,
        return_diagnostics=True,
    )
    rate0 = 0.5
    gamma0 = rate0 + 1.0
    mean0 = rate0 / gamma0
    expected = mean0 * (1.0 - mean0) * np.exp(-gamma0 * tau)

    assert diagnostics["converged"]
    assert np.allclose(Cnn, expected)
    assert np.all(Cuu == 0.0)


def test_sigmoid_dmft_returns_positive_semidefinite_spectra():
    _tau, Cnn, Cuu, diagnostics = theory_binary_sigmoid_dmft(
        sigma=1.0,
        tau_max=0.5,
        dtau=0.05,
        internal_dt=0.05,
        n_time=512,
        n_samples=8,
        max_iter=4,
        return_diagnostics=True,
        seed=17,
    )

    assert np.all(np.isfinite(Cnn))
    assert np.all(np.isfinite(Cuu))
    assert np.min(diagnostics["state_spectrum"]) >= 0.0
    assert np.min(diagnostics["drive_spectrum"]) >= 0.0
    assert diagnostics["conditional_method"] == "exact_master_equation"


def test_affine_residues_define_a_valid_subcritical_covariance():
    tau, Cnn, _Cuu, gain = theory_binary_autocorr(
        sigma=2.0, beta=1.0, mu=1.0, f0=0.5, f1=1.0,
        tau_max=5.0, dtau=0.01,
    )

    assert gain < 1.0
    assert np.all(np.isfinite(Cnn))
    assert np.max(Cnn) == pytest.approx(Cnn[0])
    assert (Cnn[1] - Cnn[0]) / (tau[1] - tau[0]) < 0.0


def test_affine_theory_rejects_nonstationary_supercritical_branch():
    _tau, Cnn, Cuu, gain = theory_binary_autocorr(
        sigma=3.0, beta=1.0, mu=1.0, f0=0.5, f1=1.0,
        tau_max=1.0, dtau=0.05,
    )

    assert gain > 1.0
    assert np.all(np.isnan(Cnn))
    assert np.all(np.isnan(Cuu))
