import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rn_core import make_weights  # noqa: E402


def test_weights_are_row_sum_corrected():
    rng = np.random.default_rng(123)
    W = make_weights(64, 2.5, lam=1, rng=rng)

    assert np.allclose(np.diag(W), 0.0)
    assert np.allclose(W.sum(axis=1), 0.0, atol=1e-12)
def test_plain_weights_are_not_forced_to_zero_row_sum():
    rng = np.random.default_rng(123)
    W = make_weights(64, 2.5, lam=0, rng=rng)

    assert np.allclose(np.diag(W), 0.0)
    assert not np.allclose(W.sum(axis=1), 0.0, atol=1e-12)
