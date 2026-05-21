"""Diagnose energy_endpoint vs sigma, and compare theory C(0) to sim C(0)."""
import sys; sys.path.insert(0, 'scripts')
import numpy as np
from numpy.polynomial.hermite import hermgauss

I, alpha = 1.0, 1.0
rho = 1 / (2 * np.pi)
sigma_c = 2 * np.pi
n_quad = 24
gh_x, gh_w = hermgauss(n_quad)
gh_w2 = np.outer(gh_w, gh_w)

def g(u):
    return rho * alpha * np.maximum(I + u, 0.0) ** (1.0 / alpha)

def mu_g(C0):
    C0 = float(C0)
    if C0 <= 0: return float(g(0.0))
    s = np.sqrt(2.0 * C0)
    return float(np.sum(gh_w * g(s * gh_x)) / np.sqrt(np.pi))

def Q_centered(c, C0, sigma):
    C0 = float(C0); c = float(c)
    if C0 <= 0: return 0.0
    rho_c = float(np.clip(c / C0, -0.999999, 0.999999))
    scale = np.sqrt(2.0 * C0)
    x = scale * gh_x[:, None]
    y = scale * (rho_c * gh_x[:, None] + np.sqrt(max(1 - rho_c**2, 0)) * gh_x[None, :])
    Q_raw = float(np.sum(gh_w2 * g(x) * g(y)) / np.pi)
    return Q_raw - mu_g(C0)**2

def energy(C0, sigma, n_grid=256):
    C0 = float(C0)
    if C0 <= 0: return 0.0
    c_grid = np.linspace(0.0, C0, n_grid)
    Q_grid = np.array([sigma**2 * Q_centered(c, C0, sigma) for c in c_grid])
    intQ = float(np.trapezoid(Q_grid, c_grid))
    return C0**2 - 2.0 * intQ

for g_ratio in [1.1, 1.5, 2.0]:
    sigma = g_ratio * sigma_c
    print(f"\ng/gc={g_ratio:.1f}  sigma={sigma:.4f}")
    print(f"  {'C0':>8}  {'H(C0)':>12}  {'mu_g':>8}")
    # Fine scan from 0.01 to 40
    c0_vals = np.concatenate([np.linspace(0.01, 1.0, 30), np.linspace(1.0, 20.0, 30), np.linspace(20.0, 60.0, 15)])
    prev_e = None
    for c0 in c0_vals:
        e = energy(c0, sigma)
        sign_change = ""
        if prev_e is not None and prev_e * e < 0:
            sign_change = " *** SIGN CHANGE ***"
        print(f"  {c0:8.3f}  {e:12.4f}  {mu_g(c0):8.5f}{sign_change}")
        prev_e = e
