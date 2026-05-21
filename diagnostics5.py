"""Debug: trace energy function for various g ratios in the new theory."""
import sys; sys.path.insert(0, 'scripts')
import numpy as np

I, alpha = 1.0, 1.0
rho = 1/(2*np.pi)
sigma_c = 2 * np.pi
print(f"sigma_c = {sigma_c:.4f}\n")

from numpy.polynomial.hermite import hermgauss
n_quad = 24
gh_x, gh_w = hermgauss(n_quad)
gh_w2 = np.outer(gh_w, gh_w)

def g_fn(u):
    return rho * alpha * np.maximum(I + u, 0.0) ** (1.0 / alpha)

def mu_g(C_val, sigma=None):
    C_val = float(C_val)
    if C_val <= 0.0:
        return float(g_fn(0.0))
    s = np.sqrt(2.0 * C_val)
    return float(np.sum(gh_w * g_fn(s * gh_x)) / np.sqrt(np.pi))

def Q_centered(k, C_11_0):
    C_11_0 = float(C_11_0); k = float(k)
    if C_11_0 <= 0.0: return 0.0
    rho_c = float(np.clip(k / C_11_0, -0.999999, 0.999999))
    scale = np.sqrt(2.0 * C_11_0)
    xi = scale * gh_x[:, None]
    xj = scale * (rho_c * gh_x[:, None] + np.sqrt(max(1.0 - rho_c**2, 0.0)) * gh_x[None, :])
    Q_raw = float(np.sum(gh_w2 * g_fn(xi) * g_fn(xj)) / np.pi)
    return Q_raw - mu_g(C_11_0)**2

def C_eq_of(C_11_0, sigma):
    return sigma**2 * mu_g(C_11_0)**2

def energy(C_11_0_val, sigma, n_grid=128):
    C_11_0_val = float(C_11_0_val)
    C_eq_val = C_eq_of(C_11_0_val, sigma)
    if C_11_0_val <= C_eq_val + 1e-14:
        return 0.0
    k_grid = np.linspace(C_eq_val, C_11_0_val, n_grid)
    Q_grid = np.array([sigma**2 * Q_centered(k, C_11_0_val) for k in k_grid])
    ie = float(np.trapezoid(Q_grid, k_grid))
    return ie - (C_11_0_val**2 - C_eq_val**2) / 2.0

for g_ratio in [1.1, 1.5, 2.0, 3.0]:
    sigma = g_ratio * sigma_c
    print(f"\n=== g/gc={g_ratio:.1f}, sigma={sigma:.4f} ===")

    # Find C_eq_rough
    lo = 0.0
    hi = max(1e-3, sigma**2 * float(g_fn(0.0))**2)
    h_hi = hi - sigma**2 * mu_g(hi)**2
    for _ in range(64):
        if h_hi > 0: break
        lo = hi; hi *= 4.0
        if hi > 1e8: break
        h_hi = hi - sigma**2 * mu_g(hi)**2
    C_eq_rough = 0.0
    if h_hi > 0:
        for _ in range(80):
            mid = 0.5*(lo+hi)
            h_mid = mid - sigma**2 * mu_g(mid)**2
            if abs(h_mid) < 1e-9 * max(1.0, abs(mid)):
                C_eq_rough = mid; break
            if h_mid < 0: lo = mid
            else: hi = mid
        C_eq_rough = 0.5*(lo+hi)
    print(f"  C_eq_rough = {C_eq_rough:.6f}")

    # Scan energy over C_11_0
    test_vals = [C_eq_rough + d for d in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]]
    print(f"  {'C_11_0':>10}  {'C_eq':>10}  {'C_tilde_0':>12}  {'energy':>12}")
    for C_11_0_test in test_vals:
        C_eq_test = C_eq_of(C_11_0_test, sigma)
        e = energy(C_11_0_test, sigma)
        print(f"  {C_11_0_test:10.4f}  {C_eq_test:10.4f}  {C_11_0_test-C_eq_test:12.4f}  {e:12.6f}")
