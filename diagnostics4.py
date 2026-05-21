"""Debug new theory_phase_autocorr: trace energy function for various g ratios."""
import sys; sys.path.insert(0, 'scripts')
import numpy as np

I, alpha, beta = 1.0, 1.0, 1.0
rho = 1/(2*np.pi)
sigma_c = 2 * np.pi
print(f"sigma_c = {sigma_c:.4f}\n")

g_ratio = 2.0
sigma = g_ratio * sigma_c

print(f"=== Tracing g={g_ratio} (sigma={sigma:.4f}) ===")

def g_fn(u):
    return rho * alpha * np.maximum(I + u, 0.0) ** (1.0 / alpha)

from numpy.polynomial.hermite import hermgauss
n_quad = 24
gh_x, gh_w = hermgauss(n_quad)
gh_w2 = np.outer(gh_w, gh_w)

def mu_g(C_val):
    C_val = float(C_val)
    if C_val <= 0.0:
        return float(g_fn(np.array([0.0]))[0])
    s = np.sqrt(2.0 * C_val)
    return float(np.sum(gh_w * g_fn(s * gh_x)) / np.sqrt(np.pi))

def Q_centered(k, C_11_0):
    C_11_0 = float(C_11_0); k = float(k)
    if C_11_0 <= 0.0: return 0.0
    rho_c = float(np.clip(k / C_11_0, -0.999999, 0.999999))
    scale = np.sqrt(2.0 * C_11_0)
    x = scale * gh_x[:, None]
    y = scale * (rho_c * gh_x[:, None] + np.sqrt(max(1.0 - rho_c**2, 0.0)) * gh_x[None, :])
    Q_raw = float(np.sum(gh_w2 * g_fn(x) * g_fn(y)) / np.pi)
    return Q_raw - mu_g(C_11_0)**2

def C_eq_of(C_11_0):
    return sigma**2 * mu_g(float(C_11_0))**2

def energy(C_11_0_val, n_grid=128):
    C_11_0_val = float(C_11_0_val)
    C_eq_val = C_eq_of(C_11_0_val)
    if C_11_0_val <= C_eq_val + 1e-14:
        return 0.0
    k_grid = np.linspace(C_eq_val, C_11_0_val, n_grid)
    Q_grid = np.array([sigma**2 * Q_centered(k, C_11_0_val) for k in k_grid])
    ie = float(np.trapz(Q_grid, k_grid))
    return ie - (C_11_0_val**2 - C_eq_val**2) / 2.0

# Step 1: find C_eq_rough
lo, hi = 0.0, max(1e-3, sigma**2 * float(g_fn(np.array([0.0]))[0])**2)
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
        if abs(h_mid) < 1e-9 * max(1.0, abs(mid)): C_eq_rough = mid; break
        if h_mid < 0: lo = mid
        else: hi = mid
    C_eq_rough = 0.5*(lo+hi)
print(f"C_eq_rough = {C_eq_rough:.6f}")
print(f"mu_g(C_eq_rough) = {mu_g(C_eq_rough):.6f}")
print(f"sigma^2 * mu_g^2 = {sigma**2 * mu_g(C_eq_rough)**2:.6f}")

# Step 2: trace energy(C_11_0)
print(f"\nC_eq_of(C_11_0) for various C_11_0:")
for C_11_0_test in [C_eq_rough, C_eq_rough+0.01, C_eq_rough+0.1, C_eq_rough+1.0, C_eq_rough+10.0]:
    C_eq_test = C_eq_of(C_11_0_test)
    e = energy(C_11_0_test)
    print(f"  C_11_0={C_11_0_test:.4f}  C_eq={C_eq_test:.4f}  C_tilde_0={C_11_0_test-C_eq_test:.4f}  energy={e:.6f}")
