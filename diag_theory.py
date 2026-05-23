import sys; sys.path.insert(0, 'scripts')
import numpy as np
import matplotlib; matplotlib.use('Agg')
from random_network import theory_phase_autocorr, shot_noise_correction

rho = 1.0 / (2.0 * np.pi)
I, alpha, beta = 1.0, 1.0, 1.0
sigma_c = 2 * np.pi

gh_x, gh_w = np.polynomial.hermite.hermgauss(24)
gh_w2 = np.outer(gh_w, gh_w)

def g(u):
    return rho * alpha * np.maximum(I + u, 0.0) ** (1.0 / alpha)

def mu_g(C_val):
    C_val = max(float(C_val), 1e-14)
    s = np.sqrt(2.0 * C_val)
    return float(np.sum(gh_w * g(s * gh_x)) / np.sqrt(np.pi))

def Q_c(C_tau, C0_val, sigma):
    if C0_val <= 0:
        return 0.0
    rho_c = float(np.clip(C_tau / C0_val, -0.999999, 0.999999))
    scale = np.sqrt(2.0 * C0_val)
    x = scale * gh_x[:, None]
    y = scale * (rho_c * gh_x[:, None] + np.sqrt(1 - rho_c**2) * gh_x[None, :])
    Q_raw = float(np.sum(gh_w2 * g(x) * g(y)) / np.pi)
    mu = mu_g(C0_val)
    return sigma**2 * (Q_raw - mu**2)

def H(C0_val, sigma, n_grid=512):
    if C0_val <= 0:
        return float('nan')
    C_grid = np.linspace(0, C0_val, n_grid)
    Q_grid = np.array([Q_c(c, C0_val, sigma) for c in C_grid])
    integral = np.trapezoid(Q_grid, C_grid)
    return C0_val**2 - 2.0 * integral

print("=== H(C0) scan ===")
C0_test = np.logspace(-2, 3, 80)
for g_val in [1.1, 1.3, 1.5]:
    sigma = g_val * sigma_c
    Hvals = np.array([H(c, sigma) for c in C0_test])
    crossings = []
    for i in range(len(C0_test) - 1):
        if np.isfinite(Hvals[i]) and np.isfinite(Hvals[i+1]):
            if Hvals[i] < 0 and Hvals[i+1] > 0:
                crossings.append(f'neg->pos at C0~{C0_test[i]:.3f}')
            elif Hvals[i] > 0 and Hvals[i+1] < 0:
                crossings.append(f'pos->neg at C0~{C0_test[i]:.3f}')
    print(f"g={g_val}: H(lo={C0_test[0]:.3f})={Hvals[0]:.4f}  H(hi={C0_test[-1]:.1f})={Hvals[-1]:.4f}  crossings: {crossings}")

print()
print("=== Theory output ===")
for g_val in [1.1, 1.3, 1.5]:
    sigma = g_val * sigma_c
    tau, C, sc = theory_phase_autocorr(I=I, alpha=alpha, sigma=sigma, beta=beta, tau_max=10, dtau=0.1)
    C_shot0 = shot_noise_correction(np.array([0.0]), sigma, beta, I, alpha, C[0])[0]
    C_scs0 = C[0] - C_shot0
    print(f"g={g_val}: C(0)={C[0]:.4f}  C_SCS(0)={C_scs0:.6f}  C_shot(0)={C_shot0:.4f}")
    print(f"  C_norm: " + "  ".join(f"tau={tau[i]:.1f}: {C[i]/C[0]:.4f}" for i in [0, 5, 10, 20, 50, 90, -1]))
    print()
