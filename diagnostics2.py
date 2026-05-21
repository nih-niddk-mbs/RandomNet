"""Diagnostic: compare theory C_eq / C_tilde vs simulation long-lag limit."""
import sys; sys.path.insert(0, 'scripts')
import numpy as np
from scipy.special import ndtr
from random_network import theory_phase_autocorr, sim_phase_network

I, alpha, beta = 1.0, 1.0, 1.0
sigma_c = 2 * np.pi

print(f"sigma_c = {sigma_c:.4f}\n")

print("=== THEORY: C_eq, C_tilde(0), sigma_c_eff/sigma_c ===")
print(f"{'g/gc':>6}  {'C_eq':>8}  {'C_tilde(0)':>12}  {'C11(0)':>8}  {'sc_eff/sc':>10}")
print("-" * 55)
for g_ratio in [0.9, 1.1, 1.5, 2.0, 3.0]:
    sigma = g_ratio * sigma_c
    tau, C11, _ = theory_phase_autocorr(
        I=I, alpha=alpha, sigma=sigma, beta=beta, tau_max=100, dtau=0.05)
    C_eq = float(C11[-1])
    C0 = float(C11[0])
    C_tilde0 = C0 - C_eq
    if C_eq > 1e-10:
        sc_eff_ratio = 1.0 / ndtr(I / np.sqrt(C_eq))
    else:
        sc_eff_ratio = 1.0
    print(f"{g_ratio:6.2f}  {C_eq:8.4f}  {C_tilde0:12.6f}  {C0:8.4f}  {sc_eff_ratio:10.3f}")

print()
print("=== SIMULATION (lam=1, row-sum corrected): long-lag C_sim ===")
print(f"{'g/gc':>6}  {'C_sim(0)':>10}  {'C_sim(inf)':>12}  {'ratio':>8}")
print("-" * 45)
for g_ratio in [1.1, 1.5, 2.0]:
    sigma = g_ratio * sigma_c
    tau_s, C_s = sim_phase_network(
        N=512, I=I, alpha=alpha, sigma=sigma, beta=beta,
        T=3000, dt=0.02, tau_max=80, n_probe=512)
    c0 = float(C_s[0]); clast = float(C_s[-1])
    print(f"{g_ratio:6.2f}  {c0:10.4f}  {clast:12.6f}  {clast/max(c0,1e-10):8.5f}")
