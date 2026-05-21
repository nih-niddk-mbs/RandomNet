"""Diagnostic for alpha=2 (sublinear gain): theory vs sim."""
import sys; sys.path.insert(0, 'scripts')
import numpy as np
from random_network import theory_phase_autocorr, sim_phase_network, _phase_sigma_c

I, alpha, beta = 1.0, 2.0, 1.0
sigma_c = _phase_sigma_c(I, alpha)
print(f"alpha={alpha}, sigma_c = {sigma_c:.4f}\n")

print("=== THEORY ===")
print(f"{'g/gc':>6}  {'C_eq':>8}  {'C_tilde(0)':>12}  {'C11(0)':>8}  {'C11(inf)':>10}")
print("-" * 55)
for g_ratio in [1.1, 1.5, 2.0, 3.0]:
    sigma = g_ratio * sigma_c
    tau, C11, _ = theory_phase_autocorr(
        I=I, alpha=alpha, sigma=sigma, beta=beta, tau_max=80, dtau=0.05)
    C_eq = float(C11[-1])
    C0 = float(C11[0])
    C_tilde0 = C0 - C_eq
    print(f"{g_ratio:6.2f}  {C_eq:8.4f}  {C_tilde0:12.4f}  {C0:8.4f}  {C_eq:10.4f}")

print()
print("=== SIMULATION (lam=1, row-sum corrected) ===")
print(f"{'g/gc':>6}  {'C_sim(0)':>10}  {'C_sim(inf)':>12}  {'C_sim inf/0':>12}")
print("-" * 50)
for g_ratio in [1.1, 1.5, 2.0, 3.0]:
    sigma = g_ratio * sigma_c
    tau_s, C_s = sim_phase_network(
        N=512, I=I, alpha=alpha, sigma=sigma, beta=beta,
        T=3000, dt=0.02, tau_max=80, n_probe=512)
    c0 = float(C_s[0]); clast = float(C_s[-1])
    print(f"{g_ratio:6.2f}  {c0:10.4f}  {clast:12.6f}  {clast/max(c0,1e-10):12.6f}")

print()
print("=== THEORY: C_tilde (fluctuating part) vs sim ===")
print("(Theory C_tilde = C11 - C_eq, should match sim since sim C_eq=0)")
for g_ratio in [1.5, 2.0, 3.0]:
    sigma = g_ratio * sigma_c
    tau_th, C11, _ = theory_phase_autocorr(
        I=I, alpha=alpha, sigma=sigma, beta=beta, tau_max=80, dtau=0.05)
    C_eq = float(C11[-1])
    C_tilde = C11 - C_eq

    tau_s, C_s = sim_phase_network(
        N=512, I=I, alpha=alpha, sigma=sigma, beta=beta,
        T=3000, dt=0.02, tau_max=80, n_probe=512)

    ct0 = float(C_tilde[0])
    cs0 = float(C_s[0])
    ct5 = float(np.interp(5.0, tau_th, C_tilde))
    cs5 = float(np.interp(5.0, tau_s, C_s))
    ct20 = float(np.interp(20.0, tau_th, C_tilde))
    cs20 = float(np.interp(20.0, tau_s, C_s))
    print(f"\ng={g_ratio:.1f}  (C_eq={C_eq:.3f})")
    print(f"  C_tilde(0)={ct0:.4f}  C_sim(0)={cs0:.4f}")
    print(f"  C_tilde(5)/C_tilde(0)={ct5/max(ct0,1e-10):.4f}  C_sim(5)/C_sim(0)={cs5/max(cs0,1e-10):.4f}")
    print(f"  C_tilde(20)/C_tilde(0)={ct20/max(ct0,1e-10):.4f}  C_sim(20)/C_sim(0)={cs20/max(cs0,1e-10):.4f}")
