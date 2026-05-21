import sys; sys.path.insert(0, 'scripts')
import numpy as np
from random_network import theory_phase_autocorr, sim_phase_network, _phase_sigma_c, make_weights

sigma_c = _phase_sigma_c(1.0, 1.0)

# 1. Confirm lam=1 gives zero row sums
W = make_weights(64, sigma_c, lam=1)
print(f"lam=1 row sums: mean={W.sum(axis=1).mean():.2e}  max|sum|={np.abs(W.sum(axis=1)).max():.2e}")

# 2. Sim per-neuron C(0) for several g
print()
print("SIM  (per-neuron variance, should be O(sigma^2 * rate) + shot noise)")
for g in [0.5, 1.1, 1.5]:
    sigma = g * sigma_c
    tau, C = sim_phase_network(N=256, I=1, alpha=1, sigma=sigma, beta=1, T=2000, dt=0.02, tau_max=50, n_probe=64)
    print(f"  g={g:.1f}: C(0)={C[0]:.4f}  C(tau_max)={C[-1]:.6f}")

# 3. Theory C(0)
print()
print("THEORY (SCS rate fluctuations only)")
for g in [0.5, 1.1, 1.5]:
    sigma = g * sigma_c
    tau, C, _ = theory_phase_autocorr(I=1, alpha=1, sigma=sigma, beta=1, tau_max=50, dtau=0.1)
    print(f"  g={g:.1f}: C(0)={C[0]:.4f}  C(tau_max)={C[-1]:.6f}")
