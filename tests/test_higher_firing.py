#!/usr/bin/env python
"""Test with increased firing rates to see if amplitude mismatch is resolved."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from python.simulations.random_network import default_results_dir, sim_phase_network, theory_phase_autocorr

I_values = [1.0, 2.0, 5.0, 10.0]
sigma = 0.8 * 2 * np.pi
N = 100
T = 30

print("Testing different baseline inputs I to increase firing rates:\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, I in enumerate(I_values):
    print(f"I = {I}:")
    
    # Simulation
    np.random.seed(42)
    tau_sim, C_sim = sim_phase_network(
        N=N, I=I, alpha=1.0, sigma=sigma, beta=1.0,
        T=T, dt=0.01, n_probe=1
    )
    
    # Theory
    tau_th, C_th, sigma_c = theory_phase_autocorr(
        I=I, alpha=1.0, sigma=sigma, beta=1.0,
        tau_max=T, dtau=0.01
    )
    
    print(f"  Sim:    C(0) = {C_sim[0]:.6f}")
    print(f"  Theory: C(0) = {C_th[0]:.6f}")
    print(f"  Ratio:  {C_sim[0]/max(C_th[0], 1e-10):.2f}x")
    print()
    
    # Plot - use common time range
    ax = axes[idx]
    t_max = min(tau_sim[-1], tau_th[-1])
    idx_sim = np.where(tau_sim <= t_max)[0]
    idx_th = np.where(tau_th <= t_max)[0]
    
    ax.plot(tau_sim[idx_sim], C_sim[idx_sim], 'b-', linewidth=2, label='Simulation')
    
    # Rescale theory to match initial value for shape comparison
    if C_th[0] > 0:
        C_th_scaled = C_th * (C_sim[0] / C_th[0])
        ax.plot(tau_th[idx_th], C_th_scaled[idx_th], 'r--', linewidth=2, label='Theory (rescaled)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'I = {I}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5])

plt.tight_layout()
outpath = Path(default_results_dir("diagnostics")) / "test_higher_firing_rates.png"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=150)
print(f"✓ Saved {outpath}")
