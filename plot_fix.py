#!/usr/bin/env python
"""Quick plots comparing sim vs theory for phase model with centered gain fix."""
import numpy as np
import matplotlib.pyplot as plt
from python.simulations.random_network import (
    sim_phase_network, theory_phase_autocorr, autocorr
)

# Two cases: below and near critical
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, sigma in enumerate([0.5*2*np.pi, 1.0*2*np.pi]):
    ax = axes[idx]
    
    # Simulation
    np.random.seed(42)
    C_sim = sim_phase_network(N=500, I=1.0, alpha=1.0, sigma=sigma, beta=1.0, T=30, dt=0.01, n_probe=1)
    C_sim = C_sim[0] if isinstance(C_sim, (tuple, list)) else C_sim
    C_sim = autocorr(C_sim, max_lag=1500)
    tau_sim = np.arange(len(C_sim)) * 0.01
    
    # Theory
    tau_th, C_th, sigma_c = theory_phase_autocorr(I=1.0, alpha=1.0, sigma=sigma, beta=1.0, tau_max=15, dtau=0.01)
    
    # Normalize to same initial amplitude for comparison of decay shape
    scale_factor = C_sim[0] / C_th[0]
    C_th_scaled = C_th * scale_factor
    
    ax.plot(tau_sim[:800], C_sim[:800], 'o-', label=f'Simulation (C₀={C_sim[0]:.1f})', markersize=2, alpha=0.6, linewidth=1)
    ax.plot(tau_th, C_th_scaled, 's--', label=f'Theory (scaled ×{scale_factor:.1f})', markersize=3, alpha=0.8, linewidth=2, color='red')
    ax.set_xlabel('Time τ', fontsize=11)
    ax.set_ylabel('Autocorr C(τ)', fontsize=11)
    sig_label = 0.5 if idx == 0 else 1.0
    ax.set_title(f'σ={sig_label}×2π, σ_c={sigma_c/(2*np.pi):.2f}×2π', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('python/plots/phase_sim_vs_theory.png', dpi=120, bbox_inches='tight')
print("✓ Saved phase_sim_vs_theory.png")
