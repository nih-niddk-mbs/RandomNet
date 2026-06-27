#!/usr/bin/env python
"""
Explore phase model dynamics across ordered, critical, and chaotic regimes
with varying network sizes N.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from python.simulations.random_network import default_results_dir, sim_phase_network, theory_phase_autocorr

# Parameters
I, alpha, beta = 1.0, 1.0, 1.0
sigma_c = 2 * np.pi  # Critical value

# Regimes to explore
sigma_values = [0.4*sigma_c, 0.8*sigma_c, 1.0*sigma_c, 1.5*sigma_c, 2.5*sigma_c]
N_values = [100, 500, 2000]

labels = ["Ordered\n(σ=0.4σ_c)", "Ordered\n(σ=0.8σ_c)", "Critical\n(σ=1.0σ_c)", 
          "Chaotic\n(σ=1.5σ_c)", "Chaotic\n(σ=2.5σ_c)"]

# Create figure with subplots: rows = N values, cols = sigma values
fig, axes = plt.subplots(len(N_values), len(sigma_values), figsize=(16, 10))

for i_N, N in enumerate(N_values):
    for i_sig, sigma in enumerate(sigma_values):
        ax = axes[i_N, i_sig]
        
        # Simulation
        np.random.seed(42)
        print(f"Running sim: N={N}, σ={sigma/(2*np.pi):.2f}×2π ...", end=" ", flush=True)
        tau_sim, C_sim = sim_phase_network(
            N=N, I=I, alpha=alpha, sigma=sigma, beta=beta, 
            T=50, dt=0.01, n_probe=min(20, N)
        )
        print("done")
        
        # Theory
        tau_th, C_th, sigma_c_calc = theory_phase_autocorr(
            I=I, alpha=alpha, sigma=sigma, beta=beta, tau_max=20, dtau=0.01
        )
        
        # Plot
        ax.plot(tau_sim[:1200], C_sim[:1200], 'o-', label='Simulation', 
                markersize=1, alpha=0.5, linewidth=0.8)
        ax.plot(tau_th, C_th, 's--', label='Theory', markersize=2, 
                alpha=0.7, linewidth=1.5, color='red')
        
        # Formatting
        ax.set_ylabel(f'C(τ)', fontsize=9)
        if i_N == len(N_values) - 1:
            ax.set_xlabel('Time τ', fontsize=9)
            ax.set_title(labels[i_sig], fontsize=10)
        else:
            ax.set_title(labels[i_sig], fontsize=10)
        
        if i_sig == 0:
            ax.text(-0.35, 0.5, f'N={N}', transform=ax.transAxes, 
                   fontsize=11, fontweight='bold', va='center')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 12)
        
        # Legend only on first subplot
        if i_N == 0 and i_sig == 0:
            ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
outpath = Path(default_results_dir("diagnostics")) / "chaotic_regimes_N_scan.png"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=120, bbox_inches='tight')
print(f"\n✓ Saved {outpath}")
plt.close()

# Create a second figure: Autocorr decay vs sigma for different N
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left panel: Initial value C(0) vs sigma
ax = axes[0]
for N in N_values:
    C_init_sim = []
    for sigma in sigma_values:
        np.random.seed(42)
        tau_sim, C_sim = sim_phase_network(N=N, I=I, alpha=alpha, sigma=sigma, beta=beta, 
                                  T=50, dt=0.01, n_probe=min(20, N))
        C_init_sim.append(C_sim[0])
    ax.plot(np.array(sigma_values)/(2*np.pi), C_init_sim, 'o-', label=f'N={N}', 
            markersize=6, linewidth=2)

ax.axvline(1.0, color='k', linestyle='--', alpha=0.3, label='σ_c')
ax.set_xlabel('σ / 2π', fontsize=11)
ax.set_ylabel('C(0) (variance)', fontsize=11)
ax.set_title('Initial autocorrelation vs coupling strength', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right panel: Decay timescale vs sigma (simple measure: C(τ=5) / C(0))
ax = axes[1]
for N in N_values:
    decay_ratio = []
    for sigma in sigma_values:
        np.random.seed(42)
        tau_sim, C_sim = sim_phase_network(N=N, I=I, alpha=alpha, sigma=sigma, beta=beta, 
                                  T=50, dt=0.01, n_probe=min(20, N))
        # Decay ratio at tau=5 time units
        idx_5 = int(5.0 / 0.01)
        ratio = C_sim[min(idx_5, len(C_sim)-1)] / max(C_sim[0], 1e-10)
        decay_ratio.append(ratio)
    ax.plot(np.array(sigma_values)/(2*np.pi), decay_ratio, 'o-', label=f'N={N}', 
            markersize=6, linewidth=2)

ax.axvline(1.0, color='k', linestyle='--', alpha=0.3, label='σ_c')
ax.set_xlabel('σ / 2π', fontsize=11)
ax.set_ylabel('C(τ=5) / C(0)', fontsize=11)
ax.set_title('Decay ratio at τ=5 time units', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
outpath = Path(default_results_dir("diagnostics")) / "chaos_summary_statistics.png"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=120, bbox_inches='tight')
print(f"✓ Saved {outpath}")
plt.close()

print("\nExperiments complete!")
