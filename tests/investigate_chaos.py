#!/usr/bin/env python
"""
Investigate the chaotic regime dynamics more carefully.
Check phase synchronization, order parameters, and distribution properties.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from python.simulations.random_network import default_results_dir, sim_phase_network

I, alpha, beta = 1.0, 1.0, 1.0
sigma_c = 2 * np.pi

# Focus on chaotic regime with different N
sigma_values = [1.2*sigma_c, 1.5*sigma_c, 2.0*sigma_c]
N_values = [100, 500, 2000]

fig, axes = plt.subplots(len(sigma_values), 4, figsize=(14, 10))

for i_sig, sigma in enumerate(sigma_values):
    for i_N, N in enumerate(N_values):
        print(f"Running: σ={sigma/(2*np.pi):.2f}σ_c, N={N} ...", end=" ", flush=True)
        
        # Run simulation with spike tracking
        np.random.seed(42)
        T_burn = 200  # Longer burn-in to reach attractor
        
        # Manual simulation to get more info
        from python.simulations.random_network import make_weights
        rng = np.random.default_rng(42)
        W = make_weights(N, sigma, lam=1, rng=rng)
        phi = rng.uniform(-np.pi, np.pi, N)
        u = np.zeros(N)
        
        def F(u_):
            return alpha * np.maximum(I + u_, 0.0) ** (1.0 / alpha)
        
        # Burn-in
        dt = 0.01
        nb = int(T_burn / dt)
        spike_count = np.zeros(N)
        for _ in range(nb):
            phi += F(u) * dt
            spikes = phi >= np.pi
            spike_count += spikes.astype(float)
            phi[spikes] -= 2.0 * np.pi
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                drive = W @ spikes.astype(float)
            u += dt * (-beta * u + beta * drive)
            if not np.all(np.isfinite(u)):
                u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Recording phase
        T = 100
        nt = int(T / dt)
        phi_traj = np.zeros((nt, min(10, N)))  # Track a few neurons
        u_traj = np.zeros((nt, min(10, N)))
        spike_rate = np.zeros(nt)
        
        for t in range(nt):
            phi += F(u) * dt
            spikes = phi >= np.pi
            spike_count += spikes.astype(float)
            phi[spikes] -= 2.0 * np.pi
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                drive = W @ spikes.astype(float)
            u += dt * (-beta * u + beta * drive)
            if not np.all(np.isfinite(u)):
                u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
            
            phi_traj[t] = phi[:min(10, N)]
            u_traj[t] = u[:min(10, N)]
            spike_rate[t] = np.mean(spikes.astype(float)) / dt
        
        print("done")
        
        # Compute statistics
        mean_spike_rate = np.mean(spike_rate)
        mean_u = np.mean(u_traj)
        std_u = np.std(u_traj)
        mean_phi = np.mean(np.cos(phi_traj))  # Coherence
        
        # Histograms of u and spike distribution
        ax_row = axes[i_sig]
        
        # 1. Phase trajectory of first neuron
        tau = np.arange(nt) * dt
        ax = ax_row[0]
        ax.plot(tau[::10], np.mod(phi_traj[::10, 0] + np.pi, 2*np.pi) - np.pi, '.', markersize=2)
        ax.set_ylabel('φ (neuron 0)', fontsize=9)
        ax.set_ylim([-np.pi, np.pi])
        if i_sig == len(sigma_values) - 1:
            ax.set_xlabel('Time (s)', fontsize=9)
        
        # 2. Input u trajectory
        ax = ax_row[1]
        for i in range(min(5, min(10, N))):
            ax.plot(tau[::20], u_traj[::20, i], '-', linewidth=0.5, alpha=0.6)
        ax.set_ylabel('u_i (inputs)', fontsize=9)
        if i_sig == len(sigma_values) - 1:
            ax.set_xlabel('Time (s)', fontsize=9)
        
        # 3. Population spike rate
        ax = ax_row[2]
        ax.plot(tau, spike_rate, linewidth=1, alpha=0.7)
        ax.set_ylabel('Spike rate (Hz)', fontsize=9)
        if i_sig == len(sigma_values) - 1:
            ax.set_xlabel('Time (s)', fontsize=9)
        
        # 4. Statistics text
        ax = ax_row[3]
        ax.axis('off')
        text_str = f"σ={sigma/(2*np.pi):.2f}σ_c, N={N}\n" \
                   f"Mean u: {mean_u:.3f}\n" \
                   f"Std u: {std_u:.3f}\n" \
                   f"Mean rate: {mean_spike_rate:.2f} Hz\n" \
                   f"Coherence: {mean_phi:.3f}"
        ax.text(0.1, 0.5, text_str, fontsize=10, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))

plt.tight_layout()
outpath = Path(default_results_dir("diagnostics")) / "chaotic_dynamics_detail.png"
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=120, bbox_inches='tight')
print(f"\n✓ Saved {outpath}")
