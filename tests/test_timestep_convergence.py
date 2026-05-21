#!/usr/bin/env python
"""Test with longer simulations and larger timestep."""
import numpy as np
import matplotlib.pyplot as plt
from python.simulations.random_network import sim_phase_network, theory_phase_autocorr

dt_values = [0.01, 0.02, 0.05]
I = 1.0
sigma = 0.8 * 2 * np.pi
N = 500

print("Testing longer runs with different timesteps:\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, dt in enumerate(dt_values):
    T = min(100, int(300 / dt) * dt)  # Scale T inversely with dt to stay ~300 steps
    
    print(f"dt = {dt}, T = {T}s:")
    
    # Simulation
    np.random.seed(42)
    tau_sim, C_sim = sim_phase_network(
        N=N, I=I, alpha=1.0, sigma=sigma, beta=1.0,
        T=T, dt=dt, n_probe=20
    )
    
    # Theory
    tau_th, C_th, sigma_c = theory_phase_autocorr(
        I=I, alpha=1.0, sigma=sigma, beta=1.0,
        tau_max=min(T, 50), dtau=dt
    )
    
    print(f"  Sim:    C(0) = {C_sim[0]:.6f}")
    print(f"  Theory: C(0) = {C_th[0]:.6f}")
    print(f"  Ratio:  {C_sim[0]/max(C_th[0], 1e-10):.1f}x")
    print()
    
    # Plot
    ax = axes[idx]
    ax.plot(tau_sim[:500], C_sim[:500], 'b-', linewidth=2, label='Simulation', alpha=0.8)
    
    if C_th[0] > 0:
        C_th_scaled = C_th * (C_sim[0] / C_th[0])
        ax.plot(tau_th[:min(500, len(tau_th))], C_th_scaled[:min(500, len(tau_th))], 
                'r--', linewidth=2, label='Theory (rescaled)', alpha=0.8)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'dt = {dt}, T = {T}s, {int(T/dt)} steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(10, tau_sim[-1])])

plt.tight_layout()
plt.savefig('test_timestep_convergence.png', dpi=150)
print("✓ Saved test_timestep_convergence.png")
