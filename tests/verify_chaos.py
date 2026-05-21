#!/usr/bin/env python
"""Check if network exhibits chaotic dynamics (sensitivity to initial conditions)."""
import numpy as np
import matplotlib.pyplot as plt
from python.simulations.random_network import sim_phase_network

I, alpha, beta = 1.0, 1.0, 1.0
sigma_c = 2 * np.pi
N = 100
T = 50
dt = 0.01

print("Testing sensitivity to initial conditions (Lyapunov exponent proxy):\n")

sigma_test = 1.5 * sigma_c  # "Chaotic" regime

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Metric 1: Two runs with slightly different random seeds
print(f"σ = {sigma_test/(2*np.pi):.2f}×σ_c\n")

# Run 1: seed=42
np.random.seed(42)
tau1, C1 = sim_phase_network(N=N, I=I, alpha=alpha, sigma=sigma_test, beta=beta, 
                             T=T, dt=dt, n_probe=1)

# Run 2: seed=43 (different initial conditions)
np.random.seed(43)
tau2, C2 = sim_phase_network(N=N, I=I, alpha=alpha, sigma=sigma_test, beta=beta, 
                             T=T, dt=dt, n_probe=1)

axes[0, 0].plot(tau1, C1, label='Seed=42', linewidth=1.5)
axes[0, 0].plot(tau2, C2, label='Seed=43', linewidth=1.5, linestyle='--')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Autocorrelation')
axes[0, 0].set_title('Autocorrelations from different IC')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Metric 2: Lyapunov-like statistic - how much do adjacent initial conditions diverge?
print("Measuring divergence of trajectories from near-identical initial conditions:")

# Extract u trajectories for both runs
np.random.seed(42)
from python.simulations.random_network import make_weights
from numpy.random import standard_normal

W = make_weights(N, sigma_test)

# Run with u0
u0_init = standard_normal(N) * 0.1
phi0_init = 2 * np.pi * standard_normal(N)

u0 = u0_init.copy()
phi0 = phi0_init.copy()
u_traj0 = []

# Run with u0 + epsilon
eps = 1e-6
u_eps = u0_init + eps * standard_normal(N)
phi_eps = phi0_init.copy()
u_traj_eps = []

divergence = []

nt = int(T / dt)
for t in range(nt):
    # Trajectory 1
    F0 = np.maximum(I + u0, 0.0) ** (1.0/alpha)
    phi0 += F0 * dt
    spikes0 = (phi0 >= np.pi)
    phi0[spikes0] -= 2*np.pi
    drive0 = W @ spikes0.astype(float)
    u0 += dt * (-beta*u0 + beta*drive0)
    u_traj0.append(u0.copy())
    
    # Trajectory 2
    F_eps = np.maximum(I + u_eps, 0.0) ** (1.0/alpha)
    phi_eps += F_eps * dt
    spikes_eps = (phi_eps >= np.pi)
    phi_eps[spikes_eps] -= 2*np.pi
    drive_eps = W @ spikes_eps.astype(float)
    u_eps += dt * (-beta*u_eps + beta*drive_eps)
    u_traj_eps.append(u_eps.copy())
    
    # Divergence metric
    div = np.linalg.norm(u0 - u_eps)
    divergence.append(div)

u_traj0 = np.array(u_traj0)
u_traj_eps = np.array(u_traj_eps)
tau_fine = np.arange(nt) * dt

axes[0, 1].semilogy(tau_fine, divergence, linewidth=2)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('||u(0) - u(ε)||')
axes[0, 1].set_title(f'Trajectory divergence (ε={eps})')
axes[0, 1].grid(True, alpha=0.3, which='both')

# Estimate Lyapunov exponent from divergence curve
if len(divergence) > 100:
    early_divergence = np.array(divergence[10:100])
    early_time = tau_fine[10:100]
    # Linear fit in log space
    log_div = np.log(early_divergence)
    fit = np.polyfit(early_time, log_div, 1)
    lyap_est = fit[0]
    axes[0, 1].text(0.95, 0.05, f'λ ≈ {lyap_est:.4f} /s',
                   transform=axes[0, 1].transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    print(f"  Estimated Lyapunov exponent: λ ≈ {lyap_est:.6f} /s")
    print(f"  {'CHAOTIC' if lyap_est > 0 else 'NOT CHAOTIC'} (λ {'>' if lyap_est > 0 else '<'} 0)")
else:
    print("  Insufficient data")

print()

# Metric 3: Phase space trajectory in (u[0], u[1]) plane
axes[1, 0].plot(u_traj0[:1000, 0], u_traj0[:1000, 1], 'b.', alpha=0.3, markersize=2)
axes[1, 0].set_xlabel('u[0]')
axes[1, 0].set_ylabel('u[1]')
axes[1, 0].set_title('Phase space trajectory (first 1000 steps)')
axes[1, 0].grid(True, alpha=0.3)

# Metric 4: Return map: u[0](t) vs u[0](t+Δt)
dt_map = int(0.5 / dt)  # 0.5 second delay
if nt > dt_map:
    axes[1, 1].scatter(u_traj0[:-dt_map, 0], u_traj0[dt_map:, 0], 
                      alpha=0.3, s=10)
    axes[1, 1].set_xlabel('u[0](t)')
    axes[1, 1].set_ylabel(f'u[0](t+{dt*dt_map:.2f}s)')
    axes[1, 1].set_title('Return map')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chaos_verification.png', dpi=150)
print("✓ Saved chaos_verification.png")
