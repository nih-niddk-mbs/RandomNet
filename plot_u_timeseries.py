#!/usr/bin/env python
"""Plot time series of u(t) from phase network simulation."""
import numpy as np
import matplotlib.pyplot as plt
from python.simulations.random_network import sim_phase_network

# Run simulation
N = 100
sigma = 1.5 * 2 * np.pi
T = 50  # seconds
dt = 0.01

print(f"Running simulation: N={N}, sigma={sigma/(2*np.pi):.2f}×σ_c, T={T}s, dt={dt}")
tau, C = sim_phase_network(N=N, sigma=sigma, T=T, dt=dt, n_probe=N)

# We need to capture u(t), but sim_phase_network doesn't return it
# Let's run it again and capture directly
from python.simulations.random_network import make_weights

I, alpha, beta = 1.0, 1.0, 1.0
W = make_weights(N, sigma, lam=1)
phi = np.random.uniform(-np.pi, np.pi, N)
u = np.zeros(N)

def F(u_):
    return alpha * np.maximum(I + u_, 0.0) ** (1.0 / alpha)

# Burn-in
nb = int(300 / dt)
for _ in range(nb):
    phi += F(u) * dt
    spikes = phi >= np.pi
    phi[spikes] -= 2.0 * np.pi
    drive = W @ spikes.astype(float)
    u += dt * (-beta * u + beta * drive)

# Record
nt = int(T / dt)
u_history = np.zeros((nt, N))
phi_history = np.zeros((nt, N))

for t in range(nt):
    phi += F(u) * dt
    spikes = phi >= np.pi
    phi[spikes] -= 2.0 * np.pi
    drive = W @ spikes.astype(float)
    u += dt * (-beta * u + beta * drive)
    u_history[t] = u.copy()
    phi_history[t] = phi.copy()

time = np.arange(nt) * dt

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Plot u(t) for first 5 neurons
ax = axes[0]
for i in range(5):
    ax.plot(time, u_history[:, i], alpha=0.7, label=f'Neuron {i}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('u(t)')
ax.set_title('Synaptic input u(t) for first 5 neurons')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot mean u across all neurons
ax = axes[1]
u_mean = np.mean(u_history, axis=1)
u_std = np.std(u_history, axis=1)
ax.plot(time, u_mean, 'b-', lw=2, label='Mean')
ax.fill_between(time, u_mean - u_std, u_mean + u_std, alpha=0.3, label='±1 std')
ax.set_xlabel('Time (s)')
ax.set_ylabel('u(t)')
ax.set_title('Mean ± std of u(t) across all neurons')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot phase φ(t) for first 5 neurons
ax = axes[2]
for i in range(5):
    ax.plot(time, phi_history[:, i], alpha=0.7, label=f'Neuron {i}')
ax.axhline(np.pi, color='r', linestyle='--', alpha=0.5, label='Spike threshold (π)')
ax.axhline(-np.pi, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('φ(t)')
ax.set_title('Phase φ(t) for first 5 neurons')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('u_timeseries.png', dpi=150)
print("Saved: u_timeseries.png")
plt.show()

# Print statistics
print(f"\nStatistics:")
print(f"  u(t) mean: {np.mean(u_history):.6f}")
print(f"  u(t) std:  {np.std(u_history):.6f}")
print(f"  u(t) min:  {np.min(u_history):.6f}")
print(f"  u(t) max:  {np.max(u_history):.6f}")
print(f"  φ(t) range: [{np.min(phi_history):.3f}, {np.max(phi_history):.3f}]")
