#!/usr/bin/env python
"""Debug why u variance is so small in simulations."""
import numpy as np
from python.simulations.random_network import make_weights

N = 100
T = 1
dt = 0.01
nt = int(T / dt)

W = make_weights(N, 0.8 * 2 * np.pi)
np.random.seed(42)
phi = 2*np.pi*np.random.randn(N)
u = np.random.randn(N) * 0.1

spike_list = []
u_samples = []
drive_samples = []

for t in range(nt):
    I, alpha = 1.0, 1.0
    F_vals = np.maximum(I + u, 0.0) ** (1.0/alpha)
    phi += F_vals * dt
    
    spikes = (phi >= np.pi)  # boolean array
    spike_list.append(spikes.astype(float))  # store as float
    phi[spikes] -= 2*np.pi
    
    drive = W @ spikes.astype(float)
    drive_samples.append(drive.copy())
    u += dt * (-u + drive)
    u_samples.append(u.copy())

u_array = np.array(u_samples)
drive_array = np.array(drive_samples)
spike_array = np.array(spike_list)

avg_spike_prob = np.mean(spike_array)
print(f"Mean spike probability per neuron: {avg_spike_prob:.4f}")
print(f"Spike rate (spikes/sec): {avg_spike_prob/dt:.1f}")
print()

print(f"u field statistics:")
print(f"  Variance: {np.var(u_array):.6f}")
print(f"  Mean: {np.mean(u_array):.6f}")
print(f"  Std: {np.std(u_array):.6f}")
print()

print(f"Drive (W @ spike) statistics:")
print(f"  Variance: {np.var(drive_array):.6f}")
print(f"  Mean: {np.mean(drive_array):.6f}")
print()

expected_drive_var = (0.8 * 2*np.pi)**2 * avg_spike_prob * (1 - avg_spike_prob)
print(f"Expected drive variance: {expected_drive_var:.6f}")
print()

print(f"Steady-state u variance prediction:")
print(f"  From du/dt = -u + drive, at steady state:")
print(f"  Var[u] ≈ Var[drive] / (2 * decay_rate)")
print(f"  Decay rate ~ 1/dt = {1/dt:.0f}")
print(f"  But dt is small so effective steady state has Var[u] ~ Var[drive] * dt")
print(f"  Var[u] ≈ {np.var(drive_array):.6f} * {dt} = {np.var(drive_array) * dt:.6f}")
print()

print(f"Actual vs predicted Var[u]: {np.var(u_array):.6f} vs {np.var(drive_array) * dt:.6f}")
print(f"Ratio: {np.var(u_array) / max(np.var(drive_array) * dt, 1e-10):.4f}")
