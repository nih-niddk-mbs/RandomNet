#!/usr/bin/env python
"""Measure spike rates with different I values."""
import numpy as np
from python.simulations.random_network import make_weights

I_values = [1.0, 2.0, 5.0, 10.0]
sigma = 0.8 * 2 * np.pi
N = 100
T = 10  # seconds
dt = 0.01

print("Spike rate vs baseline input I:\n")

for I in I_values:
    np.random.seed(42)
    
    W = make_weights(N, sigma)
    phi = 2*np.pi*np.random.randn(N)
    u = np.random.randn(N) * 0.1
    
    total_spikes = 0
    u_samples = []
    
    nt = int(T / dt)
    for t in range(nt):
        F_vals = np.maximum(I + u, 0.0) ** (1.0/1.0)  # alpha=1.0
        phi += F_vals * dt
        
        spikes = (phi >= np.pi)
        total_spikes += np.sum(spikes)
        phi[spikes] -= 2*np.pi
        
        drive = W @ spikes.astype(float)
        u += dt * (-u + drive)
        u_samples.append(u.copy())
    
    u_array = np.array(u_samples)
    spike_rate = total_spikes / (N * T)
    u_var = np.var(u_array)
    
    print(f"I = {I:5.1f}:")
    print(f"  Spike rate: {spike_rate:.4f} spikes/neuron/sec")
    print(f"  u variance: {u_var:.6f}")
    print()
