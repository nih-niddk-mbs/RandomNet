#!/usr/bin/env python
"""
Verify what autocorr is computing: u-field vs spike train
"""
import numpy as np
from python.simulations.random_network import make_weights, autocorr

I, alpha, beta = 1.0, 1.0, 1.0
sigma = 0.5 * 2 * np.pi
N = 500

print("Running sim to extract raw u field and spike train...")
rng = np.random.default_rng(42)
W = make_weights(N, sigma, lam=1, rng=rng)
phi = rng.uniform(-np.pi, np.pi, N)
u = np.zeros(N)

def F(u_):
    return alpha * np.maximum(I + u_, 0.0) ** (1.0 / alpha)

# Burn-in
dt = 0.01
nb = int(200 / dt)
for _ in range(nb):
    phi += F(u) * dt
    spikes = phi >= np.pi
    phi[spikes] -= 2.0 * np.pi
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        drive = W @ spikes.astype(float)
    u += dt * (-beta * u + beta * drive)
    if not np.all(np.isfinite(u)):
        u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)

# Recording
T = 100
nt = int(T / dt)
u_trace = np.zeros((nt, N))
spike_trace = np.zeros(nt)

for t in range(nt):
    phi += F(u) * dt
    spikes = phi >= np.pi
    phi[spikes] -= 2.0 * np.pi
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        drive = W @ spikes.astype(float)
    u += dt * (-beta * u + beta * drive)
    if not np.all(np.isfinite(u)):
        u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
    
    u_trace[t] = u
    spike_trace[t] = np.mean(spikes.astype(float)) / dt

# Compute autocorr
C_u_raw = autocorr(u_trace[:, 0], max_lag=1000)  # One neuron's u
C_spike = autocorr(spike_trace, max_lag=1000)     # Population spike rate
C_u_avg = autocorr(np.mean(u_trace, axis=1), max_lag=1000)  # Average u

print(f"\nAutocorrelation at τ=0:")
print(f"  C_u(single neuron)[0] = {C_u_raw[0]:.4f}  (variance of u for neuron 0)")
print(f"  C_spike[0] = {C_spike[0]:.4f}  (variance of population spike rate)")
print(f"  C_u_avg[0] = {C_u_avg[0]:.4f}  (variance of mean u)")

print(f"\nDirect statistics:")
print(f"  var(u_trace[:, 0]) = {np.var(u_trace[:, 0]):.6f}")
print(f"  var(spike_trace) = {np.var(spike_trace):.4f}")
print(f"  var(mean(u_trace)) = {np.var(np.mean(u_trace, axis=1)):.6f}")

print(f"\nWhat sim_phase_network returns (average across probes):")
C_sim_output = np.mean([autocorr(u_trace[:, i], max_lag=1000) for i in range(min(20, N))], axis=0)
print(f"  C_sim[0] = {C_sim_output[0]:.4f}")
