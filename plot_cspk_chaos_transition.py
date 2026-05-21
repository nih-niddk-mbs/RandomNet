#!/usr/bin/env python
"""
Plot spike rate autocorrelation C_spk(τ) to visualize chaos transition.

The key insight: C_11(τ) is the synaptic input filtered by β e^{-βτ}, which masks
chaotic structure. The spike rate autocorrelation C_spk(τ) = Q(τ) is unfiltered
and directly shows the chaos transition.

Theory predicts decay timescale ~ β√(1-g²) sub-critical, diverging as g→1⁻
and finite (chaotic) above σ_c.
"""
import numpy as np
import matplotlib.pyplot as plt
from python.simulations.random_network import (
    sim_phase_network,
    theory_phase_spike_autocorr,
)

# Parameters
I, alpha, beta = 1.0, 1.0, 1.0
N = 256
T = 200.0  # longer for better stats on slow decays
dt = 0.02
rng = np.random.default_rng(seed=42)

# Theory critical coupling
rho = 1.0 / (2.0 * np.pi)
Fprime0 = I ** (1.0 / alpha - 1.0)
sigma_c = 1.0 / (rho * Fprime0)

print(f"Phase model: I={I}, alpha={alpha}, β={beta}, σ_c={sigma_c:.3f}\n")

# σ values spanning the transition
sigma_vals = [
    0.75 * sigma_c,  # g = 0.75, well sub-critical
    0.90 * sigma_c,  # g = 0.90, near transition
    0.95 * sigma_c,  # g = 0.95, very close to transition
    1.05 * sigma_c,  # g = 1.05, just above transition
    1.50 * sigma_c,  # g = 1.50, well super-critical
]

# Storage
results = {}

# Run simulations
print("Running simulations...")
for sigma in sigma_vals:
    g = sigma / sigma_c
    print(f"  σ = {sigma:.3f} (g = {g:.2f})... ", end="", flush=True)

    # Simulate with spike rate
    tau_sim, C_uu, C_spk = sim_phase_network(
        N=N,
        I=I,
        alpha=alpha,
        sigma=sigma,
        beta=beta,
        T=T,
        dt=dt,
        n_probe=N,
        return_spike=True,
        rng=rng,
    )

    # Theory spike autocorr (when available)
    if sigma < sigma_c:
        tau_th, C_spk_th, _ = theory_phase_spike_autocorr(
            I=I,
            alpha=alpha,
            sigma=sigma,
            beta=beta,
            tau_max=float(tau_sim[-1]),
            dtau=dt,
        )
    else:
        tau_th, C_spk_th = None, None

    results[g] = {
        "sigma": sigma,
        "tau_sim": tau_sim,
        "C_uu": C_uu,
        "C_spk": C_spk,
        "tau_th": tau_th,
        "C_spk_th": C_spk_th,
    }
    print("done")

print()

# Plot 1: C_spk(τ) across σ values
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, (g, data) in enumerate(sorted(results.items())):
    ax = axes[idx]
    sigma = data["sigma"]
    tau_sim = data["tau_sim"]
    C_spk = data["C_spk"]
    tau_th = data["tau_th"]
    C_spk_th = data["C_spk_th"]

    # Normalize
    if abs(C_spk[0]) > 1e-12:
        C_spk_norm = C_spk / C_spk[0]
    else:
        C_spk_norm = C_spk

    if C_spk_th is not None and abs(C_spk_th[0]) > 1e-12:
        C_spk_th_norm = C_spk_th / C_spk_th[0]
    else:
        C_spk_th_norm = C_spk_th

    # Plot
    ax.plot(tau_sim, C_spk_norm, "b-", lw=1.5, label="Simulation", alpha=0.85)
    if C_spk_th is not None:
        ax.plot(tau_th, C_spk_th_norm, "r--", lw=2, label="Theory", alpha=0.75)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$\tau$", fontsize=11)
    ax.set_ylabel(r"$C_{\rm spk}(\tau) / C_{\rm spk}(0)$", fontsize=11)
    ax.set_title(rf"$\sigma = {g:.2f} \sigma_c$", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, min(50, tau_sim[-1]))
    if g <= 0.95:
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=9, loc="upper right")

# Remove the last (unused) subplot
fig.delaxes(axes[-1])

plt.suptitle(
    r"Spike rate autocorrelation $C_{\rm spk}(\tau) = Q(\tau)$ across the chaos transition",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("cspk_chaos_transition.png", dpi=150)
print("Saved: cspk_chaos_transition.png")
plt.show()

# Plot 2: Overlay normalized curves with different σ
fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(results)))

for (g, data), color in zip(sorted(results.items()), colors):
    sigma = data["sigma"]
    tau_sim = data["tau_sim"]
    C_spk = data["C_spk"]

    if abs(C_spk[0]) > 1e-12:
        C_spk_norm = C_spk / C_spk[0]
    else:
        C_spk_norm = C_spk

    ax.semilogy(tau_sim, np.maximum(np.abs(C_spk_norm), 1e-4), lw=2, label=f"g={g:.2f}", color=color)

ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5, label=r"$\beta^{-1}$ timescale")
ax.set_xlabel(r"$\tau$ (s)", fontsize=12)
ax.set_ylabel(r"$|C_{\rm spk}(\tau) / C_{\rm spk}(0)|$", fontsize=12)
ax.set_title(
    r"Log-scale: observe timescale slowing as $g \to 1^-$ and persistent tail for $g > 1$",
    fontsize=12,
)
ax.grid(True, alpha=0.3, which="both")
ax.set_xlim(0, 50)
ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig("cspk_logscale_overlay.png", dpi=150)
print("Saved: cspk_logscale_overlay.png")
plt.show()

# Print timescale measurements
print("\n" + "=" * 70)
print("DECAY TIMESCALE ANALYSIS (time to decay to 1/e of initial)")
print("=" * 70)
print(f"{'g = σ/σ_c':<15} {'τ_decay (theory)':<20} {'τ_decay (sim)':<20} {'Regime':<20}")
print("-" * 70)

for g, data in sorted(results.items()):
    C_spk = data["C_spk"]
    tau_sim = data["tau_sim"]
    C_spk_th = data["C_spk_th"]
    tau_th = data["tau_th"]

    # Find decay timescale from simulation (time to 1/e)
    target = 1.0 / np.e if abs(C_spk[0]) < 1e-12 else C_spk[0] / np.e
    C_spk_norm = np.abs(C_spk / C_spk[0]) if abs(C_spk[0]) > 1e-12 else np.abs(C_spk)
    
    # Find first crossing below 1/e
    idx_sim = np.where(C_spk_norm <= 1.0 / np.e)[0]
    tau_decay_sim = tau_sim[idx_sim[0]] if len(idx_sim) > 0 else np.nan

    # Theory timescale
    if C_spk_th is not None and len(C_spk_th) > 0 and abs(C_spk_th[0]) > 1e-12:
        C_spk_th_norm = np.abs(C_spk_th / C_spk_th[0])
        idx_th = np.where(C_spk_th_norm <= 1.0 / np.e)[0]
        tau_decay_th = tau_th[idx_th[0]] if len(idx_th) > 0 else np.nan
        regime = "sub-crit"
    else:
        tau_decay_th = np.nan
        regime = "super-crit"

    print(
        f"{g:<15.2f} {tau_decay_th:<20.3f} {tau_decay_sim:<20.3f} {regime:<20}"
    )

print()
print("Theory prediction:")
print("  Sub-critical (g < 1): τ ~ β√(1-g²) → ∞ as g → 1⁻")
print("  Super-critical (g > 1): τ ~ finite (chaotic), determined by F nonlinearity")
