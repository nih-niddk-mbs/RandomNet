#!/usr/bin/env python
"""Debug theory solutions to understand amplitude mismatch."""
import numpy as np
from python.simulations.random_network import theory_phase_autocorr

I_values = [1.0, 2.0, 5.0, 10.0]
sigma = 0.8 * 2 * np.pi

print("Theory analysis for different I:\n")

for I in I_values:
    print(f"I = {I}:")
    
    # Get theory solution
    tau, C, sigma_c = theory_phase_autocorr(
        I=I, alpha=1.0, sigma=sigma, beta=1.0, tau_max=5, dtau=0.01
    )
    
    # Manually compute the mean of the gain function
    rho = 1/(2*np.pi)
    def g(u):
        return rho * np.maximum(I + u, 0.0)
    
    # At the theory's solution, what is C0?
    C0_theory = C[0]
    
    # What's E[g] at this C0?
    gh_x, gh_w = np.polynomial.hermite.hermgauss(24)
    s = np.sqrt(2.0 * C0_theory)
    mu_g = np.sum(gh_w * g(s * gh_x)) / np.sqrt(np.pi)
    
    print(f"  σ_c = {sigma_c/(2*np.pi):.3f}×2π")
    print(f"  Theory C(0) = {C0_theory:.6f}")
    print(f"  E[g] at this C0 = {mu_g:.6f}")
    print(f"  E[g]² = {mu_g**2:.6f}")
    print()
