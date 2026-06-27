#!/usr/bin/env python
"""
Generate phase model simulation vs theory comparisons.
All outputs are saved to the configured RandomNet results folder.
"""

import sys
import os
import numpy as np
sys.path.insert(0, 'scripts')

from random_network import (
    default_results_dir,
    plot_phase_spike_correlation,
    plot_phase_network,
    plot_u_timeseries,
    plot_phase_raster,
    plot_phase_corr_params,
    plot_phase_corr_N,
    theory_phase_autocorr,
)

# Consistent output directory
plot_dir = default_results_dir()
os.makedirs(plot_dir, exist_ok=True)

print('='*70)
print('PHASE MODEL: Simulation vs Theory Comparisons')
print('='*70)
print(f'\nOutput directory: {plot_dir}\n')

# Compute sigma_c for I=1, alpha=1 (= 2*pi for alpha=1)
I, alpha, beta = 1.0, 1.0, 1.0
_, _, sigma_c = theory_phase_autocorr(I=I, alpha=alpha, sigma=1.0, beta=beta, tau_max=5, dtau=0.1)
print(f'sigma_c = {sigma_c:.4f}\n')

# ------------------------------------------------------------------
# 1. u(t) time series: sub, critical, and chaotic regimes
# ------------------------------------------------------------------
print('1. u(t) time series...')
plot_u_timeseries(
    sigma_vals=[0.5 * sigma_c, 0.9 * sigma_c, 1.1 * sigma_c, 1.5 * sigma_c, 3.0 * sigma_c],
    I=I, alpha=alpha, beta=beta,
    N=256, T=300.0, dt=0.02, n_show=8, burn=100.0,
    plot_dir=plot_dir,
)
print('   ✓ Complete\n')

# ------------------------------------------------------------------
# 2. Spike raster + population rate
# ------------------------------------------------------------------
print('2. Spike raster...')
plot_phase_raster(
    sigma_vals=[0.5 * sigma_c, 0.9 * sigma_c, 1.1 * sigma_c, 1.5 * sigma_c, 3.0 * sigma_c],
    I=I, alpha=alpha, beta=beta,
    N=256, T=500.0, dt=0.02, burn=100.0,
    plot_dir=plot_dir,
)
print('   ✓ Complete\n')

# ------------------------------------------------------------------
# 3. Sim vs theory: parameter dependence
# ------------------------------------------------------------------
print('3. Sim vs theory — parameter dependence...')
plot_phase_corr_params(
    N=512, T=5000.0, dt=0.02, tau_max=80.0, sim_reps=2,
    plot_dir=plot_dir,
)
print('   ✓ Complete\n')

# ------------------------------------------------------------------
# 4. Sim vs theory: N convergence
# ------------------------------------------------------------------
print('4. Sim vs theory — N convergence...')
plot_phase_corr_N(
    N_vals=(64, 128, 256, 512),
    I=I, alpha=alpha, beta=beta,
    T=4000.0, dt=0.02, tau_max=80.0,
    plot_dir=plot_dir,
)
print('   ✓ Complete\n')

# ------------------------------------------------------------------
# 5. Phase network: correlation functions spanning sub → chaotic
# ------------------------------------------------------------------
sigma_vals = [r * sigma_c for r in (1.0, 1.1, 1.5, 2.0, 3.0)]
print(f'5. Phase network C_uu: sigma/sigma_c = 1.0, 1.1, 1.5, 2.0, 3.0')
plot_phase_network(
    I=I, alpha=alpha, beta=beta,
    N=512, sigma_vals=sigma_vals,
    T=8000.0, dt=0.02, tau_max=150.0, sim_reps=2,
    plot_dir=plot_dir,
)
print('   ✓ Complete\n')

print('='*70)
print(f'All plots saved to {plot_dir}')
print('='*70)
