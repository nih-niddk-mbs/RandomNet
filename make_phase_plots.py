"""Generate phase model comparison plots with shot noise correction."""
import sys
import os
sys.path.insert(0, 'scripts')
import matplotlib
matplotlib.use('Agg')
import numpy as np

from random_network import (
    default_results_dir,
    plot_phase_corr_params,
    plot_phase_corr_N,
    plot_phase_network,
    theory_phase_autocorr,
)

plot_dir = default_results_dir()
os.makedirs(plot_dir, exist_ok=True)

I, alpha, beta = 1.0, 1.0, 1.0
_, _, sigma_c = theory_phase_autocorr(
    I=I, alpha=alpha, sigma=1.0, beta=beta, tau_max=5, dtau=0.1
)
print(f'sigma_c = {sigma_c:.4f}')

# ------------------------------------------------------------------
# 1. Parameter dependence: 3 sigma values
# ------------------------------------------------------------------
print('\n--- plot_phase_corr_params ---')
plot_phase_corr_params(
    param_sets=[
        dict(I=1.0, alpha=1.0, beta=1.0, sigma=1.1 * sigma_c,
             label=r'$\sigma=1.1\sigma_c$'),
        dict(I=1.0, alpha=1.0, beta=1.0, sigma=1.5 * sigma_c,
             label=r'$\sigma=1.5\sigma_c$'),
        dict(I=1.0, alpha=1.0, beta=1.0, sigma=2.0 * sigma_c,
             label=r'$\sigma=2.0\sigma_c$'),
    ],
    N=512, T=3000.0, dt=0.02, tau_max=60.0, sim_reps=2,
    plot_dir=plot_dir,
)
print('  done.')

# ------------------------------------------------------------------
# 2. Phase network across sigma
# ------------------------------------------------------------------
print('\n--- plot_phase_network ---')
plot_phase_network(
    I=I, alpha=alpha, beta=beta,
    N=512,
    sigma_vals=[1.1 * sigma_c, 1.5 * sigma_c, 2.0 * sigma_c],
    T=4000.0, dt=0.02, tau_max=80.0, sim_reps=2,
    plot_dir=plot_dir,
)
print('  done.')

# ------------------------------------------------------------------
# 3. N convergence
# ------------------------------------------------------------------
print('\n--- plot_phase_corr_N ---')
plot_phase_corr_N(
    N_vals=(128, 256, 512),
    I=I, alpha=alpha, beta=beta,
    T=3000.0, dt=0.02, tau_max=60.0,
    plot_dir=plot_dir,
)
print('  done.')

print(f'\nAll plots saved to {plot_dir}')
