import sys, os
sys.path.insert(0, 'scripts')
import matplotlib; matplotlib.use('Agg')
from random_network import (
    plot_phase_corr_params,
    plot_phase_corr_N,
    plot_phase_operational_criticality,
    theory_phase_autocorr,
)


PHASE_THEORY_KWARGS = dict(
    # Minimal exploratory oscillatory kernel. These are dimensionless multiples
    # of beta because kernel_scaled_by_beta=True.
    kernel_omega=2.0,
    kernel_damping=1.0,
    kernel_scaled_by_beta=True,
    # Hermite covariance avoids cancellation for large-variance nonlinear gains.
    q_method='hermite',
    n_quad=48,
    hermite_order=32,
    # Used by solver='fd'; other solvers ignore these controls.
    fd_max_nfev=140,
    fd_tail_weight=10.0,
)

if __name__ == '__main__':
    plot_dir = os.path.join(os.getcwd(), 'data', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    theory_kwargs = dict(PHASE_THEORY_KWARGS)

    _, _, sigma_c = theory_phase_autocorr(
        I=1.0, alpha=1.0, sigma=1.0, beta=1.0,
        tau_max=5, dtau=0.1, **theory_kwargs,
    )
    print(f'sigma_c = {sigma_c:.4f}')
    print(f'theory_kwargs = {theory_kwargs}')

    print('--- plot_phase_corr_params ---')
    plot_phase_corr_params(
        N=256, T=1200.0, dt=0.02, tau_max=35.0, sim_reps=2,
        plot_dir=plot_dir, n_jobs=4, theory_kwargs=theory_kwargs,
    )

    print('--- plot_phase_corr_N ---')
    plot_phase_corr_N(
        N_vals=(128, 256, 512),
        T=1200.0, dt=0.02, tau_max=35.0,
        plot_dir=plot_dir, n_jobs=4, theory_kwargs=theory_kwargs,
    )

    print('--- plot_phase_operational_criticality ---')
    plot_phase_operational_criticality(
        alpha_vals=(0.5, 0.75, 1.0, 1.5, 2.0),
        g_bounds=(0.4, 1.4),
        n_scan=18,
        theory_kwargs=dict(tau_max=1.0, dtau=0.1),
        plot_dir=plot_dir,
    )
    print('Done.')
