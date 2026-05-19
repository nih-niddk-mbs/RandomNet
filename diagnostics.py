import sys
import os
import numpy as np
from scipy.interpolate import interp1d

# Add python/simulations to path
sys.path.append(os.path.join(os.getcwd(), 'python/simulations'))
import two_pi_random_network_tests as tp

# 1. Run simulation
N=512; sigma=1.5; T=1200; dt=0.05; burn=200; lam=1
tau_s, C_s = tp.sim_rate_network(N=N, sigma=sigma, T=T, dt=dt, burn=burn, lam=lam)

# we want tau <= 12
mask = tau_s <= 12
tau_s_cut = tau_s[mask]
C_s_cut = C_s[mask]

def nrmse(y_true, y_pred, tau_grid):
    # interpolate pred onto true grid
    f_pred = interp1d(tau_grid, y_pred, bounds_error=False, fill_value="extrapolate")
    y_interp = f_pred(tau_s_cut)
    return np.sqrt(np.mean((C_s_cut - y_interp)**2)) / (np.max(C_s_cut) - np.min(C_s_cut))

# 2. Evaluate theory_rate_autocorr
# Variant A: C0=None
tau_th_A, C_th_A = tp.theory_rate_autocorr(C0=None, sigma=sigma, tau_max=15, dtau=0.05)
nrmse_A = nrmse(C_s_cut, C_th_A, tau_th_A)

# Variant B: C0=float(C_s[0])
tau_th_B, C_th_B = tp.theory_rtau_th_B, C_th_B = at(C_s[0]), sigma=sigmatau_th_Bx=tau_th_u=0.05)
nrmsnrmsnrmsnrmsnrmsnrmsnrmsnrmsnrmsnrm_B)


rmsnrmsnrmsnrmsnrmsnrmsnrmsnrmorrmsnrmsnrmsnrmsnrmsnrmsnrmsnrmorrmsnrmsnrmsnrmsnrmsnrmsnrmsne
defdefdefdefdefdefdefdefdefdefdefdefdefdeftaudefdefdefdtau=0.05, defdefdefdefdefdefdefdefdefdefdefdefdefdeftaudefdu defdefdefdefdefdefdefdefdefdefdefdefdefdeftaudefdefdefdtau=0.05, defdefdefdequad)
                    h_                 np                   nc(C_                                         h_              r                    h_         al, -0.999999, 0.999999))
        s = np.sqrt(2.0 * C0_val)
        x = s * gh_x[:, None]
                                  e]                                  e]                                  e]                 (np.sum(gw2 * vals) / np.pi)

    def integrate(c0):
                                                                                            
                                              Q = sigma**2 * Q_func(C[i], c0)
            Cpp = -C[i] + Q # SIGN FLIPPED
            C[i+1] = C[i] + dtau*Cp[i] + 0.5*dtau**2*Cpp
            Q_new = sigma**2 * Q_func(C            Q_new = sigma**2 * Q_func(C            Q_new = sigma**2 * Q_func(C            Q_+             Q_new = sigma**2 * Q_func(C            Q_new = sigma**2 * Q_func(C            Q_new = sigma**2 * Q_func(C            Q_+             Q_new = sigma**2 * Q_func(C            Q_new = sip.in            Q_new = sigma**2 * Q_func(C            Q_new = sigma**2 * Q_func(C            Q_new = sigma**2 * Q_func(C            Q_+             Q_new = sigma**2 * Q_func(C                    Q_new = sigma**2                  Q_new =float(best_c0))
    else:
        C_out = integrate(C0_val)
    return tau, C_out

# Variant C: alt C0=sim C0
tau_th_C, C_th_C = alternative_theory(C0_val=float(C_s[0tau_th_C, C_th_C = u_matau_th_C, C_th_C = alternative_theory(C0_val=float(C_s[0tau_th_C, C_th_C = u_matau_th_C, C_0 grid
tau_th_D, C_th_D = alternative_theory(C0_val=None, sigma=sigma, tau_max=1tau_th_D, C_th_D = alternative_theory(C0_thtau_th_D, C_th_D = alternative_theory(C0_val=None, sise_A:.6f}")
print(f"Theory (C0=sim): NRMSE = {nrmse_B:.6f}")
print(f"Alt Integrator (C0=sim): NRMSE = {nrmse_C:.6f}")
print(f"Alt Integrator (C0=auto): NRMSE = {nrmse_D:.6f}")

nrmses = [nrmse_A, nrmse_B, nrmse_C, nrmse_D]
labels = ["Theory (C0=None)", "Theory (C0=sim)", "Alt Integrator (C0=sim)", "Alt Integrator (C0=auto)"]
best_idx = np.argmin(nrmses)
print(f"\nBest variant: {labels[best_idx]}")
