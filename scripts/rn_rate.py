"""Rate-network simulations and SCS theory."""

import os

import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt

from rn_core import autocorr, make_weights, rng

# -----------------------------------------------------------------------------
# 1. RATE NEURON NETWORK
#    Model: du_i/dt = -u_i + sum_j W_ij f(u_j)
#    Theory: C''(tau) = C(tau) - sigma^2 Q(tau),
#            Q(tau) = <f(u(0))f(u(tau))>
# -----------------------------------------------------------------------------

def sim_rate_network(
    N=512,
    sigma=1.5,
    T=2000.0,
    dt=0.05,
    f=np.tanh,
    lam=1,
    burn=200,
    n_probe=64,
    rng=rng,
):
    """
    Euler-Maruyama integration of the rate network.
    Returns time-lagged autocorrelation of u averaged over neurons.

    Parameters
    ----------
    N     : network size
    sigma : weight std  (chaos when sigma > 1 for tanh)
    T     : total simulation time
    dt    : time step
    f     : gain function
    lam   : 1=row-sum corrected, 0=plain
    burn  : burn-in time (discarded)
    """
    W = make_weights(N, sigma, lam, rng)
    nt = int(T / dt)
    nb = int(burn / dt)
    u = rng.normal(0, 0.1, N)

    # burn in
    for _ in range(nb):
        u += dt * (-u + W @ f(u))

    # record
    U = np.zeros((nt, N))
    for t in range(nt):
        u += dt * (-u + W @ f(u))
        U[t] = u

    # Average autocorrelation over a subset of neurons.
    # Using more probes reduces large-lag variance in finite simulations.
    max_lag = int(50 / dt)
    n_probe = int(max(1, min(N, n_probe)))
    C = np.mean([autocorr(U[:, i], max_lag) for i in range(n_probe)], axis=0)
    tau = np.arange(len(C)) * dt
    return tau, C


def theory_rate_autocorr(
    C0=None,
    sigma=1.5,
    tau_max=50,
    dtau=0.01,
    f=np.tanh,
    n_quad=20,
    C0_bounds=(0.05, 5.0),
    mu_f=None,
):
    """
         Solve the SCS equation:
             C''(tau) = C(tau) - sigma^2 Q(tau)
       Q(tau) = <f(u(0))f(u(tau))>_Gaussian
    with u ~ N(0, C(0)).

     We use deterministic Gauss-Hermite quadrature to evaluate
         Q(tau) = E[f(x) f(y)] with (x,y) jointly Gaussian.
     This keeps the theory curve reproducible and removes RNG dependence.

    The physical solution is the monotone branch that satisfies the energy
    condition V(C(0)) = V(0) = 0, where V'(C) = -C + sigma^2 Q(C).
    If C0 is None, we solve for that self-consistent C0 by bisection.
    
    If mu_f is provided (a scalar or callable), Q is computed as centered covariance:
        Q_centered(tau) = E[f(x)f(y)] - mu_f(C0)^2
    This is needed for activation functions with nonzero mean (e.g., ReLU-like gains).
    """
    ntau = int(tau_max / dtau)
    tau = np.arange(ntau) * dtau
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    gh_w2 = np.outer(gh_w, gh_w)

    def Q_func(C_tau, C0_val):
        """Deterministic quadrature for E[f(x)f(y)] with correlated Gaussian x,y,
        optionally centered by subtracting mu_f^2."""
        if C0_val <= 0:
            return 0.0
        rho = float(np.clip(C_tau / C0_val, -0.999999, 0.999999))
        scale = np.sqrt(2.0 * C0_val)
        x = scale * gh_x[:, None]
        y = scale * (rho * gh_x[:, None] + np.sqrt(1.0 - rho**2) * gh_x[None, :])
        vals = f(x) * f(y)
        Q_raw = float(np.sum(gh_w2 * vals) / np.pi)
        
        # Subtract mu^2 if centering is requested (for nonzero-mean activation functions)
        if mu_f is not None:
            mu = mu_f(C0_val) if callable(mu_f) else float(mu_f)
            Q_centered = Q_raw - mu**2
            return Q_centered
        return Q_raw

    def energy_endpoint(C0_val, n_grid=256):
        """Return H(C0) = C0^2 - 2∫_0^{C0} Q(C; C0) dC."""
        if C0_val <= 0:
            return np.nan
        C_grid = np.linspace(0.0, float(C0_val), n_grid)
        Q_grid = np.array([sigma**2 * Q_func(c, C0_val) for c in C_grid])
        integral_Q = np.zeros_like(C_grid)
        integral_Q[1:] = np.cumsum(0.5 * (Q_grid[1:] + Q_grid[:-1]) * np.diff(C_grid))
        return float(C0_val**2 - 2.0 * integral_Q[-1])

    def solve_c0(initial_guess=None):
        lo, hi = C0_bounds
        candidates = np.linspace(lo, hi, 24)
        if initial_guess is not None:
            candidates = np.unique(np.sort(np.append(candidates, float(initial_guess))))
        values = np.array([energy_endpoint(c0) for c0 in candidates])

        finite = np.isfinite(values)
        candidates = candidates[finite]
        values = values[finite]
        if len(candidates) == 0:
            return float(initial_guess if initial_guess is not None else 0.65)

        # Look for a NEGATIVE-to-POSITIVE sign change only.
        # +→- crossings are numerical artifacts near C0=0 and must be ignored;
        # the physical SCS fixed point is where H goes from negative to positive.
        for i in range(len(candidates) - 1):
            if values[i] == 0 and values[i] < values[i + 1]:
                return float(candidates[i])
            if values[i] < 0 and values[i + 1] > 0:          # physical -→+ crossing
                a, b = float(candidates[i]), float(candidates[i + 1])
                fa, fb = float(values[i]), float(values[i + 1])
                for _ in range(40):
                    m = 0.5 * (a + b)
                    fm = energy_endpoint(m)
                    if not np.isfinite(fm):
                        break
                    if abs(fm) < 1e-8:
                        return float(m)
                    if fa * fm <= 0:
                        b, fb = m, fm
                    else:
                        a, fa = m, fm
                return float(0.5 * (a + b))

        # No physical -→+ crossing found: no SCS fixed point exists.
        return float(lo)

    def monotone_solution(C0_val):
        C0_val = float(C0_val)
        n_grid = max(512, int(300 * max(C0_val, 1.0)))
        C_grid = np.linspace(0.0, C0_val, n_grid)
        Q_grid = np.array([sigma**2 * Q_func(c, C0_val) for c in C_grid])
        integral_Q = np.zeros_like(C_grid)
        integral_Q[1:] = np.cumsum(0.5 * (Q_grid[1:] + Q_grid[:-1]) * np.diff(C_grid))
        H_grid = np.maximum(C_grid**2 - 2.0 * integral_Q, 1e-14)

        # Since C decays monotonically, integrate d tau / dC = 1 / sqrt(H(C)).
        C_desc = C_grid[::-1]
        H_desc = H_grid[::-1]
        speed = np.sqrt(H_desc)
        dC = -np.diff(C_desc)
        seg_speed = 0.5 * (speed[:-1] + speed[1:])
        tau_desc = np.concatenate([[0.0], np.cumsum(dC / np.maximum(seg_speed, 1e-14))])
        tau_out = np.arange(ntau) * dtau
        C_out = np.interp(tau_out, tau_desc, C_desc, left=C0_val, right=0.0)
        return C_out

    if C0 is None:
        C0 = solve_c0(initial_guess=None)
    else:
        # Treat the supplied value as a guess, not as a hard constraint.
        C0 = solve_c0(initial_guess=C0)

    C = monotone_solution(C0)
    return tau, C


def plot_rate_network(sigma=1.5, N=512, C0_guess=0.8, plot_dir=None):
    """Compare simulation vs theory for rate network."""
    import os
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Simulating rate network: N={N}, sigma={sigma} ...")
    tau_sim, C_sim = sim_rate_network(N=N, sigma=sigma)
    C_sim /= C_sim[0]  # normalise

    print("Computing theory ...")
    tau_th, C_th = theory_rate_autocorr(C0=C0_guess, sigma=sigma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(tau_sim, C_sim, "b", lw=1.5, label="Simulation")
    ax.plot(tau_th, C_th / C_th[0], "r--", lw=2, label="SCS theory")
    ax.set(
        xlabel=r"$\tau$",
        ylabel=r"$C_{11}(\tau)/C_{11}(0)$",
        title=f"Rate network  $\sigma={sigma}$",
        xlim=(0, 20),
    )
    ax.legend()

    # power spectrum
    ax = axes[1]
    n = len(C_sim)
    fq = fftfreq(2 * n, d=tau_sim[1] - tau_sim[0])[:n]
    Sw = np.abs(fft(np.concatenate([C_sim, C_sim[::-1]]))[:n])
    ax.semilogy(fq, Sw, "b", lw=1.5, label="Simulation")
    ax.set(
        xlabel=r"$\omega / 2\pi$",
        ylabel="Power spectrum",
        title="Power spectrum",
        xlim=(0, 2),
    )
    ax.legend()

    plt.suptitle("Rate network: SCS test", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "rate_network_test.png"), dpi=150)
    print(f"Saved to {os.path.join(plot_dir, 'rate_network_test.png')}")
    plt.close("all")


