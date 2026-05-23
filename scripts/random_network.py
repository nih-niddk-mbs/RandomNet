"""
Simulations to test the 2PI effective action theory for random neural networks:
        1. Rate neuron network          -- Sompolinsky-Crisanti-Sommers equation
        2. Phase neuron network         -- phase-reset spiking with synaptic drive u
        3. Binary neuron network        -- exact two-exponential formula

Run each section independently. Requires numpy, scipy, matplotlib.

Conventions used throughout:
  - Weights have variance sigma^2/N.  Theory quantities named Q_smooth or
    Q_centered include the sigma^2 prefactor when they appear in a C_uu equation.
  - Phase-model g(u)=rho*F(u), rho=1/(2*pi), has units of firing rate.
    D_shot=sigma^2*E[g(u)] and the filtered shot variance is beta*D_shot/2.
  - Binary-model C_uu is the covariance of the synaptic drive u itself; helper
    functions that receive C_uu should not multiply its standard deviation by
    sigma a second time.
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def autocorr(x, max_lag):
    """Unbiased autocorrelation via FFT, normalised so C(0)=variance."""
    n = len(x)
    max_lag = min(max_lag, n - 1)  # Can't correlate beyond series length
    xc = x - x.mean()
    full = np.real(ifft(np.abs(fft(xc, n=2 * n)) ** 2))[:max_lag]
    nrm = n - np.arange(max_lag)  # unbiased normalisation
    return full / nrm


def make_weights(N, sigma, lam=1, rng=rng):
    """
    Draw NxN Gaussian weights with std sigma/sqrt(N).
    lam=1  -> row-sum corrected (W @ 1 = 0)
    lam=0  -> plain Gaussian
    """
    W = rng.normal(0, sigma / np.sqrt(N), (N, N))
    np.fill_diagonal(W, 0)          # zero diagonal first
    if lam:
        # Correct off-diagonal row sums to zero (N-1 terms per row)
        row_sums = W.sum(axis=1, keepdims=True)
        W -= row_sums / (N - 1)
        np.fill_diagonal(W, 0)      # re-zero diagonal (it absorbed -correction)
    return W


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


# -----------------------------------------------------------------------------
# 1b. PHASE NEURON NETWORK
#     phi_i in [-pi, pi], dphi_i/dt = alpha * max(I + u_i, 0)^(1/alpha)
#     spikes when phi_i crosses pi; u_i is driven by spike input through W.
# -----------------------------------------------------------------------------

def sim_phase_network(
    N=512,
    I=1.0,
    alpha=1.0,
    sigma=7.0,
    beta=1.0,
    T=3000.0,
    dt=0.02,
    lam=1,
    burn=300,
    tau_max=50.0,
    n_probe=None,
    return_spike=False,
    return_u=False,
    rng=rng,
):
    """Simulate a phase-reset network and return C_uu(tau).

    If return_spike is True, also return C_spk(tau) estimated from the
    whole-network population spike-rate time series r(t)=N_spk(t)/(N*dt).
    This is much smoother than averaging per-neuron sparse spike trains.
    """
    W = make_weights(N, sigma, lam, rng)
    phi = rng.uniform(-np.pi, np.pi, N)
    u = np.zeros(N)

    def F(u_):
        return alpha * np.clip(I + u_, 0.0, 1e12) ** (1.0 / alpha)

    nb = int(burn / dt)
    for _ in range(nb):
        phi += F(u) * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        # Correct multi-spike: wrap phi into (-pi, pi) regardless of how many
        # cycles were completed in this step (F(u)*dt can exceed 2*pi for large u).
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            drive = W @ spike_counts
        u += -beta * u * dt + beta * drive
        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)

    nt = int(T / dt)
    if n_probe is None:
        n_probe = N
    n_probe = int(max(1, min(N, n_probe)))
    probe_idx = np.arange(n_probe)
    U_probe = np.zeros((nt, n_probe))
    R_pop = np.zeros(nt, dtype=float) if return_spike else None
    for t in range(nt):
        phi += F(u) * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            drive = W @ spike_counts
        u += -beta * u * dt + beta * drive
        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
        U_probe[t] = u[probe_idx]
        if return_spike:
            R_pop[t] = np.mean(spike_counts) / dt

    max_lag = int(tau_max / dt)
    # Average per-neuron autocorrelation: matches C_SCS(tau) at long lags (tau >> 1/beta)
    # where shot noise has decayed. At tau~0 includes shot noise contribution.
    C = np.mean([autocorr(U_probe[:, i], max_lag) for i in range(n_probe)], axis=0)
    tau = np.arange(len(C)) * dt
    if not return_spike and not return_u:
        return tau, C

    out = [tau, C]
    if return_spike:
        C_spk = autocorr(R_pop, max_lag)
        out.append(C_spk)
    if return_u:
        out.append(U_probe)
    return tuple(out)


def theory_phase_autocorr(
    I=1.0,
    alpha=1.0,
    sigma=7.0,
    beta=1.0,
    C0=None,
    tau_max=50,
    dtau=0.01,
    n_quad=24,
    solver="fd",
    kernel_omega=0.0,
    kernel_damping=0.0,
    kernel_scaled_by_beta=True,
    fd_max_nfev=250,
    fd_tail_weight=10.0,
    fd_kick_weight=5.0,
    q_method="gh",
    n_qmc=2048,
    hermite_order=32,
    cov_penalty_weight=5.0,
    warn_on_no_branch=True,
):
    """
    Solve the 2PI SCS equation for the phase neuron network with lambda=1,
    including the point-process shot-noise kick at tau=0.

    With row-sum correction (lambda=1), W@1=0 so E[u_i]=0 exactly, meaning
    C_11(tau->inf)=0 and C_eq=0. The centered SCS with gain g is directly correct:
    no g_shifted or C_eq machinery needed.

    The shot noise enters Q as D_shot delta(tau), which gives the initial
    velocity C'(0+) = -beta^2 D_shot/2.  The smooth tau>0 trajectory is then
    selected by the modified energy condition

        H(C0) = C0^2 - 2 int_0^C0 Q_smooth(C; C0) dC = beta^2 D_shot(C0)^2 / 4.

    solver="energy" uses the monotone conserved-energy branch and is valid only
    for the rate-like SCS operator.  solver="fd" solves the second-order equation
    directly on a finite-difference grid with the shot-noise derivative kick and
    quiet-tail boundary residuals.  solver="inflated_ic" tests the alternative
    interpretation where the filtered shot-noise variance only inflates C(0) and
    the smooth ODE is integrated with C'(0)=0.

    The finite-difference solver can test a minimal generalized phase kernel,

        L_C C = C'' + 2*kernel_damping*C'
                + (kernel_omega**2 - beta**2)*C,

    so the tau>0 equation is

        L_C C = -beta**2 * Q_smooth(C; C0).

    kernel_omega=kernel_damping=0 recovers the rate-like SCS reduction.  By
    default the generalized-kernel parameters are dimensionless multiples of
    beta, so kernel_omega=2 means omega=2*beta and kernel_damping=1 means
    damping=beta.  Set kernel_scaled_by_beta=False to pass raw time units.
    The fd_* weights are numerical continuation controls for the boundary
    residuals; oscillatory kernels usually need stronger tail weighting than the
    monotone SCS branch.
    q_method="gh" uses tensor Gauss-Hermite quadrature for Q_smooth. q_method="qmc"
    uses common-random Sobol Gaussian samples, which is slower but more robust
    for hard rectification and strongly nonlinear gains. q_method="hermite" uses
    a 1-D Hermite expansion of g(u) and evaluates the centered covariance as a
    series in C(tau)/C(0), avoiding cancellation from subtracting E[g]^2.
    warn_on_no_branch controls whether alpha<1 branch failures are printed; scans
    turn this off to avoid repetitive console noise.
    """
    rho = 1.0 / (2.0 * np.pi)

    def g(u):
        return rho * alpha * np.clip(I + u, 0.0, 1e12) ** (1.0 / alpha)

    Fprime0 = float(np.maximum(I, 1e-10) ** (1.0 / alpha - 1.0))
    sigma_c = 1.0 / (rho * Fprime0)

    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    gh_w2 = np.outer(gh_w, gh_w)
    q_method = str(q_method).lower()
    use_hermite_series = q_method in ("hermite", "hermite-series", "series")
    if q_method in ("sobol", "qmc", "mc"):
        try:
            from scipy.special import ndtri
            from scipy.stats import qmc

            sampler = qmc.Sobol(d=2, scramble=True, seed=12345)
            m = int(np.ceil(np.log2(max(16, int(n_qmc)))))
            u_sobol = sampler.random_base2(m)
            eps = np.finfo(float).eps
            z_qmc = ndtri(np.clip(u_sobol, eps, 1.0 - eps))
        except Exception:
            q_rng = np.random.default_rng(12345)
            z_qmc = q_rng.normal(size=(max(16, int(n_qmc)), 2))
        z0_qmc = z_qmc[:, 0]
        z1_qmc = z_qmc[:, 1]
    else:
        z0_qmc = z1_qmc = None
    hermite_cache = {}

    def hermite_coeffs(C0_val):
        """
        Return normalized Hermite coefficients b_n=E[g(sZ) phi_n(Z)].

        phi_n(Z)=He_n(Z)/sqrt(n!) are orthonormal under Z~N(0,1).  The
        centered covariance is sum_{n>=1} b_n^2 rho^n, so the mean mode b_0 is
        intentionally omitted by Q_centered.
        """
        C0_val = float(C0_val)
        key = round(C0_val, 12)
        if key in hermite_cache:
            return hermite_cache[key]

        z = np.sqrt(2.0) * gh_x
        vals = g(np.sqrt(max(C0_val, 0.0)) * z)
        w = gh_w / np.sqrt(np.pi)
        order = int(max(1, hermite_order))
        coeffs = np.zeros(order + 1)

        phi_nm1 = np.ones_like(z)
        coeffs[0] = float(np.dot(w, vals * phi_nm1))
        if order >= 1:
            phi_n = z
            coeffs[1] = float(np.dot(w, vals * phi_n))
            for n in range(1, order):
                phi_np1 = (z * phi_n - np.sqrt(n) * phi_nm1) / np.sqrt(n + 1.0)
                coeffs[n + 1] = float(np.dot(w, vals * phi_np1))
                phi_nm1, phi_n = phi_n, phi_np1

        # Avoid unbounded cache growth during finite-difference least_squares.
        if len(hermite_cache) > 2048:
            hermite_cache.clear()
        hermite_cache[key] = coeffs
        return coeffs

    def mu_g(C_val):
        """E[g(u)] for u ~ N(0, C_val) via 1-D GH quadrature."""
        C_val = float(C_val)
        if C_val <= 0.0:
            return float(g(0.0))
        if z0_qmc is not None:
            return float(np.mean(g(np.sqrt(C_val) * z0_qmc)))
        s = np.sqrt(2.0 * C_val)
        return float(np.sum(gh_w * g(s * gh_x)) / np.sqrt(np.pi))

    def Q_centered(C_tau, C0_val):
        """sigma^2 * Cov[g(u(0)), g(u(tau))] for a Gaussian pair."""
        C0_val = float(C0_val)
        if C0_val <= 0.0:
            return 0.0
        rho_tau = float(np.clip(C_tau / C0_val, -0.999999, 0.999999))
        if use_hermite_series:
            coeffs = hermite_coeffs(C0_val)
            powers = rho_tau ** np.arange(1, len(coeffs))
            return sigma**2 * float(np.dot(coeffs[1:] ** 2, powers))
        if z0_qmc is not None:
            s = np.sqrt(C0_val)
            x = s * z0_qmc
            y = s * (rho_tau * z0_qmc + np.sqrt(1.0 - rho_tau**2) * z1_qmc)
            gx = g(x)
            gy = g(y)
            return sigma**2 * float(np.mean(gx * gy) - np.mean(gx) * np.mean(gy))
        scale = np.sqrt(2.0 * C0_val)
        x = scale * gh_x[:, None]
        y = scale * (
            rho_tau * gh_x[:, None] + np.sqrt(1.0 - rho_tau**2) * gh_x[None, :]
        )
        raw = float(np.sum(gh_w2 * g(x) * g(y)) / np.pi)
        mu = mu_g(C0_val)
        return sigma**2 * (raw - mu**2)

    def D_shot(C0_val):
        """D_shot = sigma^2 * rho * <F(u)> = sigma^2 * <g(u)>."""
        return sigma**2 * mu_g(C0_val)

    beta_val = max(float(beta), 1e-10)
    kernel_scale = beta_val if kernel_scaled_by_beta else 1.0
    omega_val = float(kernel_omega) * kernel_scale
    damping_val = float(kernel_damping) * kernel_scale
    C0_hi = max(50.0, 10.0 * sigma ** 4 / (16.0 * np.pi ** 3))

    def energy_balance(C0_val, n_grid=384):
        C0_val = float(C0_val)
        if C0_val <= 0.0:
            return np.nan
        C_grid = np.linspace(0.0, C0_val, n_grid)
        Q_grid = np.array([Q_centered(c, C0_val) for c in C_grid])
        integral_Q = np.trapezoid(Q_grid, C_grid)
        H0 = C0_val**2 - 2.0 * integral_Q
        target = 0.25 * beta_val**2 * D_shot(C0_val) ** 2
        return float(H0 - target)

    def smooth_energy(C_val, C0_val, n_grid=384):
        """H(C; C0) = C^2 - 2 int_0^C Q_smooth(x; C0) dx."""
        C_val = float(C_val)
        C0_val = float(C0_val)
        if C_val <= 0.0 or C0_val <= 0.0:
            return np.nan
        C_grid = np.linspace(0.0, C_val, n_grid)
        Q_grid = np.array([Q_centered(c, C0_val) for c in C_grid])
        return float(C_val**2 - 2.0 * np.trapezoid(Q_grid, C_grid))

    def solve_c0():
        if C0 is not None:
            guess_val = float(C0)
            if guess_val > 0.0:
                return guess_val

        lo, hi = 1e-5, C0_hi
        linear = np.linspace(lo, min(hi, 10.0), 80)
        log = np.logspace(np.log10(lo), np.log10(hi), 80)
        candidates = np.unique(np.sort(np.concatenate([linear, log])))
        values = np.array([energy_balance(c) for c in candidates])
        finite = np.isfinite(values)
        candidates, values = candidates[finite], values[finite]
        if len(candidates) == 0:
            return 1e-4

        # Prefer the first negative-to-positive crossing; it is the monotone
        # branch continuously connected to the transition.
        for i in range(len(candidates) - 1):
            if values[i] <= 0.0 and values[i + 1] >= 0.0:
                a, b = float(candidates[i]), float(candidates[i + 1])
                fa = float(values[i])
                for _ in range(60):
                    m = 0.5 * (a + b)
                    fm = energy_balance(m)
                    if not np.isfinite(fm) or abs(fm) < 1e-10:
                        return float(m)
                    if fa * fm <= 0.0:
                        b = m
                    else:
                        a, fa = m, fm
                return float(0.5 * (a + b))

        # Superlinear gains can genuinely lose the finite stationary branch.
        # Returning the closest tiny value is misleading: it looks like a theory
        # curve but is really the failed zero-amplitude fallback.
        if alpha < 1.0:
            if warn_on_no_branch:
                print(
                    "  no finite phase-theory amplitude found for alpha<1; "
                    "Gaussian closure likely has no stationary branch here"
                )
            return np.nan

        # If no bracket appears, choose the closest balance point so plotting can
        # still reveal the mismatch instead of failing hard for benign regimes.
        return float(candidates[np.argmin(np.abs(values))])

    ntau = int(tau_max / dtau)
    tau = np.arange(ntau) * dtau

    solver_key = str(solver).lower()
    C0_val = None if solver_key in ("inflated_ic", "inflated-ic", "inflated") else solve_c0()

    def energy_solution(C0_val):
        n_grid = max(768, int(350 * max(C0_val, 1.0)))
        C_grid = np.linspace(0.0, C0_val, n_grid)
        Q_grid = np.array([Q_centered(c, C0_val) for c in C_grid])
        integral_Q = np.zeros_like(C_grid)
        integral_Q[1:] = np.cumsum(
            0.5 * (Q_grid[1:] + Q_grid[:-1]) * np.diff(C_grid)
        )

        # The shot-noise kick determines C0 through H(C0)=beta^2 D_shot^2/4.
        # For tau>0 the conserved-energy branch is still dC/dtau=-beta*sqrt(H(C));
        # no extra constant is added under the square root.
        H_grid = np.maximum(C_grid**2 - 2.0 * integral_Q, 1e-14)
        C_desc = C_grid[::-1]
        speed = beta_val * np.sqrt(H_grid[::-1])
        dC = -np.diff(C_desc)
        seg_speed = 0.5 * (speed[:-1] + speed[1:])
        tau_desc = np.concatenate(
            [[0.0], np.cumsum(dC / np.maximum(seg_speed, 1e-14))]
        )
        return np.interp(tau, tau_desc, C_desc, left=C0_val, right=0.0)

    def finite_difference_solution(C0_seed):
        from scipy.optimize import least_squares

        n_fd = int(min(max(ntau, 64), 700))
        tau_fd = np.linspace(0.0, float(tau[-1] if len(tau) else tau_max), n_fd)
        h = float(tau_fd[1] - tau_fd[0]) if n_fd > 1 else float(dtau)
        C_init = np.interp(tau_fd, tau, energy_solution(C0_seed))

        lower = np.full(n_fd, -C0_hi)
        upper = np.full(n_fd, C0_hi)
        lower[0] = 1e-8

        def residual(C_vec):
            C_vec = np.asarray(C_vec, dtype=float)
            C0_fd = float(max(C_vec[0], 1e-8))
            Q_vals = np.array([Q_centered(c, C0_fd) for c in C_vec])

            # Interior second-order finite-difference equation:
            # L_C C = -beta^2 Q_smooth(C; C0), with
            # L_C = d^2 + 2*damping*d + (omega^2-beta^2).
            first_deriv = (C_vec[2:] - C_vec[:-2]) / (2.0 * h)
            interior = (
                (C_vec[2:] - 2.0 * C_vec[1:-1] + C_vec[:-2]) / h**2
                + 2.0 * damping_val * first_deriv
                + (omega_val**2 - beta_val**2) * C_vec[1:-1]
                + beta_val**2 * Q_vals[1:-1]
            )

            # One-sided derivative kick at tau=0.
            left_deriv = (-3.0 * C_vec[0] + 4.0 * C_vec[1] - C_vec[2]) / (2.0 * h)
            left_bc = left_deriv + 0.5 * beta_val**2 * D_shot(C0_fd)

            # Quiet-tail residuals approximate C(tau->inf)=C'(tau->inf)=0.
            right_deriv = (3.0 * C_vec[-1] - 4.0 * C_vec[-2] + C_vec[-3]) / (2.0 * h)
            scale = max(abs(C0_fd), 1.0)
            cov_violation = np.maximum(np.abs(C_vec[1:]) - C0_fd, 0.0)
            return np.concatenate(
                [
                    interior / scale,
                    cov_penalty_weight * cov_violation / scale,
                    np.array(
                        [
                            fd_kick_weight * left_bc / (beta_val * scale),
                            fd_tail_weight * C_vec[-1] / scale,
                            fd_tail_weight * right_deriv / (beta_val * scale),
                        ]
                    ),
                ]
            )

        fit = least_squares(
            residual,
            C_init,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.1,
            max_nfev=int(fd_max_nfev),
            xtol=1e-5,
            ftol=1e-5,
            gtol=1e-5,
        )
        if not fit.success:
            print(f"  finite-difference phase theory did not fully converge: {fit.message}")
        return np.interp(tau, tau_fd, fit.x)

    def solve_inflated_total_c0():
        if C0 is not None:
            guess_val = float(C0)
            if guess_val > 0.0:
                return guess_val

        def balance(C_total):
            C_total = float(C_total)
            C_smooth0 = C_total - 0.5 * beta_val * D_shot(C_total)
            if C_total <= 0.0 or C_smooth0 <= 0.0:
                return np.nan
            return smooth_energy(C_smooth0, C_total)

        lo, hi = 1e-5, C0_hi
        linear = np.linspace(lo, min(hi, 10.0), 80)
        log = np.logspace(np.log10(lo), np.log10(hi), 80)
        candidates = np.unique(np.sort(np.concatenate([linear, log])))
        values = np.array([balance(c) for c in candidates])
        finite = np.isfinite(values)
        candidates, values = candidates[finite], values[finite]
        if len(candidates) == 0:
            return solve_c0()

        for i in range(len(candidates) - 1):
            if values[i] == 0.0:
                return float(candidates[i])
            if values[i] * values[i + 1] < 0.0:
                a, b = float(candidates[i]), float(candidates[i + 1])
                fa = float(values[i])
                for _ in range(60):
                    m = 0.5 * (a + b)
                    fm = balance(m)
                    if not np.isfinite(fm) or abs(fm) < 1e-10:
                        return float(m)
                    if fa * fm <= 0.0:
                        b = m
                    else:
                        a, fa = m, fm
                return float(0.5 * (a + b))
        if alpha < 1.0:
            if warn_on_no_branch:
                print(
                    "  no finite inflated-IC branch found for alpha<1; "
                    "not plotting a spurious near-zero theory curve"
                )
            return np.nan
        return float(candidates[np.argmin(np.abs(values))])

    def inflated_ic_solution():
        from scipy.integrate import solve_ivp

        C_total0 = solve_inflated_total_c0()
        if not np.isfinite(C_total0):
            return np.full_like(tau, np.nan, dtype=float)

        def rhs(_, y):
            c, cp = float(y[0]), float(y[1])
            q = Q_centered(c, C_total0)
            cpp = (
                -2.0 * damping_val * cp
                - (omega_val**2 - beta_val**2) * c
                - beta_val**2 * q
            )
            return [cp, cpp]

        sol = solve_ivp(
            rhs,
            (0.0, float(tau[-1] if len(tau) else tau_max)),
            [C_total0, 0.0],
            t_eval=tau,
            method="DOP853",
            rtol=1e-6,
            atol=1e-8,
        )
        if not sol.success:
            print(f"  inflated-IC phase theory did not fully converge: {sol.message}")
        return sol.y[0] if sol.y.shape[1] == len(tau) else np.interp(tau, sol.t, sol.y[0])

    if solver_key in ("energy", "monotone"):
        if not np.isfinite(C0_val):
            return tau, np.full_like(tau, np.nan, dtype=float), sigma_c
        if abs(omega_val) > 0.0 or abs(damping_val) > 0.0:
            raise ValueError("solver='energy' is only valid for kernel_omega=kernel_damping=0")
        C = energy_solution(C0_val)
    elif solver_key in ("fd", "finite_difference", "finite-difference"):
        if not np.isfinite(C0_val):
            return tau, np.full_like(tau, np.nan, dtype=float), sigma_c
        C = finite_difference_solution(C0_val)
    elif solver_key in ("inflated_ic", "inflated-ic", "inflated"):
        C = inflated_ic_solution()
    else:
        raise ValueError("solver must be 'fd', 'energy', or 'inflated_ic'")

    return tau, C, sigma_c


def shot_noise_correction(tau, sigma, beta, I, alpha, C0_total, n_quad=24):
    """
    Return the shot-noise autocorrelation to subtract from C_uu^sim(tau).

    When spikes are Poisson with rate g(u) = rho*F(u) and u is filtered
    through an exponential kernel with rate beta, each spike contributes a
    PSC h(s) = beta*W_ij*exp(-beta*s).  The resulting shot-noise covariance is

        C_shot(tau) = (sigma^2 * beta / 2) * E[g(u)] * exp(-beta * |tau|)

    where E[g(u)] is estimated from 1-D GH quadrature at variance C0_total
    (the empirical total C_uu(0), which already includes both SCS and shot-noise
    contributions, giving a self-consistent estimate of the mean firing rate).
    """
    rho = 1.0 / (2.0 * np.pi)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)

    def g(u):
        return rho * alpha * np.clip(I + u, 0.0, 1e12) ** (1.0 / alpha)

    C0 = float(max(C0_total, 1e-14))
    s = np.sqrt(2.0 * C0)
    mean_g = float(np.sum(gh_w * g(s * gh_x)) / np.sqrt(np.pi))

    amplitude = 0.5 * sigma ** 2 * float(beta) * mean_g
    return amplitude * np.exp(-float(beta) * np.abs(tau))


def phase_gaussian_gain_covariance(C_tau, C0, I=1.0, alpha=1.0, sigma=1.0,
                                   n_quad=32, q_method="gh", n_qmc=4096,
                                   hermite_order=32):
    """Gaussian closure for sigma^2 * Cov[g(u0), g(utau)] with g=rho*F."""
    rho_phase = 1.0 / (2.0 * np.pi)

    def g(u):
        return rho_phase * alpha * np.clip(I + u, 0.0, 1e12) ** (1.0 / alpha)

    C0 = float(C0)
    if C0 <= 0.0:
        return 0.0
    r = float(np.clip(C_tau / C0, -0.999999, 0.999999))
    if str(q_method).lower() in ("hermite", "hermite-series", "series"):
        gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
        z = np.sqrt(2.0) * gh_x
        vals = g(np.sqrt(C0) * z)
        w = gh_w / np.sqrt(np.pi)
        order = int(max(1, hermite_order))
        coeffs = np.zeros(order + 1)
        phi_nm1 = np.ones_like(z)
        coeffs[0] = float(np.dot(w, vals * phi_nm1))
        if order >= 1:
            phi_n = z
            coeffs[1] = float(np.dot(w, vals * phi_n))
            for n in range(1, order):
                phi_np1 = (z * phi_n - np.sqrt(n) * phi_nm1) / np.sqrt(n + 1.0)
                coeffs[n + 1] = float(np.dot(w, vals * phi_np1))
                phi_nm1, phi_n = phi_n, phi_np1
        powers = r ** np.arange(1, len(coeffs))
        return sigma**2 * float(np.dot(coeffs[1:] ** 2, powers))

    if str(q_method).lower() in ("qmc", "sobol", "mc"):
        try:
            from scipy.special import ndtri
            from scipy.stats import qmc

            sampler = qmc.Sobol(d=2, scramble=True, seed=12345)
            m = int(np.ceil(np.log2(max(16, int(n_qmc)))))
            uu = sampler.random_base2(m)
            eps = np.finfo(float).eps
            z = ndtri(np.clip(uu, eps, 1.0 - eps))
        except Exception:
            z = np.random.default_rng(12345).normal(size=(max(16, int(n_qmc)), 2))
        x = np.sqrt(C0) * z[:, 0]
        y = np.sqrt(C0) * (r * z[:, 0] + np.sqrt(1.0 - r**2) * z[:, 1])
        gx = g(x)
        gy = g(y)
        return sigma**2 * float(np.mean(gx * gy) - np.mean(gx) * np.mean(gy))

    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    gh_w2 = np.outer(gh_w, gh_w)
    scale = np.sqrt(2.0 * C0)
    x = scale * gh_x[:, None]
    y = scale * (r * gh_x[:, None] + np.sqrt(1.0 - r**2) * gh_x[None, :])
    raw = float(np.sum(gh_w2 * g(x) * g(y)) / np.pi)
    mu = float(np.sum(gh_w * g(scale * gh_x)) / np.sqrt(np.pi))
    return sigma**2 * (raw - mu**2)


def plot_phase_nonlinear_closure_diagnostic(
    examples=None,
    N=512,
    T=1200.0,
    dt=0.02,
    burn=300.0,
    tau_max=20.0,
    n_probe=128,
    plot_dir=None,
    q_method="qmc",
):
    """Compare empirical nonlinear gain covariance to the Gaussian Q(C) closure."""
    import os

    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if examples is None:
        examples = []
        for alpha_val in (0.5, 1.0, 2.0):
            sc = _phase_sigma_c(1.0, alpha_val)
            examples.append(dict(I=1.0, alpha=alpha_val, beta=1.0, sigma=1.3 * sc,
                                 label=fr"$\alpha={alpha_val},\ g=1.3$"))

    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2))
    if n == 1:
        axes = [axes]

    max_lag = int(tau_max / dt)

    for ax, ex in zip(axes, examples):
        I = ex.get("I", 1.0)
        alpha = ex.get("alpha", 1.0)
        beta = ex.get("beta", 1.0)
        sigma = ex["sigma"]
        label = ex.get("label", fr"$\alpha={alpha}$")
        print(f"  nonlinear closure diagnostic {label}: sigma={sigma:.3f}")

        tau, Cuu, U = sim_phase_network(
            N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
            T=T, dt=dt, burn=burn, tau_max=tau_max, n_probe=n_probe,
            return_u=True,
        )

        rho_phase = 1.0 / (2.0 * np.pi)
        G = rho_phase * alpha * np.clip(I + U, 0.0, 1e12) ** (1.0 / alpha)
        Cg = np.mean([autocorr(G[:, i], max_lag) for i in range(G.shape[1])], axis=0)

        C0 = float(max(Cuu[0], 1e-12))
        C_grid = np.linspace(min(Cuu[:max_lag].min(), -0.2 * C0), C0, 160)
        Q_gauss = np.array([
            phase_gaussian_gain_covariance(
                c, C0, I=I, alpha=alpha, sigma=sigma, q_method=q_method
            )
            for c in C_grid
        ])
        q_norm = max(abs(Q_gauss[-1]), 1e-12)

        ax.plot(C_grid / C0, Q_gauss / q_norm, "k--", lw=2, label="Gaussian closure")
        ax.plot(Cuu[:max_lag] / C0, sigma**2 * Cg[:max_lag] / q_norm,
                color="C0", lw=1.7, label="Empirical sim")
        ax.scatter(Cuu[:max_lag:25] / C0, sigma**2 * Cg[:max_lag:25] / q_norm,
                   s=14, color="C0", alpha=0.6)

        u_flat = U.reshape(-1)
        std = np.std(u_flat)
        z = (u_flat - np.mean(u_flat)) / max(std, 1e-12)
        skew = np.mean(z**3)
        kurt = np.mean(z**4) - 3.0
        ax.set(
            xlabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
            ylabel=r"$Q_{\rm smooth}/Q_{\rm smooth}(0)$",
            title=label + "\n" + fr"skew={skew:.2f}, excess kurt={kurt:.2f}",
        )
        ax.legend(fontsize=8)
        ax.axhline(0, color="0.7", lw=0.7)
        ax.axvline(0, color="0.7", lw=0.7)

    plt.suptitle("Nonlinear gain covariance: empirical vs Gaussian closure", fontsize=13, fontweight="bold")
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_nonlinear_closure_diagnostic.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


def phase_shot_fixed_point_map(C_vals, I=1.0, alpha=1.0, sigma=7.0, beta=1.0,
                               n_quad=64):
    """Return shot-noise-only variance contribution beta*D_shot(C)/2."""
    rho_phase = 1.0 / (2.0 * np.pi)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    C_vals = np.asarray(C_vals, dtype=float)
    out = np.zeros_like(C_vals)
    for idx, C in np.ndenumerate(C_vals):
        C = float(max(C, 0.0))
        if C <= 0.0:
            mean_F = alpha * max(I, 0.0) ** (1.0 / alpha)
        else:
            u = np.sqrt(2.0 * C) * gh_x
            F = alpha * np.clip(I + u, 0.0, 1e12) ** (1.0 / alpha)
            mean_F = float(np.dot(gh_w, F) / np.sqrt(np.pi))
        D_shot = sigma**2 * rho_phase * mean_F
        out[idx] = 0.5 * beta * D_shot
    return out


def plot_phase_shot_fixed_point_diagnostic(
    I=1.0,
    beta=1.0,
    alpha_vals=(0.5, 1.0, 2.0, 3.0),
    g_vals=(0.8, 1.0, 1.2, 1.5),
    C_min=0.1,
    C_max=80.0,
    n_C=400,
    plot_dir=None,
):
    """Visualize when shot-noise self-consistency ceases to be contractive.

    Plots S(C)/C where S(C)=beta*D_shot(C)/2.  The ratio diverges trivially at
    tiny C when baseline firing is nonzero, so the plot starts at C_min and
    clips the y-axis to focus on large-variance contractivity.
    """
    import os

    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    C_vals = np.linspace(C_min, C_max, n_C)
    n = len(alpha_vals)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.4 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    for ax, alpha in zip(axes_flat, alpha_vals):
        sigma_c = _phase_sigma_c(I, alpha)
        for g_val in g_vals:
            sigma = g_val * sigma_c
            S = phase_shot_fixed_point_map(
                C_vals, I=I, alpha=alpha, sigma=sigma, beta=beta,
            )
            ratio = S / C_vals
            ax.plot(C_vals, ratio, lw=1.8,
                    label=fr"$g={g_val:g}$, tail={ratio[-1]:.2f}")

        if alpha < 1.0:
            category = "superlinear: separate regime"
        elif alpha == 1.0:
            category = "linear-growth boundary"
        else:
            category = "sublinear"
        ax.axhline(1.0, color="k", ls=":", lw=1.2)
        ax.set(
            xlabel=r"$C_{11}(0)$",
            ylabel=r"$[\beta D_{\rm shot}(C)/2]/C$",
            title=fr"$\alpha={alpha:g}$  ({category})",
            xlim=(C_min, C_max),
            ylim=(0, 2.5),
        )
        ax.legend(fontsize=8)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle("Phase shot-noise self-consistency diagnostic", fontsize=13, fontweight="bold")
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_shot_fixed_point_diagnostic.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


def phase_branch_exists(
    sigma,
    I=1.0,
    alpha=1.0,
    beta=1.0,
    theory_kwargs=None,
):
    """Return (exists, C0) for the operational finite-branch criterion."""
    kwargs = {} if theory_kwargs is None else dict(theory_kwargs)
    kwargs.setdefault("solver", "inflated_ic")
    kwargs.setdefault("q_method", "hermite")
    kwargs.setdefault("n_quad", 48)
    kwargs.setdefault("hermite_order", 32)
    kwargs.setdefault("kernel_omega", 2.0)
    kwargs.setdefault("kernel_damping", 1.0)
    kwargs.setdefault("tau_max", 1.0)
    kwargs.setdefault("dtau", 0.1)
    kwargs.setdefault("warn_on_no_branch", False)
    try:
        _tau, C, _sigma_smooth = theory_phase_autocorr(
            I=I,
            alpha=alpha,
            sigma=float(sigma),
            beta=beta,
            **kwargs,
        )
    except Exception:
        return False, np.nan
    if len(C) == 0 or not np.isfinite(C[0]) or C[0] <= 1e-10:
        return False, np.nan
    return True, float(C[0])


def phase_operational_criticality(
    I=1.0,
    alpha=1.0,
    beta=1.0,
    g_bounds=(0.05, 2.0),
    n_scan=40,
    tol=1e-3,
    theory_kwargs=None,
):
    """
    Estimate the branch-existence criticality for the chosen closure.

    Returns a dict with:
      sigma_smooth : analytic smooth-feedback threshold
      sigma_branch : largest sigma with a finite stationary branch
      g_branch     : sigma_branch / sigma_smooth

    This is an operational threshold, not a universal bifurcation theorem.  It
    depends on the selected approximation in theory_kwargs.
    """
    sigma_smooth = _phase_sigma_c(I, alpha)
    g_lo, g_hi = map(float, g_bounds)
    g_grid = np.linspace(g_lo, g_hi, int(max(3, n_scan)))
    exists = []
    C0_vals = []
    for g_val in g_grid:
        ok, C0 = phase_branch_exists(
            g_val * sigma_smooth,
            I=I,
            alpha=alpha,
            beta=beta,
            theory_kwargs=theory_kwargs,
        )
        exists.append(ok)
        C0_vals.append(C0)

    exists = np.asarray(exists, dtype=bool)
    C0_vals = np.asarray(C0_vals, dtype=float)
    if not np.any(exists):
        return dict(
            sigma_smooth=sigma_smooth,
            sigma_branch=np.nan,
            g_branch=np.nan,
            g_grid=g_grid,
            exists=exists,
            C0=C0_vals,
        )

    last_ok_idx = int(np.where(exists)[0][-1])
    if last_ok_idx == len(g_grid) - 1:
        g_branch = float(g_grid[-1])
        branch_censored = True
    else:
        branch_censored = False
        lo = float(g_grid[last_ok_idx])
        hi = float(g_grid[last_ok_idx + 1])
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            ok, _ = phase_branch_exists(
                mid * sigma_smooth,
                I=I,
                alpha=alpha,
                beta=beta,
                theory_kwargs=theory_kwargs,
            )
            if ok:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        g_branch = lo

    return dict(
        sigma_smooth=sigma_smooth,
        sigma_branch=g_branch * sigma_smooth,
        g_branch=g_branch,
        branch_censored=branch_censored,
        g_grid=g_grid,
        exists=exists,
        C0=C0_vals,
    )


def plot_phase_operational_criticality(
    I=1.0,
    beta=1.0,
    alpha_vals=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    g_bounds=(0.05, 2.0),
    n_scan=36,
    theory_kwargs=None,
    plot_dir=None,
):
    """Plot smooth vs branch-existence criticality for the phase closure."""
    import os

    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    rows = []
    for alpha in alpha_vals:
        print(f"  estimating operational criticality for alpha={alpha:g}")
        res = phase_operational_criticality(
            I=I,
            alpha=alpha,
            beta=beta,
            g_bounds=g_bounds,
            n_scan=n_scan,
            theory_kwargs=theory_kwargs,
        )
        rows.append((alpha, res))
        print(
            f"    sigma_smooth={res['sigma_smooth']:.4g}, "
            f"sigma_branch={res['sigma_branch']:.4g}, "
            f"g_branch={res['g_branch']:.4g}"
        )

    alphas = np.array([r[0] for r in rows], dtype=float)
    sigma_smooth = np.array([r[1]["sigma_smooth"] for r in rows], dtype=float)
    sigma_branch = np.array([r[1]["sigma_branch"] for r in rows], dtype=float)
    g_branch = np.array([r[1]["g_branch"] for r in rows], dtype=float)
    censored = np.array([r[1].get("branch_censored", False) for r in rows], dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(alphas, sigma_smooth, "ko-", lw=1.8, label=r"smooth $\sigma_c$")
    axes[0].plot(alphas[~censored], sigma_branch[~censored], "ro-", lw=1.8, label=r"branch $\sigma_\ast$")
    if np.any(censored):
        axes[0].plot(alphas[censored], sigma_branch[censored], "r^", ms=8,
                     label=r"branch $\sigma_\ast$ lower bound")
    axes[0].set(xlabel=r"$\alpha$", ylabel=r"critical $\sigma$")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(alphas[~censored], g_branch[~censored], "bo-", lw=1.8)
    if np.any(censored):
        axes[1].plot(alphas[censored], g_branch[censored], "b^", ms=8)
    axes[1].axhline(1.0, color="k", ls=":", lw=1.2)
    axes[1].set(
        xlabel=r"$\alpha$",
        ylabel=r"$g_\ast=\sigma_\ast/\sigma_c^{\rm smooth}$",
        ylim=(0, max(1.2, np.nanmax(g_branch) * 1.15 if np.any(np.isfinite(g_branch)) else 1.2)),
    )
    axes[1].grid(alpha=0.25)

    plt.suptitle("Phase operational criticality: finite stationary branch", fontsize=13, fontweight="bold")
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_operational_criticality.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")
    return rows


def theory_phase_spike_autocorr(
    I=1.0,
    alpha=1.0,
    sigma=7.0,
    beta=1.0,
    C_uu=None,
    tau_max=50,
    dtau=0.01,
    n_quad=24,
):
    """
    Phase-model spike autocorrelation from the phi-rate closure.

    Uses
        C_spk(tau) = rho^2 * E[F(u(0)) F(u(tau))], rho = 1/(2*pi),
    where (u(0), u(tau)) is Gaussian with covariance from C_uu.
    """
    rho = 1.0 / (2.0 * np.pi)

    def F(u):
        return alpha * np.clip(I + u, 0.0, 1e12) ** (1.0 / alpha)

    if C_uu is None:
        tau, C_uu, sigma_c = theory_phase_autocorr(
            I=I,
            alpha=alpha,
            sigma=sigma,
            beta=beta,
            tau_max=tau_max,
            dtau=dtau,
            n_quad=n_quad,
        )
    else:
        tau = np.arange(len(C_uu)) * dtau
        Fprime0 = float(np.maximum(I, 1e-10) ** (1.0 / alpha - 1.0))
        sigma_c = 1.0 / (rho * Fprime0)

    C0 = float(C_uu[0]) if len(C_uu) > 0 else 0.0
    if C0 <= 0:
        return tau, np.zeros_like(tau), sigma_c

    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    gh_w2 = np.outer(gh_w, gh_w)
    scale = np.sqrt(2.0 * C0)

    def pair_expectation(C_tau):
        rho_tau = float(np.clip(C_tau / C0, -0.999999, 0.999999))
        x = scale * gh_x[:, None]
        y = scale * (
            rho_tau * gh_x[:, None] + np.sqrt(1.0 - rho_tau**2) * gh_x[None, :]
        )
        return float(np.sum(gh_w2 * F(x) * F(y)) / np.pi)

    C_spk_raw = rho**2 * np.array([pair_expectation(c_tau) for c_tau in C_uu])

    # Centered correlation: Cov[r(0), r(tau)] = E[r0 r_tau] - E[r]^2.
    u_1d = scale * gh_x
    mean_rate = rho * float(np.dot(gh_w, F(u_1d)) / np.sqrt(np.pi))
    C_spk = C_spk_raw - mean_rate**2
    return tau, C_spk, sigma_c


def plot_phase_spike_correlation(
    I=1.0,
    alpha=1.0,
    beta=1.0,
    sigma=None,
    N=512,
    T=8000.0,
    dt=0.02,
    dtau=0.02,
    tau_max=120.0,
    plot_dir=None,
):
    """Compare phase-model spike autocorrelation from simulation and theory."""
    import os
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    _, _, sigma_c = theory_phase_autocorr(
        I=I,
        alpha=alpha,
        sigma=1.0,
        beta=beta,
        tau_max=5,
        dtau=dtau,
    )
    if sigma is None:
        sigma = 0.6 * sigma_c

    tau_s, _Cuu_s, Cspk_s = sim_phase_network(
        N=N,
        I=I,
        alpha=alpha,
        sigma=sigma,
        beta=beta,
        T=T,
        dt=dt,
        tau_max=tau_max,
        return_spike=True,
    )
    tau_t, Cspk_t, _ = theory_phase_spike_autocorr(
        I=I,
        alpha=alpha,
        sigma=sigma,
        beta=beta,
        tau_max=max(float(tau_s[-1]), tau_max, 1.0),
        dtau=dtau,
    )

    # Remove the zero-lag bin to avoid the finite-dt point-process spike peak.
    tau_s_plot, Cspk_s_plot = tau_s[1:], Cspk_s[1:]
    tau_t_plot, Cspk_t_plot = tau_t[1:], Cspk_t[1:]

    def norm(x):
        x0 = float(x[0]) if len(x) > 0 else 1.0
        return x / x0 if abs(x0) > 1e-12 else x

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tau_s_plot, norm(Cspk_s_plot), lw=1.8, label="Simulation")
    ax.plot(tau_t_plot, norm(Cspk_t_plot), "--", lw=2.2, label="Theory (phi-rate)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$C_{spk}(\tau) / C_{spk}(0^+)$")
    ax.set_title(
        f"Phase spike correlation: sigma={sigma:.2f} (sigma_c={sigma_c:.2f})"
    )
    ax.set_xlim(0, 120)
    ax.legend()
    fig.tight_layout()
    import os
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "phase_spike_correlation_test.png"), dpi=150)
    print(f"Saved to {os.path.join(plot_dir, 'phase_spike_correlation_test.png')}")
    plt.close("all")


def plot_phase_network(
    I=1.0,
    alpha=1.0,
    beta=1.0,
    N=512,
    sigma_vals=None,
    T=8000.0,
    dt=0.02,
    dtau=0.02,
    tau_max=120.0,
    sim_reps=3,
    plot_dir=None,
    theory_kwargs=None,
):
    """Compare phase-network simulation and reduced-theory autocorrelations."""
    import os
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    rho = 1.0 / (2.0 * np.pi)
    Fprime0 = float(np.maximum(I, 1e-10) ** (1.0 / alpha - 1.0))
    sigma_c = 1.0 / (rho * Fprime0)
    print(f"Phase model: I={I}, alpha={alpha}, sigma_c={sigma_c:.3f}")
    theory_kwargs = {} if theory_kwargs is None else dict(theory_kwargs)
    omega_label = float(theory_kwargs.get("kernel_omega", 0.0))
    damping_label = float(theory_kwargs.get("kernel_damping", 0.0))
    scaled_label = bool(theory_kwargs.get("kernel_scaled_by_beta", True))
    solver_label = str(theory_kwargs.get("solver", "fd"))
    theory_label = "2PI shot-kick"
    if solver_label.lower() in ("inflated_ic", "inflated-ic", "inflated"):
        theory_label = "2PI inflated IC"
    if abs(omega_label) > 0.0 or abs(damping_label) > 0.0:
        suffix = r"\times\beta" if scaled_label else ""
        theory_label = fr"{theory_label} ($\omega={omega_label:g}{suffix}$, $\gamma={damping_label:g}{suffix}$)"

    if sigma_vals is None:
        sigma_vals = [0.75 * sigma_c, 0.95 * sigma_c, 1.1 * sigma_c]

    fig, axes = plt.subplots(1, len(sigma_vals), figsize=(5 * len(sigma_vals), 4))
    if len(sigma_vals) == 1:
        axes = [axes]

    for ax, sigma in zip(axes, sigma_vals):
        g_val = sigma / sigma_c
        print(f"\n-- phase sigma={sigma:.3f} (g={g_val:.2f}) --")

        try:
            tau_th, C_th, _ = theory_phase_autocorr(
                I=I,
                alpha=alpha,
                sigma=sigma,
                beta=beta,
                tau_max=tau_max,
                dtau=dtau,
                **theory_kwargs,
            )
        except Exception as e:
            print(f"  theory failed: {e}")
            tau_th, C_th = None, None

        C_runs = []
        tau_s = None
        for _ in range(max(1, int(sim_reps))):
            tau_run, C_run = sim_phase_network(
                N=N,
                I=I,
                alpha=alpha,
                sigma=sigma,
                beta=beta,
                T=T,
                dt=dt,
                tau_max=tau_max,
                n_probe=N,
            )
            tau_s = tau_run
            C_runs.append(C_run)
        C_s = np.mean(C_runs, axis=0)

        C_s_norm = C_s / C_s[0] if C_s[0] > 0 else C_s
        has_theory = C_th is not None and C_th[0] > 1e-2
        if has_theory:
            C_th_norm = C_th / C_th[0]
            ax.plot(tau_th, C_th_norm, "r--", lw=2, label=theory_label, zorder=3)
        ax.plot(tau_s, C_s_norm, "b", lw=1.5, alpha=0.85, label=f"Sim ({sim_reps} runs)", zorder=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(
            xlabel=r"$\tau$",
            ylabel=r"$C_{uu}(\tau) / C_{uu}(0)$",
            ylim=(-0.2, 1.05),
            title=fr"$\sigma={sigma:.2f}$, $g={g_val:.2f}$",
            xlim=(0, tau_max),
        )
        ax.legend(fontsize=8)

    plt.suptitle(
        fr"Phase neuron network: $I={I}$, $\alpha={alpha}$, $\sigma_c={sigma_c:.2f}$",
        fontsize=13,
        fontweight="bold",
    )
    import os
    os.makedirs(plot_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "phase_network_test.png"), dpi=150)
    print(f"Saved to {os.path.join(plot_dir, 'phase_network_test.png')}")
    plt.close("all")


def plot_phase_theory_comparison(
    I=1.0,
    alpha=1.0,
    beta=1.0,
    sigma=None,
    N=256,
    T=900.0,
    dt=0.02,
    dtau=0.1,
    tau_max=35.0,
    sim_reps=2,
    plot_dir=None,
    theory_variants=None,
):
    """Overlay simulation with several approximate phase-network theories."""
    import os

    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    sigma_c = _phase_sigma_c(I, alpha)
    if sigma is None:
        sigma = 1.3 * sigma_c

    if theory_variants is None:
        theory_variants = [
            dict(
                label="shot-kick FD",
                kwargs=dict(solver="fd", n_quad=10, fd_max_nfev=140),
                style=dict(color="C3", ls="--", lw=2.0),
            ),
            dict(
                label=r"gen. kernel FD ($\omega=2,\gamma=1$)",
                kwargs=dict(
                    solver="fd", kernel_omega=2.0, kernel_damping=1.0,
                    n_quad=10, fd_max_nfev=140,
                ),
                style=dict(color="C1", ls="-.", lw=2.0),
            ),
            dict(
                label="inflated IC",
                kwargs=dict(solver="inflated_ic", n_quad=10),
                style=dict(color="C2", ls=":", lw=2.2),
            ),
            dict(
                label=r"inflated IC + gen. kernel",
                kwargs=dict(
                    solver="inflated_ic", kernel_omega=2.0, kernel_damping=1.0,
                    n_quad=10,
                ),
                style=dict(color="C4", ls=(0, (5, 2)), lw=2.0),
            ),
        ]

    C_runs = []
    tau_s = None
    for _ in range(max(1, int(sim_reps))):
        tau_run, C_run = sim_phase_network(
            N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
            T=T, dt=dt, tau_max=tau_max, n_probe=N,
        )
        tau_s = tau_run
        C_runs.append(C_run)
    C_s = np.mean(C_runs, axis=0)
    C_s_norm = C_s / C_s[0] if C_s[0] > 0 else C_s

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(tau_s, C_s_norm, color="C0", lw=1.8, label=f"Sim ({sim_reps} runs)")

    for variant in theory_variants:
        kwargs = dict(variant.get("kwargs", {}))
        style = dict(variant.get("style", {}))
        label = variant.get("label", kwargs.get("solver", "theory"))
        try:
            tau_th, C_th, _ = theory_phase_autocorr(
                I=I, alpha=alpha, sigma=sigma, beta=beta,
                tau_max=tau_max, dtau=dtau, **kwargs,
            )
            norm = max(abs(C_th[0]), 1e-12)
            ax.plot(tau_th, C_th / norm, label=label, **style)
        except Exception as err:
            print(f"  theory variant failed ({label}): {err}")

    ax.axhline(0, color="k", lw=0.5)
    ax.set(
        xlabel=r"$\tau$",
        ylabel=r"$C_{uu}(\tau) / C_{uu}(0)$",
        title=fr"Phase theory comparison: $\sigma={sigma:.2f}$, $g={sigma/sigma_c:.2f}$",
        xlim=(0, tau_max),
        ylim=(-0.35, 1.1),
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_theory_comparison.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


def plot_phase_theory_examples(
    examples=None,
    N=192,
    T=700.0,
    dt=0.02,
    dtau=0.12,
    tau_max=30.0,
    sim_reps=1,
    plot_dir=None,
    theory_variants=None,
):
    """Grid of phase-network examples comparing simulation to theory variants."""
    import os

    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if examples is None:
        examples = []
        for g_val in (1.1, 1.3, 1.5):
            sc = _phase_sigma_c(1.0, 1.0)
            examples.append(dict(I=1.0, alpha=1.0, beta=1.0, sigma=g_val * sc,
                                 label=fr"$\alpha=1,\ \beta=1,\ g={g_val:.1f}$"))
        sc = _phase_sigma_c(1.0, 2.0)
        examples.append(dict(I=1.0, alpha=2.0, beta=1.0, sigma=1.3 * sc,
                             label=fr"$\alpha=2,\ \beta=1,\ g=1.3$"))
        sc = _phase_sigma_c(1.0, 0.5)
        examples.append(dict(I=1.0, alpha=0.5, beta=1.0, sigma=1.3 * sc,
                             label=fr"$\alpha=0.5,\ \beta=1,\ g=1.3$"))
        sc = _phase_sigma_c(1.0, 1.0)
        examples.append(dict(I=1.0, alpha=1.0, beta=0.5, sigma=1.3 * sc,
                             label=fr"$\alpha=1,\ \beta=0.5,\ g=1.3$"))

    if theory_variants is None:
        theory_variants = [
            dict(
                label="shot-kick FD",
                kwargs=dict(solver="fd", q_method="qmc", n_qmc=1024, fd_max_nfev=120),
                style=dict(color="C3", ls="--", lw=1.7),
            ),
            dict(
                label=r"gen. FD",
                kwargs=dict(
                    solver="fd", kernel_omega=2.0, kernel_damping=1.0,
                    q_method="qmc", n_qmc=1024, fd_max_nfev=120,
                ),
                style=dict(color="C1", ls="-.", lw=1.8),
            ),
            dict(
                label=r"infl. IC + gen.",
                kwargs=dict(
                    solver="inflated_ic", kernel_omega=2.0, kernel_damping=1.0,
                    q_method="qmc", n_qmc=1024,
                ),
                style=dict(color="C4", ls=(0, (5, 2)), lw=1.8),
            ),
        ]

    n = len(examples)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    for ax, ex in zip(axes_flat, examples):
        I = ex.get("I", 1.0)
        alpha = ex.get("alpha", 1.0)
        beta = ex.get("beta", 1.0)
        sigma = ex["sigma"]
        sigma_c = _phase_sigma_c(I, alpha)
        label = ex.get("label", fr"$g={sigma/sigma_c:.2f}$")
        print(f"  example {label}: sigma={sigma:.3f}, sigma/sigma_c={sigma/sigma_c:.2f}")

        C_runs = []
        tau_s = None
        for _ in range(max(1, int(sim_reps))):
            tau_run, C_run = sim_phase_network(
                N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
                T=T, dt=dt, tau_max=tau_max, n_probe=N,
            )
            tau_s = tau_run
            C_runs.append(C_run)
        C_s = np.mean(C_runs, axis=0)
        C_s_norm = C_s / C_s[0] if C_s[0] > 0 else C_s
        ax.plot(tau_s, C_s_norm, color="C0", lw=1.5, label=f"Sim")

        for variant in theory_variants:
            kwargs = dict(variant.get("kwargs", {}))
            style = dict(variant.get("style", {}))
            vlabel = variant.get("label", kwargs.get("solver", "theory"))
            try:
                tau_th, C_th, _ = theory_phase_autocorr(
                    I=I, alpha=alpha, sigma=sigma, beta=beta,
                    tau_max=tau_max, dtau=dtau, **kwargs,
                )
                norm = max(abs(C_th[0]), 1e-12)
                ax.plot(tau_th, C_th / norm, label=vlabel, **style)
            except Exception as err:
                print(f"    theory variant failed ({vlabel}): {err}")

        ax.axhline(0, color="k", lw=0.5)
        ax.set(
            xlabel=r"$\tau$",
            ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
            title=label,
            xlim=(0, tau_max),
            ylim=(-0.35, 1.1),
        )
        ax.legend(fontsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle("Phase network: theory variants across parameters", fontsize=13, fontweight="bold")
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_theory_examples.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


def plot_phase_beta_scaling_diagnostic(
    beta_vals=(0.5, 1.0, 2.0),
    I=1.0,
    alpha=1.0,
    g_val=1.3,
    N=192,
    T=700.0,
    dt=0.02,
    dtau=0.12,
    tau_max=25.0,
    sim_reps=1,
    plot_dir=None,
    theory_kwargs=None,
):
    """Check whether phase theory and simulation scale consistently with beta."""
    import os

    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if theory_kwargs is None:
        theory_kwargs = dict(
            solver="fd",
            kernel_omega=2.0,
            kernel_damping=1.0,
            kernel_scaled_by_beta=True,
            q_method="qmc",
            n_qmc=1024,
            fd_max_nfev=140,
        )
    else:
        theory_kwargs = dict(theory_kwargs)

    sigma_c = _phase_sigma_c(I, alpha)
    sigma = g_val * sigma_c

    n = len(beta_vals)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.0))
    if n == 1:
        axes = [axes]

    for ax, beta in zip(axes, beta_vals):
        print(f"  beta scaling: beta={beta:g}, sigma={sigma:.3f}, g={g_val:.2f}")
        C_runs = []
        tau_s = None
        for _ in range(max(1, int(sim_reps))):
            tau_run, C_run = sim_phase_network(
                N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
                T=T, dt=dt, tau_max=tau_max, n_probe=N,
            )
            tau_s = tau_run
            C_runs.append(C_run)
        C_s = np.mean(C_runs, axis=0)
        ax.plot(beta * tau_s, C_s / max(abs(C_s[0]), 1e-12),
                color="C0", lw=1.6, label="Sim")

        try:
            tau_th, C_th, _ = theory_phase_autocorr(
                I=I, alpha=alpha, sigma=sigma, beta=beta,
                tau_max=tau_max, dtau=dtau, **theory_kwargs,
            )
            ax.plot(beta * tau_th, C_th / max(abs(C_th[0]), 1e-12),
                    color="C1", ls="--", lw=2.0, label="Theory")
        except Exception as err:
            print(f"    theory failed for beta={beta:g}: {err}")

        ax.axhline(0, color="k", lw=0.5)
        ax.set(
            xlabel=r"$\beta\tau$",
            ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
            title=fr"$\beta={beta:g}$",
            xlim=(0, beta * tau_max),
            ylim=(-0.35, 1.1),
        )
        ax.legend(fontsize=8)

    plt.suptitle(
        fr"Phase beta-scaling diagnostic: $\alpha={alpha}$, $g={g_val}$",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_beta_scaling_diagnostic.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


# -----------------------------------------------------------------------------
# 2. BINARY NEURON NETWORK
#    n_i in {0,1}, 0->1 at rate f(u_i), 1->0 at rate mu
#    u_i: synaptic drive, Gaussian weights, row-sum corrected
#
#    Theory: C_nn(tau) = A+ exp(-kappa+ |tau|) + A- exp(-kappa- |tau|)
# -----------------------------------------------------------------------------

def sim_binary_network(
    N=1000,
    sigma=0.8,
    beta=1.0,
    mu=1.0,
    f0=0.5,
    f1=1.0,
    T=5000.0,
    dt=0.02,
    lam=1,
    burn=500,
    clip_rate_on=True,
    method="tau-leap",
    return_spike=False,
    rng=rng,
):
    """
    Binary network simulation.

    f(u) = f0 + f1*u (linear gain).
    Set clip_rate_on=True to use max(f0 + f1*u, 0) for a biophysical variant.
    State: n_i in {0,1}, u_i continuous.

    method:
      - "tau-leap": parallel Bernoulli updates each dt (fast, approximate)
      - "ssa" or "gibson-bruck": event-driven SSA for binary flips
        with exact between-event u integration (more accurate, often slower)
    """
    method = method.lower()
    if method == "gibson-bruck":
        # For this dense-coupled model, we use direct SSA semantics.
        method = "ssa"
    if method not in ("tau-leap", "ssa"):
        raise ValueError("method must be one of: 'tau-leap', 'ssa', 'gibson-bruck'")

    W = make_weights(N, sigma, lam, rng)
    n = rng.integers(0, 2, N).astype(float)
    u = np.zeros(N)

    def rate_on(u_):
        rates = f0 + f1 * u_
        if clip_rate_on:
            return np.maximum(rates, 0.0)
        if np.any(rates < 0):
            min_rate = float(np.min(rates))
            raise ValueError(
                f"Unclipped linear-gain model produced negative on-rates (min={min_rate:.3e}). "
                "Increase f0, decrease f1/sigma, or use clip_rate_on=True."
            )
        return rates

    if return_spike and method != "tau-leap":
        raise ValueError("return_spike=True is currently supported only for method='tau-leap'.")

    nt = int(T / dt)
    N_rec = np.zeros((nt, N))
    U_rec = np.zeros((nt, N))
    SPK_rec = np.zeros((nt, N)) if return_spike else None

    if method == "tau-leap":
        nb = int(burn / dt)
        for _ in range(nb):
            r_on = rate_on(u) * (1 - n)
            r_off = mu * n
            p_on = 1.0 - np.exp(-r_on * dt)
            p_off = 1.0 - np.exp(-r_off * dt)
            flip_on = rng.random(N) < p_on
            flip_off = rng.random(N) < p_off
            n += flip_on.astype(float) - flip_off.astype(float)
            n = np.clip(n, 0, 1)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                drive = W @ n
            if not np.all(np.isfinite(drive)):
                drive = np.nan_to_num(drive, nan=0.0, posinf=1e6, neginf=-1e6)
            u += dt * (-beta * u + beta * drive)
            if not np.all(np.isfinite(u)):
                u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)

        for t in range(nt):
            r_on = rate_on(u) * (1 - n)
            r_off = mu * n
            p_on = 1.0 - np.exp(-r_on * dt)
            p_off = 1.0 - np.exp(-r_off * dt)
            flip_on = rng.random(N) < p_on
            flip_off = rng.random(N) < p_off
            n += flip_on.astype(float) - flip_off.astype(float)
            n = np.clip(n, 0, 1)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                drive = W @ n
            if not np.all(np.isfinite(drive)):
                drive = np.nan_to_num(drive, nan=0.0, posinf=1e6, neginf=-1e6)
            u += dt * (-beta * u + beta * drive)
            if not np.all(np.isfinite(u)):
                u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
            N_rec[t] = n
            U_rec[t] = u
            if return_spike:
                SPK_rec[t] = flip_on.astype(float)
    else:
        # Event-driven SSA for binary transitions, with exact between-event u evolution.
        t = 0.0
        t_end = burn + T
        next_sample = burn
        rec_idx = 0

        def advance_u(delta_t):
            nonlocal u
            if delta_t <= 0:
                return
            drive = W @ n
            if beta > 0:
                decay = np.exp(-beta * delta_t)
                u = u * decay + drive * (1.0 - decay)
            else:
                u = u + delta_t * drive

        while t < t_end and rec_idx < nt:
            r_on = rate_on(u) * (1 - n)
            r_off = mu * n
            a = np.concatenate([r_on, r_off])
            a0 = float(np.sum(a))

            if a0 <= 0:
                # No more stochastic events; only deterministic u drift to sampling grid.
                while next_sample <= t_end and rec_idx < nt:
                    advance_u(next_sample - t)
                    t = next_sample
                    N_rec[rec_idx] = n
                    U_rec[rec_idx] = u
                    rec_idx += 1
                    next_sample += dt
                break

            tau_event = -np.log(max(rng.random(), 1e-15)) / a0
            t_event = t + tau_event
            t_stop = min(t_event, t_end)

            while next_sample <= t_stop and rec_idx < nt:
                advance_u(next_sample - t)
                t = next_sample
                N_rec[rec_idx] = n
                U_rec[rec_idx] = u
                rec_idx += 1
                next_sample += dt

            if t_event > t_end:
                break

            # Jump to event and fire one channel.
            advance_u(t_event - t)
            t = t_event
            r = rng.random() * a0
            evt = int(np.searchsorted(np.cumsum(a), r, side="right"))
            if evt < N:
                n[evt] = 1.0
            else:
                n[evt - N] = 0.0

    max_lag = int(30 / dt)
    Cnn = np.mean([autocorr(N_rec[:, i], max_lag) for i in range(min(N, 64))], axis=0)
    Cuu = np.mean([autocorr(U_rec[:, i], max_lag) for i in range(min(N, 64))], axis=0)
    tau = np.arange(len(Cnn)) * dt
    if not return_spike:
        return tau, Cnn, Cuu

    Cspk = np.mean(
        [autocorr(SPK_rec[:, i] / dt, max_lag) for i in range(min(N, 64))],
        axis=0,
    )
    return tau, Cnn, Cuu, Cspk


def theory_binary_autocorr(sigma, beta, mu, f0, f1, tau_max=30, dtau=0.001):
    """
    Exact 2PI theory for the binary network with linear gain f(u) = f0 + f1*u.

    Returns C_nn(tau) and C_uu(tau) from the closed-form expressions.

    Parameters
    ----------
    sigma  : weight std
    beta   : synaptic decay rate
    mu     : neuron off-rate
    f0, f1 : gain function coefficients
    """
    # Steady state from the linear-gain binary model.
    gamma = mu + f0
    n_bar = f0 / gamma
    c1 = f1 * mu / gamma
    D0 = 2 * n_bar * (1 - n_bar) * gamma
    g = c1 * sigma / gamma

    print(f"  gamma={gamma:.3f}, n_bar={n_bar:.3f}, c1={c1:.3f}, g={g:.3f}, D0={D0:.3f}")
    if g >= 1:
        print(f"  WARNING: g={g:.3f} >= 1, above transition; stationary theory branch becomes oscillatory.")

    # Pole locations
    disc = (gamma**2 - beta**2) ** 2 + 4 * c1**2 * beta**2 * sigma**2
    kp2 = 0.5 * ((gamma**2 + beta**2) + np.sqrt(disc))
    km2 = 0.5 * ((gamma**2 + beta**2) - np.sqrt(disc))

    tau = np.arange(0, tau_max, dtau)

    if km2 >= 0:  # g < 1: exact two-exponential branch
        kp, km = np.sqrt(kp2), np.sqrt(km2)
        Ap = D0 * 0.5 * (beta**2 - kp2) / (kp * (km2 - kp2))
        Am = D0 * 0.5 * (beta**2 - km2) / (km * (km2 - kp2))
        Cnn = Ap * np.exp(-kp * tau) + Am * np.exp(-km * tau)

        # Use the exact frequency-space relation:
        #   C_uu(w) = [beta^2 sigma^2 / (beta^2 + w^2)] C_nn(w)
        def uu_term(A, k):
            denom = beta**2 - k**2
            if abs(denom) < 1e-10 * max(1.0, beta**2):
                # L'Hopital limit as k -> beta.
                return A * beta * sigma**2 * tau * np.exp(-beta * tau)
            return A * beta**2 * sigma**2 / denom * (
                np.exp(-k * tau) - (k / beta) * np.exp(-beta * tau)
            )

        Cuu = uu_term(Ap, kp) + uu_term(Am, km)
    else:  # g > 1: complex-conjugate pole pair kp +/- i kr
        kp = np.sqrt(kp2)
        kr = np.sqrt(-km2)
        # Exact damped-oscillatory C_nn from the complex pole pair.
        Cnn = np.exp(-kp * tau) * (
            D0 / (2 * kp) * np.cos(kr * tau)
            + D0 * (beta**2 - kp2) / (4 * kp * kr) * np.sin(kr * tau)
        )

        # Use the residue-based complex extension of the subcritical C_uu formula,
        # then take the real part so C_uu remains real-valued.
        k_cmplx = kp + 1j * kr
        d_cmplx = beta**2 - k_cmplx**2
        if abs(d_cmplx) > 1e-14:
            term = (beta**2 * sigma**2 / d_cmplx) * (
                np.exp(-k_cmplx * tau) - (k_cmplx / beta) * np.exp(-beta * tau)
            )
            Cuu = 2 * np.real(
                D0 * 0.5 * (beta**2 - kp2 + 2j * kp * kr) / (4 * kp * kr) * term
            )
        else:
            Cuu = np.zeros_like(tau)

    return tau, Cnn, Cuu, g


def theory_binary_spike_autocorr(
    sigma,
    beta,
    mu,
    f0,
    f1,
    tau_max=30,
    dtau=0.001,
    n_quad=32,
):
    """
    Approximate binary-network spike autocorrelation using the quasi-static map.

    We map spike intensity to nu_eff(u)=f(u)^2/(f(u)+mu) with clipped
    f(u)=max(f0+f1*u,0), then compute
        C_spk(tau) ~ E[nu_eff(u(0)) nu_eff(u(tau))]
    under the Gaussian pair implied by theory C_uu.
    """
    tau, _Cnn, Cuu, g = theory_binary_autocorr(
        sigma=sigma,
        beta=beta,
        mu=mu,
        f0=f0,
        f1=f1,
        tau_max=tau_max,
        dtau=dtau,
    )
    Cuu0 = float(Cuu[0]) if len(Cuu) > 0 else 0.0
    if Cuu0 <= 0:
        return tau, np.zeros_like(tau), g

    # C_uu is already the u-covariance, so no extra sigma prefactor here.
    Cspk_raw = np.array(
        [
            _Q_clipped_correct(Cuu[k], Cuu0, f0, f1, mu, sigma, n_quad=n_quad)
            for k in range(len(tau))
        ]
    )
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    u = np.sqrt(Cuu0) * gh_x * np.sqrt(2.0)
    mean_rate = float(np.dot(gh_w, _nu_eff(u, f0, f1, mu)) / np.sqrt(np.pi))
    Cspk = Cspk_raw - mean_rate**2
    return tau, Cspk, g


def _nu_eff(u, f0, f1, mu):
    """
    Effective rate under quasi-static n(u): nu_eff(u) = f(u)^2 / (f(u)+mu).
    """
    fu = np.maximum(f0 + f1 * u, 0.0)
    return fu**2 / (fu + mu)


def _Q_clipped_correct(C_uu_tau, C_uu_0, f0, f1, mu, sigma, n_quad=32):
    """
    Q(tau) = E[nu_eff(u(0)) nu_eff(u(tau))] for a joint Gaussian (u0, utau).

    C_uu is already the covariance of the synaptic drive u.  The sigma argument
    is retained for backward-compatible call signatures but is not used here.
    """
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    gh_w2 = np.outer(gh_w, gh_w)
    if C_uu_0 <= 0:
        return float(_nu_eff(0.0, f0, f1, mu) ** 2)

    # C_uu is already the u-covariance, so no extra sigma prefactor here.
    s = np.sqrt(C_uu_0)
    rho = float(np.clip(C_uu_tau / C_uu_0, -0.999999, 0.999999))
    xi = gh_x[:, None] * np.sqrt(2.0)
    yi = gh_x[None, :] * np.sqrt(2.0)
    u0 = s * xi
    utau = s * (rho * xi + np.sqrt(1.0 - rho**2) * yi)
    vals = _nu_eff(u0, f0, f1, mu) * _nu_eff(utau, f0, f1, mu)
    return float(np.sum(gh_w2 * vals) / np.pi)


def _D0_clipped(C_uu_0, f0, f1, mu, sigma, n_quad=32):
    """
    Shot-noise amplitude D0 for clipped gain, averaged over Gaussian u.

    C_uu_0 is already the variance of u; do not multiply by sigma again.
    """
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    s = np.sqrt(max(C_uu_0, 0.0))
    u = s * gh_x * np.sqrt(2.0)
    fu = np.maximum(f0 + f1 * u, 0.0)
    integrand = 2.0 * fu * mu / (fu + mu)
    return float(np.dot(gh_w, integrand) / np.sqrt(np.pi))


def _binary_clipped_moments(C_uu_0, f0, f1, mu, n_quad=48):
    """One-time Gaussian expectations for clipped binary rates."""
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    u = np.sqrt(max(C_uu_0, 0.0)) * gh_x * np.sqrt(2.0)
    fu = np.maximum(f0 + f1 * u, 0.0)
    gamma_u = fu + mu
    p_u = fu / gamma_u
    w = gh_w / np.sqrt(np.pi)
    nbar = float(np.dot(w, p_u))
    var_cond = float(np.dot(w, p_u * (1.0 - p_u)))
    gamma_mean = float(np.dot(w, gamma_u))
    return nbar, var_cond, gamma_mean


def _normalized_hermite_coeffs(func, variance, n_quad=48, order=32):
    """
    Coefficients b_n = E[f(sZ) He_n(Z)/sqrt(n!)] for Z~N(0,1).

    The covariance of f(sZ_0), f(sZ_tau) under a Gaussian pair with correlation
    rho is sum_{n>=1} b_n^2 rho^n.  This gives a stable centered covariance
    without subtracting the squared mean.
    """
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    z = np.sqrt(2.0) * gh_x
    vals = func(np.sqrt(max(float(variance), 0.0)) * z)
    w = gh_w / np.sqrt(np.pi)
    order = int(max(1, order))
    coeffs = np.zeros(order + 1)
    phi_nm1 = np.ones_like(z)
    coeffs[0] = float(np.dot(w, vals * phi_nm1))
    if order >= 1:
        phi_n = z
        coeffs[1] = float(np.dot(w, vals * phi_n))
        for n in range(1, order):
            phi_np1 = (z * phi_n - np.sqrt(n) * phi_nm1) / np.sqrt(n + 1.0)
            coeffs[n + 1] = float(np.dot(w, vals * phi_np1))
            phi_nm1, phi_n = phi_n, phi_np1
    return coeffs


def _hermite_covariance_from_coeffs(C_tau, C0, coeffs):
    if C0 <= 0.0:
        return 0.0
    rho = float(np.clip(C_tau / C0, -0.999999, 0.999999))
    powers = rho ** np.arange(1, len(coeffs))
    return float(np.dot(coeffs[1:] ** 2, powers))


def _binary_clipped_nbar_cov(
    C_uu_tau,
    C_uu_0,
    f0,
    f1,
    mu,
    n_quad=48,
    q_method="gh",
    hermite_order=32,
    coeffs=None,
):
    """Cov[nbar(u0), nbar(utau)] for clipped gain under a Gaussian u pair."""
    if C_uu_0 <= 0.0:
        return 0.0
    if str(q_method).lower() in ("hermite", "hermite-series", "series"):
        if coeffs is None:
            def p_of_u(u):
                fu = np.maximum(f0 + f1 * u, 0.0)
                return fu / (fu + mu)
            coeffs = _normalized_hermite_coeffs(
                p_of_u, C_uu_0, n_quad=n_quad, order=hermite_order
            )
        return _hermite_covariance_from_coeffs(C_uu_tau, C_uu_0, coeffs)

    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    gh_w2 = np.outer(gh_w, gh_w)
    rho = float(np.clip(C_uu_tau / C_uu_0, -0.999999, 0.999999))
    s = np.sqrt(2.0 * C_uu_0)
    u0 = s * gh_x[:, None]
    ut = s * (rho * gh_x[:, None] + np.sqrt(1.0 - rho**2) * gh_x[None, :])
    f0_u = np.maximum(f0 + f1 * u0, 0.0)
    ft_u = np.maximum(f0 + f1 * ut, 0.0)
    p0 = f0_u / (f0_u + mu)
    pt = ft_u / (ft_u + mu)
    raw = float(np.sum(gh_w2 * p0 * pt) / np.pi)

    u_1d = s * gh_x
    f_1d = np.maximum(f0 + f1 * u_1d, 0.0)
    p_1d = f_1d / (f_1d + mu)
    mean = float(np.dot(gh_w, p_1d) / np.sqrt(np.pi))
    return raw - mean**2


def _binary_filter_even_cov(tau, C_source, beta, sigma, pad_factor=2):
    """
    Filter an even source covariance through beta*exp(-beta t)H(t).

    The source covariance is supplied for tau >= 0.  We mirror it to reduce the
    one-sided FFT artifact, apply beta^2*sigma^2/(beta^2+w^2), then return the
    tau >= 0 half.
    """
    C_source = np.asarray(C_source, dtype=float)
    if len(C_source) < 2:
        return sigma**2 * C_source.copy()
    dt = float(tau[1] - tau[0])
    even = np.concatenate([C_source, C_source[-2:0:-1]])
    n_even = len(even)
    n_fft = int(2 ** np.ceil(np.log2(max(n_even, pad_factor * n_even))))
    spec = fft(even, n=n_fft)
    omega = fftfreq(n_fft, d=dt) * 2.0 * np.pi
    filt = beta**2 * sigma**2 / (beta**2 + omega**2)
    filtered = np.real(ifft(spec * filt, n=n_fft))[:n_even]
    return filtered[:len(C_source)]


def theory_binary_clipped_integral(
    sigma,
    beta,
    mu,
    f0,
    f1,
    tau_max=30,
    dtau=0.05,
    n_quad=48,
    max_iter=160,
    tol=1e-5,
    mix=0.25,
    intrinsic="telegraph",
    init="linear",
    q_method="hermite",
    hermite_order=32,
):
    """
    Integral Gaussian-closure theory for clipped binary neurons.

    This option avoids the effective-linear coefficients used by
    theory_binary_clipped.  Given a candidate C_uu(tau), it computes

        C_nn_smooth(tau) = Cov[p(u(0)), p(u(tau))]

    by direct 2-D quadrature, where p(u)=f_+(u)/(f_+(u)+mu).  It can also add a
    local intrinsic binary-state covariance.  This is still a closure, but the
    clipping nonlinearity itself is handled by numerical integration.

    q_method:
        "hermite" -> 1-D Hermite expansion for Cov[p(u0),p(utau)]
        "gh"      -> direct 2-D Gauss-Hermite product quadrature

    intrinsic:
        "telegraph"  -> add E[p(1-p)] exp(-E[gamma] tau)
        "white"      -> add D0/(2 gamma_eff) exp(-gamma_eff tau)
        "none"       -> no intrinsic binary switching term
    """
    ntau = int(tau_max / dtau)
    tau = np.arange(ntau) * dtau
    if len(tau) == 0:
        return tau, np.array([]), np.array([]), np.nan

    if init == "linear":
        try:
            _, _Cnn_lin, Cuu_lin, _ = theory_binary_autocorr(
                sigma, beta, mu, f0, f1, tau_max=tau_max, dtau=dtau
            )
            C_uu = np.maximum(np.interp(tau, np.arange(len(Cuu_lin)) * dtau, Cuu_lin), 0.0)
        except Exception:
            C_uu = np.zeros_like(tau)
    else:
        C_uu = np.zeros_like(tau)

    if not np.any(C_uu > 0.0):
        nbar0 = f0 / (f0 + mu)
        source0 = nbar0 * (1.0 - nbar0) * np.exp(-(f0 + mu) * tau)
        C_uu = _binary_filter_even_cov(tau, source0, beta, sigma)

    intrinsic_mode = str(intrinsic).lower()
    q_method = str(q_method).lower()
    C_nn = np.zeros_like(tau)

    print(f"  [clipped-integral] sigma={sigma:.3f}, intrinsic={intrinsic_mode}, iterating...")
    for it in range(max_iter):
        C_uu_0 = max(float(C_uu[0]), 1e-12)
        nbar, var_cond, gamma_eff = _binary_clipped_moments(
            C_uu_0, f0, f1, mu, n_quad=n_quad
        )
        p_coeffs = None
        if q_method in ("hermite", "hermite-series", "series"):
            def p_of_u(u):
                fu = np.maximum(f0 + f1 * u, 0.0)
                return fu / (fu + mu)
            p_coeffs = _normalized_hermite_coeffs(
                p_of_u, C_uu_0, n_quad=n_quad, order=hermite_order
            )
        C_smooth = np.array([
            _binary_clipped_nbar_cov(
                C_uu[k],
                C_uu_0,
                f0,
                f1,
                mu,
                n_quad=n_quad,
                q_method=q_method,
                hermite_order=hermite_order,
                coeffs=p_coeffs,
            )
            for k in range(ntau)
        ])

        if intrinsic_mode in ("telegraph", "colored", "markov"):
            C_intrinsic = var_cond * np.exp(-gamma_eff * tau)
        elif intrinsic_mode in ("white", "d0"):
            D0_eff = 2.0 * var_cond * gamma_eff
            C_intrinsic = (D0_eff / (2.0 * gamma_eff)) * np.exp(-gamma_eff * tau)
        elif intrinsic_mode in ("none", "off", "smooth"):
            C_intrinsic = np.zeros_like(tau)
        else:
            raise ValueError("intrinsic must be 'telegraph', 'white', or 'none'")

        C_nn_new = np.maximum(C_smooth + C_intrinsic, 0.0)
        C_uu_new = np.maximum(_binary_filter_even_cov(tau, C_nn_new, beta, sigma), 0.0)

        change = float(np.max(np.abs(C_uu_new - C_uu)))
        C_uu = (1.0 - mix) * C_uu + mix * C_uu_new
        C_nn = (1.0 - mix) * C_nn + mix * C_nn_new
        if change < tol:
            print(f"    converged at iteration {it + 1}, change={change:.2e}")
            break
    else:
        print(f"    did not converge after {max_iter} iters, last change={change:.2e}")

    C_uu_0 = max(float(C_uu[0]), 1e-12)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    u = np.sqrt(C_uu_0) * gh_x * np.sqrt(2.0)
    fu = np.maximum(f0 + f1 * u, 0.0)
    c1_eff = float(np.dot(
        gh_w,
        f1 * (fu > 0.0).astype(float) * mu / (fu + mu) ** 2,
    ) / np.sqrt(np.pi))
    gamma_ref = mu + f0
    g_eff = c1_eff * sigma / gamma_ref
    print(
        f"    C_uu(0)={C_uu[0]:.4f}, C_nn(0)={C_nn[0]:.4f}, "
        f"nbar={nbar:.4f}, gamma_eff={gamma_eff:.4f}, g_eff={g_eff:.4f}"
    )
    return tau, C_nn, C_uu, g_eff


def theory_binary_clipped(
    sigma,
    beta,
    mu,
    f0,
    f1,
    tau_max=30,
    dtau=0.05,
    n_quad=32,
    max_iter=120,
    tol=1e-5,
    mix=0.3,
):
    """
    Self-consistent clipped-gain 2PI theory with coupled (C_uu, C_un, C_nn).
    """
    gamma = mu + f0
    ntau = int(tau_max / dtau)
    tau = np.arange(ntau) * dtau
    omega = fftfreq(ntau, d=dtau) * 2.0 * np.pi
    b2 = beta**2
    g2 = gamma**2
    w2 = omega**2
    s2 = sigma**2

    # Correct initialization object: linear C_uu from linear theory.
    _, _Cnn_lin, Cuu_lin, g_lin = theory_binary_autocorr(
        sigma, beta, mu, f0, f1, tau_max=tau_max, dtau=dtau
    )
    C_uu = np.maximum(Cuu_lin.copy(), 0.0)

    print(f"  [clipped] sigma={sigma:.3f}, g_linear={g_lin:.3f}, iterating...")

    for it in range(max_iter):
        C_uu_0 = max(float(C_uu[0]), 1e-12)

        # Effective coefficients from the current C_uu(0).  C_uu is already
        # the synaptic-drive covariance, while sigma^2 enters when source
        # covariances are filtered into u.
        gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
        s = np.sqrt(C_uu_0)
        u = s * gh_x * np.sqrt(2.0)
        fu = np.maximum(f0 + f1 * u, 0.0)
        nbar_eff = float(np.dot(gh_w, fu / (fu + mu)) / np.sqrt(np.pi))
        c1_eff = float(np.dot(
            gh_w,
            f1 * (fu > 0).astype(float) * mu / (fu + mu) ** 2,
        ) / np.sqrt(np.pi))
        D0_eff = _D0_clipped(C_uu_0, f0, f1, mu, sigma, n_quad=n_quad)

        Q = np.array([
            _Q_clipped_correct(C_uu[k], C_uu_0, f0, f1, mu, sigma, n_quad=n_quad)
            for k in range(ntau)
        ])
        Q_hat = fft(Q)

        # Coupled 3-component update in frequency space.
        C_uu_hat = b2 * s2 * Q_hat / (b2 + w2)
        C_un_hat = (
            b2 * s2 * c1_eff * nbar_eff * Q_hat / ((b2 + w2) * (g2 + w2))
        )
        C_nn_hat = (
            D0_eff + c1_eff * nbar_eff * b2 * s2 * C_un_hat / (g2 + w2)
        ) / (g2 + w2)

        C_uu_new = np.maximum(np.real(ifft(C_uu_hat)), 0.0)
        C_nn_new = np.maximum(np.real(ifft(C_nn_hat)), 0.0)

        change = float(np.max(np.abs(C_uu_new - C_uu)))
        C_uu = (1.0 - mix) * C_uu + mix * C_uu_new

        if change < tol:
            print(f"    converged at iteration {it + 1}, change={change:.2e}")
            break
    else:
        print(f"    did not converge after {max_iter} iters, last change={change:.2e}")

    # Final C_nn at converged C_uu.
    C_uu_0 = max(float(C_uu[0]), 1e-12)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    s = np.sqrt(C_uu_0)
    u = s * gh_x * np.sqrt(2.0)
    fu = np.maximum(f0 + f1 * u, 0.0)
    nbar_eff = float(np.dot(gh_w, fu / (fu + mu)) / np.sqrt(np.pi))
    c1_eff = float(np.dot(
        gh_w,
        f1 * (fu > 0).astype(float) * mu / (fu + mu) ** 2,
    ) / np.sqrt(np.pi))
    D0_eff = _D0_clipped(C_uu_0, f0, f1, mu, sigma, n_quad=n_quad)
    Q = np.array([
        _Q_clipped_correct(C_uu[k], C_uu_0, f0, f1, mu, sigma, n_quad=n_quad)
        for k in range(ntau)
    ])
    Q_hat = fft(Q)
    C_un_hat = b2 * s2 * c1_eff * nbar_eff * Q_hat / ((b2 + w2) * (g2 + w2))
    C_nn_hat = (
        D0_eff + c1_eff * nbar_eff * b2 * s2 * C_un_hat / (g2 + w2)
    ) / (g2 + w2)
    C_nn = np.maximum(np.real(ifft(C_nn_hat)), 0.0)

    g_eff = c1_eff * sigma / gamma
    print(
        f"    C_uu(0)={C_uu_0:.4f}, C_nn(0)={C_nn[0]:.4f}, "
        f"nbar_eff={nbar_eff:.4f}, g_eff={g_eff:.4f}"
    )
    return tau, C_nn, C_uu, g_eff


def plot_clipped_vs_linear(
    sigma_vals=(0.7, 1.0, 1.3),
    N=800,
    beta=1.0,
    mu=1.0,
    f0=0.5,
    f1=1.0,
    T=5000.0,
    dt=0.02,
    burn=500.0,
    tau_max=20.0,
    sim_method="tau-leap",
    sim_cache_path=None,
    force_resim=False,
    plot_dir=None,
    clipped_methods=("effective", "integral-telegraph"),
    integral_q_method="hermite",
    integral_hermite_order=32,
):
    """Compare clipped simulation with several clipped-vs-linear theory predictions."""
    import os
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    cached = {}
    cache_updated = False
    if sim_cache_path is not None and (not force_resim):
        try:
            blob = np.load(sim_cache_path, allow_pickle=False)
            sigmas_cached = blob["sigmas"]
            for i, sigma_cached in enumerate(sigmas_cached):
                cached[float(sigma_cached)] = (
                    blob[f"tau_{i}"],
                    blob[f"Cnn_{i}"],
                    blob[f"Cuu_{i}"],
                )
            print(f"Loaded simulation cache: {sim_cache_path}")
        except FileNotFoundError:
            pass
        except Exception as err:
            print(f"WARNING: failed to read cache {sim_cache_path}: {err}")

    fig, axes = plt.subplots(1, len(sigma_vals), figsize=(5 * len(sigma_vals), 4))
    if len(sigma_vals) == 1:
        axes = [axes]

    for ax, sigma in zip(axes, sigma_vals):
        print(f"\n-- clipped compare sigma={sigma} --")

        gamma = mu + f0
        c1 = f1 * mu / gamma
        g_lin = c1 * sigma / gamma

        if g_lin < 1.0:
            tau_lin, Cnn_lin, _, _ = theory_binary_autocorr(
                sigma, beta, mu, f0, f1, tau_max=tau_max, dtau=max(dt, 0.05)
            )
        else:
            tau_lin, Cnn_lin = None, None

        clipped_curves = []
        for method in clipped_methods:
            method_key = str(method).lower()
            if method_key in ("effective", "quasi", "quasi-static"):
                tau_clip, Cnn_clip, _Cuu_clip, g_eff = theory_binary_clipped(
                    sigma, beta, mu, f0, f1, tau_max=tau_max, dtau=max(dt, 0.05)
                )
                clipped_curves.append((
                    tau_clip,
                    Cnn_clip,
                    fr"2PI eff-linear ($g_{{eff}}={g_eff:.2f}$)",
                    "r--",
                ))
            elif method_key in ("integral", "integral-telegraph", "telegraph"):
                tau_clip, Cnn_clip, _Cuu_clip, g_eff = theory_binary_clipped_integral(
                    sigma, beta, mu, f0, f1, tau_max=tau_max, dtau=max(dt, 0.05),
                    intrinsic="telegraph",
                    q_method=integral_q_method,
                    hermite_order=integral_hermite_order,
                )
                clipped_curves.append((
                    tau_clip,
                    Cnn_clip,
                    fr"Integral+telegraph ($g_{{eff}}={g_eff:.2f}$)",
                    "m-.",
                ))
            elif method_key in ("integral-none", "smooth"):
                tau_clip, Cnn_clip, _Cuu_clip, g_eff = theory_binary_clipped_integral(
                    sigma, beta, mu, f0, f1, tau_max=tau_max, dtau=max(dt, 0.05),
                    intrinsic="none",
                    q_method=integral_q_method,
                    hermite_order=integral_hermite_order,
                )
                clipped_curves.append((
                    tau_clip,
                    Cnn_clip,
                    fr"Integral smooth ($g_{{eff}}={g_eff:.2f}$)",
                    "c-.",
                ))
            else:
                raise ValueError(f"Unknown clipped method: {method}")

        if sigma in cached:
            tau_s, Cnn_s, _Cuu_s = cached[sigma]
            print(f"  using cached simulation for sigma={sigma}")
        else:
            tau_s, Cnn_s, Cuu_s = sim_binary_network(
                N=N,
                sigma=sigma,
                beta=beta,
                mu=mu,
                f0=f0,
                f1=f1,
                T=T,
                dt=dt,
                burn=burn,
                clip_rate_on=True,
                method=sim_method,
            )
            cached[sigma] = (tau_s, Cnn_s, Cuu_s)
            cache_updated = True

        def norm(x):
            x0 = float(x[0])
            return x / x0 if abs(x0) > 1e-12 else x

        ax.plot(tau_s, norm(Cnn_s), "b", lw=1.5, alpha=0.8, label="Sim (clipped)")
        for tau_clip, Cnn_clip, label, style in clipped_curves:
            ax.plot(tau_clip, norm(Cnn_clip), style, lw=2, label=label)
        if Cnn_lin is not None:
            ax.plot(
                tau_lin,
                norm(Cnn_lin),
                "g:",
                lw=1.5,
                label=fr"Linear ($g={g_lin:.2f}$)",
            )
        ax.set(
            xlabel=r"$\tau$",
            ylabel=r"$C_{nn}(\tau)/C_{nn}(0)$",
            title=fr"$\sigma={sigma}$",
            xlim=(0, min(15.0, tau_max)),
        )
        ax.legend(fontsize=8)

    plt.suptitle("Clipped vs linear gain: binary 2PI", fontsize=13, fontweight="bold")
    import os
    os.makedirs(plot_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "clipped_vs_linear.png"), dpi=150)
    print(f"Saved to {os.path.join(plot_dir, 'clipped_vs_linear.png')}")
    plt.close("all")

    if sim_cache_path is not None and (cache_updated or force_resim):
        to_save = {"sigmas": np.array(sorted(cached.keys()), dtype=float)}
        for i, sigma_cached in enumerate(to_save["sigmas"]):
            tau_s, Cnn_s, Cuu_s = cached[float(sigma_cached)]
            to_save[f"tau_{i}"] = np.asarray(tau_s)
            to_save[f"Cnn_{i}"] = np.asarray(Cnn_s)
            to_save[f"Cuu_{i}"] = np.asarray(Cuu_s)
        np.savez(sim_cache_path, **to_save)
        print(f"Saved simulation cache: {sim_cache_path}")


def plot_binary_network(
    sigma_vals=(0.5, 0.8, 0.95),
    N=800,
    beta=1.0,
    mu=1.0,
    f0=0.5,
    f1=1.0,
    clip_rate_on=True,
    sim_method="tau-leap",
    plot_dir=None,
):
    """
    For each sigma, compare simulation vs theory.
    Also shows the g=1 transition.
    """
    import os
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(2, len(sigma_vals), figsize=(5 * len(sigma_vals), 8))
    if len(sigma_vals) == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, sigma in enumerate(sigma_vals):
        print(f"\n-- sigma={sigma} --")
        tau_th, Cnn_th, Cuu_th, g = theory_binary_autocorr(sigma, beta, mu, f0, f1)

        print("  Simulating ...")
        tau_s, Cnn_s, Cuu_s = sim_binary_network(
            N=N,
            sigma=sigma,
            beta=beta,
            mu=mu,
            f0=f0,
            f1=f1,
            clip_rate_on=clip_rate_on,
            method=sim_method,
        )

        # normalise by C(0)
        Cnn_s /= Cnn_s[0]
        Cuu_s /= max(Cuu_s[0], 1e-10)
        Cnn_th_n = Cnn_th / max(Cnn_th[0], 1e-10)
        Cuu_th_n = Cuu_th / max(Cuu_th[0], 1e-10) if Cuu_th[0] > 0 else Cuu_th

        # C_nn
        ax = axes[0, col]
        ax.plot(tau_s, Cnn_s, "b", lw=1.5, label="Sim")
        ax.plot(tau_th, Cnn_th_n, "r--", lw=2, label="Theory")
        ax.set(
            xlabel=r"$\tau$",
            ylabel=r"$C_{nn}(\tau)/C_{nn}(0)$",
            title=fr"$\sigma={sigma},\ g={g:.2f}$",
            xlim=(0, 20),
        )
        ax.legend(fontsize=8)

        # C_uu
        ax = axes[1, col]
        ax.plot(tau_s, Cuu_s, "g", lw=1.5, label="Sim")
        ax.plot(tau_th, Cuu_th_n, "m--", lw=2, label="Theory")
        ax.set(
            xlabel=r"$\tau$",
            ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
            title=fr"$C_{{uu}}$,  $\sigma={sigma}$",
            xlim=(0, 20),
        )
        ax.legend(fontsize=8)

    plt.suptitle("Binary neuron network: 2PI theory test", fontsize=13, fontweight="bold")
    import os
    os.makedirs(plot_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "binary_network_test.png"), dpi=150)
    print(f"Saved to {os.path.join(plot_dir, 'binary_network_test.png')}")
    plt.close("all")


# -----------------------------------------------------------------------------
# 3. TWO-TIMESCALE FIT
#    Fit the simulated C_nn(tau) to the two-exponential form
#    A+ exp(-kappa+ tau) + A- exp(-kappa- tau) and compare with theory.
# -----------------------------------------------------------------------------

def fit_two_exponentials(tau, C, p0_theory=None):
    """Least-squares fit of C(tau) = A+ e^{-k+ tau} + A- e^{-k- tau}."""
    from scipy.optimize import curve_fit

    def model(t, Ap, kp, Am, km):
        return Ap * np.exp(-kp * t) + Am * np.exp(-km * t)

    if p0_theory is not None:
        p0 = list(p0_theory)
    else:
        C0 = C[0]
        p0 = [C0 * 0.5, 2.0, C0 * 0.5, 0.3]
    bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(model, tau, C, p0=p0, bounds=bounds, maxfev=8000)
    except Exception:
        popt = p0
    return popt


def plot_two_timescale_fit(
    sigma=0.8,
    N=800,
    beta=1.0,
    mu=1.0,
    f0=0.5,
    f1=1.0,
    clip_rate_on=True,
    sim_method="tau-leap",
    plot_dir=None,
):
    """
    Show that the simulated correlation function is well fit by two exponentials,
    with decay rates matching theory predictions kappa+ and kappa-.
    """
    import os
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    tau_th, Cnn_th, _, g = theory_binary_autocorr(sigma, beta, mu, f0, f1)
    tau_s, Cnn_s, _ = sim_binary_network(
        N=N,
        sigma=sigma,
        beta=beta,
        mu=mu,
        f0=f0,
        f1=f1,
        clip_rate_on=clip_rate_on,
        method=sim_method,
    )
    gamma = mu + f0
    c1 = f1 * mu / gamma
    disc = (gamma**2 - beta**2) ** 2 + 4 * c1**2 * beta**2 * sigma**2
    kp_th = np.sqrt(0.5 * ((gamma**2 + beta**2) + np.sqrt(disc)))
    km_th = np.sqrt(0.5 * ((gamma**2 + beta**2) - np.sqrt(disc)))
    D0 = 2 * (f0 / gamma) * (1 - f0 / gamma) * gamma
    Ap_th = D0 * 0.5 * (beta**2 - kp_th**2) / (kp_th * (km_th**2 - kp_th**2))
    Am_th = D0 * 0.5 * (beta**2 - km_th**2) / (km_th * (km_th**2 - kp_th**2))

    popt = fit_two_exponentials(tau_s, Cnn_s, p0_theory=[Ap_th, kp_th, Am_th, km_th])
    Ap, kp_fit, Am, km_fit = popt

    fit_curve = Ap * np.exp(-kp_fit * tau_s) + Am * np.exp(-km_fit * tau_s)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tau_s, Cnn_s, "b", lw=1.5, alpha=0.7, label="Simulation")
    ax.plot(
        tau_s,
        fit_curve,
        "r--",
        lw=2,
        label=fr"Fit: $\kappa_+={kp_fit:.3f}$, $\kappa_-={km_fit:.3f}$",
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set(
        xlabel=r"$\tau$",
        ylabel=r"$C_{nn}(\tau)$",
        title=fr"Two-timescale structure: $\sigma={sigma}$, $g={g:.2f}$",
        xlim=(0, 20),
    )
    ax.legend()
    txt = (
        f"Theory:  $\\kappa_+={kp_th:.3f}$,  $\\kappa_-={km_th:.3f}$\n"
        f"Fit sim: $\\kappa_+={kp_fit:.3f}$,  $\\kappa_-={km_fit:.3f}$"
    )
    ax.text(
        0.55,
        0.75,
        txt,
        transform=ax.transAxes,
        bbox=dict(fc="white", ec="gray"),
        fontsize=9,
    )
    plt.tight_layout()
    import os
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "two_timescale_fit.png"), dpi=150)
    print(f"Saved to {os.path.join(plot_dir, 'two_timescale_fit.png')}")
    plt.close("all")


# -----------------------------------------------------------------------------
# 4. CONVERGENCE WITH NETWORK SIZE
#    Plot correlation functions for binary network at different N values
#    against theory to visualize finite-size effects
# -----------------------------------------------------------------------------

def plot_binary_network_N_convergence(sigma=0.8, N_vals=(128, 300, 800, 1600),
                                       beta=1.0, mu=1.0, f0=0.5, f1=1.0,
                                       clip_rate_on=True, sim_method="tau-leap",
                                       plot_dir=None):
    """
    Compare simulation vs theory for binary network at multiple network sizes.
    Shows how finite-size effects diminish as N increases.
    """
    import os
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\n-- Binary network: N convergence at sigma={sigma} --")
    tau_th, Cnn_th, Cuu_th, g = theory_binary_autocorr(sigma, beta, mu, f0, f1)
    Cnn_th_n = Cnn_th / max(Cnn_th[0], 1e-10)
    Cuu_th_n = Cuu_th / max(Cuu_th[0], 1e-10) if Cuu_th[0] > 0 else Cuu_th

    colors = plt.cm.viridis(np.linspace(0, 1, len(N_vals)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # C_nn convergence
    ax = axes[0]
    ax.plot(tau_th, Cnn_th_n, "k--", lw=3, label="Theory", alpha=0.8, zorder=100)
    sim_curves = {}
    for N, color in zip(N_vals, colors):
        print(f"  Simulating N={N} ...")
        tau_s, Cnn_s, Cuu_s = sim_binary_network(
            N=N,
            sigma=sigma,
            beta=beta,
            mu=mu,
            f0=f0,
            f1=f1,
            clip_rate_on=clip_rate_on,
            method=sim_method,
        )
        sim_curves[N] = (tau_s, Cnn_s, Cuu_s)
        Cnn_s_n = Cnn_s / max(Cnn_s[0], 1e-10)
        ax.plot(tau_s, Cnn_s_n, lw=1.5, color=color, label=f"N={N}")
    ax.set(xlabel=r"$\tau$", ylabel=r"$C_{nn}(\tau)/C_{nn}(0)$",
           title=fr"$C_{{nn}}$ convergence: $\sigma={sigma}$, $g={g:.2f}$",
           xlim=(0, 20))
    ax.legend(fontsize=9)

    # C_uu convergence
    ax = axes[1]
    ax.plot(tau_th, Cuu_th_n, "k--", lw=3, label="Theory", alpha=0.8, zorder=100)
    for N, color in zip(N_vals, colors):
        tau_s, _, Cuu_s = sim_curves[N]
        Cuu_s_n = Cuu_s / max(abs(Cuu_s[0]), 1e-10)
        ax.plot(tau_s, Cuu_s_n, lw=1.5, color=color, label=f"N={N}")
    ax.set(xlabel=r"$\tau$", ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
           title=fr"$C_{{uu}}$ convergence: $\sigma={sigma}$",
           xlim=(0, 20))
    ax.legend(fontsize=9)

    plt.suptitle(f"Binary network: finite-size convergence", fontsize=13, fontweight="bold")
    import os
    os.makedirs(plot_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "binary_network_N_convergence.png"), dpi=150)
    print(f"Saved to {os.path.join(plot_dir, 'binary_network_N_convergence.png')}")
    plt.close("all")


# -----------------------------------------------------------------------------
# Phase model: timeseries and raster helpers
# -----------------------------------------------------------------------------

def _sim_phase_timeseries(
    N=256,
    I=1.0,
    alpha=1.0,
    sigma=7.0,
    beta=1.0,
    T=500.0,
    dt=0.02,
    n_show=20,
    burn=200.0,
    rng_=None,
):
    """Simulate phase network and return raw u timeseries and per-neuron spike times.

    Returns
    -------
    t         : (nt,) time array
    U         : (nt, n_show) u-values for the first n_show neurons
    spk_times : list of n_show arrays, each holding spike times for that neuron
    """
    if rng_ is None:
        rng_ = np.random.default_rng(42)
    n_show = min(n_show, N)
    W = make_weights(N, sigma, lam=1, rng=rng_)
    phi = rng_.uniform(-np.pi, np.pi, N)
    u = np.zeros(N)

    def F(u_):
        return alpha * np.clip(I + u_, 0.0, 1e12) ** (1.0 / alpha)

    nb = int(burn / dt)
    for _ in range(nb):
        phi += F(u) * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        drive = W @ spike_counts
        u += -beta * u * dt + beta * drive
        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)

    nt = int(T / dt)
    t = np.arange(nt) * dt
    idx = np.arange(n_show)
    U = np.zeros((nt, n_show))
    spk_times = [[] for _ in range(n_show)]

    for step in range(nt):
        phi += F(u) * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        drive = W @ spike_counts
        u += -beta * u * dt + beta * drive
        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
        U[step] = u[idx]
        for k in range(n_show):
            count = int(spike_counts[idx[k]])
            if count:
                spk_times[k].extend([t[step]] * count)

    return t, U, [np.array(st) for st in spk_times]


def _phase_sigma_c(I, alpha):
    rho = 1.0 / (2.0 * np.pi)
    Fprime0 = float(np.maximum(I, 1e-10) ** (1.0 / alpha - 1.0))
    return 1.0 / (rho * Fprime0)


# -----------------------------------------------------------------------------
# 1. u(t) time series
# -----------------------------------------------------------------------------

def plot_u_timeseries(
    sigma_vals=None,
    I=1.0,
    alpha=1.0,
    beta=1.0,
    N=256,
    T=200.0,
    dt=0.02,
    n_show=8,
    burn=100.0,
    plot_dir=None,
):
    """Plot u(t) traces for several neurons across sigma values.

    One column per sigma value; each panel shows n_show overlaid traces.
    """
    import os
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    sigma_c = _phase_sigma_c(I, alpha)
    if sigma_vals is None:
        sigma_vals = [0.5 * sigma_c, 1.0 * sigma_c, 2.0 * sigma_c]

    ncols = len(sigma_vals)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, sigma in zip(axes, sigma_vals):
        g = sigma / sigma_c
        print(f"  u timeseries: sigma={sigma:.2f} (g={g:.2f})")
        t, U, _ = _sim_phase_timeseries(
            N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
            T=T, dt=dt, n_show=n_show, burn=burn,
        )
        for k in range(n_show):
            ax.plot(t, U[:, k], lw=0.8, alpha=0.7)
        ax.set(
            xlabel="time",
            ylabel=r"$u_i(t)$",
            title=fr"$\sigma={sigma:.2f}$, $g=\sigma/\sigma_c={g:.2f}$",
        )

    plt.suptitle(
        fr"Phase network: $u(t)$ time series  ($I={I}$, $\alpha={alpha}$, $\sigma_c={sigma_c:.2f}$)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_u_timeseries.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


# -----------------------------------------------------------------------------
# 2. Spike raster + population rate
# -----------------------------------------------------------------------------

def plot_phase_raster(
    sigma_vals=None,
    I=1.0,
    alpha=1.0,
    beta=1.0,
    N=256,
    T=500.0,
    dt=0.02,
    burn=100.0,
    plot_dir=None,
):
    """Spike raster (neuron index vs time) + population rate for several sigma.

    Two rows per sigma: raster (top), smoothed population firing rate (bottom).
    """
    import os
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    sigma_c = _phase_sigma_c(I, alpha)
    if sigma_vals is None:
        sigma_vals = [0.5 * sigma_c, 1.0 * sigma_c, 2.0 * sigma_c]

    ncols = len(sigma_vals)
    fig, axes = plt.subplots(
        2, ncols, figsize=(5 * ncols, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for col, sigma in enumerate(sigma_vals):
        g = sigma / sigma_c
        print(f"  raster: sigma={sigma:.2f} (g={g:.2f})")
        t, _, spk_times = _sim_phase_timeseries(
            N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
            T=T, dt=dt, n_show=N, burn=burn,
        )
        ax_raster = axes[0, col]
        ax_rate   = axes[1, col]

        # Raster
        for k, st in enumerate(spk_times):
            if len(st):
                ax_raster.scatter(st, np.full(len(st), k), s=0.5, c="k", linewidths=0)
        ax_raster.set(
            xlim=(0, T), ylim=(-1, N),
            ylabel="neuron" if col == 0 else "",
            title=fr"$\sigma={sigma:.2f}$, $g={g:.2f}$",
        )
        ax_raster.tick_params(labelbottom=False)

        # Population rate (smoothed)
        nt = len(t)
        pop_rate = np.zeros(nt)
        for st in spk_times:
            for ts in st:
                idx = int(ts / dt)
                if idx < nt:
                    pop_rate[idx] += 1.0
        pop_rate /= N * dt
        win = max(1, int(5.0 / dt))
        pop_rate_sm = np.convolve(pop_rate, np.ones(win) / win, mode="same")
        ax_rate.plot(t, pop_rate_sm, lw=1.0, color="steelblue")
        ax_rate.set(
            xlabel="time",
            ylabel="rate" if col == 0 else "",
            xlim=(0, T),
        )

    plt.suptitle(
        fr"Phase network: spike raster  ($I={I}$, $\alpha={alpha}$, $\sigma_c={sigma_c:.2f}$)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_raster.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


# -----------------------------------------------------------------------------
# Parallel simulation helper (module-level so ProcessPoolExecutor can pickle it)
# -----------------------------------------------------------------------------

def _phase_sim_job(kwargs):
    """Worker for parallel sim_phase_network calls.

    kwargs must contain a 'seed' key (int or None) — a fresh rng is created
    per job so workers don't share RNG state.
    """
    seed = kwargs.pop("seed", None)
    job_rng = np.random.default_rng(seed)
    return sim_phase_network(**kwargs, rng=job_rng)


def _run_jobs_parallel(jobs, n_jobs):
    """Run a list of job-dicts via _phase_sim_job, returning results in order.

    n_jobs=1  → sequential (no extra processes).
    n_jobs>1  → ProcessPoolExecutor with that many workers.
    n_jobs=-1 → use os.cpu_count() workers.
    """
    import os, sys
    from concurrent.futures import ProcessPoolExecutor

    if n_jobs == 1:
        return [_phase_sim_job(dict(j)) for j in jobs]

    workers = os.cpu_count() if n_jobs == -1 else int(n_jobs)
    # Use 'fork' on POSIX so child processes inherit the already-imported
    # module state without re-executing __main__, avoiding the recursive-spawn
    # problem that occurs on macOS when the start method is 'spawn'.
    mp_ctx = None
    if sys.platform != "win32":
        import multiprocessing
        mp_ctx = multiprocessing.get_context("fork")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
            return list(ex.map(_phase_sim_job, [dict(j) for j in jobs]))
    except PermissionError as err:
        print(f"  Parallel workers unavailable ({err}); running sequentially.")
        return [_phase_sim_job(dict(j)) for j in jobs]


def _phase_theory_label(theory_kwargs):
    solver = str(theory_kwargs.get("solver", "fd")).replace("_", "-")
    q_method = str(theory_kwargs.get("q_method", "gh"))
    return theory_kwargs.get("label", f"2PI {solver}/{q_method}")


# -----------------------------------------------------------------------------
# 3. Sim vs theory: parameter dependence
# -----------------------------------------------------------------------------

def plot_phase_corr_params(
    param_sets=None,
    N=512,
    T=5000.0,
    dt=0.02,
    tau_max=80.0,
    sim_reps=2,
    plot_dir=None,
    n_jobs=1,
    theory_kwargs=None,
):
    """Sim vs theory C_uu(tau) for different parameter combinations.

    param_sets : list of dicts with keys I, alpha, beta, sigma (absolute values),
                 and optional 'label'. Theory is shown only when sigma < sigma_c.
    """
    import os
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    theory_kwargs = {} if theory_kwargs is None else dict(theory_kwargs)

    if param_sets is None:
        sc1 = _phase_sigma_c(1.0, 1.0)   # I=1, alpha=1
        sc2 = _phase_sigma_c(1.0, 2.0)   # I=1, alpha=2
        sc05 = _phase_sigma_c(1.0, 0.5)  # I=1, alpha=0.5
        param_sets = [
            dict(I=1.0, alpha=1.0, beta=1.0, sigma=1.1 * sc1,
                 label=r"$\sigma=1.1\sigma_c$, $\alpha=1$"),
            dict(I=1.0, alpha=1.0, beta=1.0, sigma=1.3 * sc1,
                 label=r"$\sigma=1.3\sigma_c$, $\alpha=1$"),
            dict(I=1.0, alpha=1.0, beta=1.0, sigma=1.5 * sc1,
                 label=r"$\sigma=1.5\sigma_c$, $\alpha=1$"),
            dict(I=1.0, alpha=2.0, beta=1.0, sigma=1.3 * sc2,
                 label=r"$\sigma=1.3\sigma_c$, $\alpha=2$"),
            dict(I=1.0, alpha=0.5, beta=1.0, sigma=0.75 * sc05,
                 label=r"$\sigma=0.75\sigma_c$, $\alpha=0.5$"),
            dict(I=1.0, alpha=1.0, beta=0.5, sigma=1.3 * sc1,
                 label=r"$\sigma=1.3\sigma_c$, $\beta=0.5$"),
        ]

    n = len(param_sets)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    # Build all sim jobs up-front so they can be dispatched in parallel.
    ss = np.random.SeedSequence()
    all_jobs = []
    for ps in param_sets:
        for _ in range(max(1, sim_reps)):
            seed = int(ss.spawn(1)[0].generate_state(1)[0])
            all_jobs.append(dict(
                N=N, I=ps["I"], alpha=ps["alpha"], sigma=ps["sigma"],
                beta=ps["beta"], T=T, dt=dt, tau_max=tau_max, n_probe=N,
                seed=seed,
            ))

    print(f"  Dispatching {len(all_jobs)} sims (n_jobs={n_jobs}) …")
    all_results = _run_jobs_parallel(all_jobs, n_jobs)

    result_iter = iter(all_results)
    for ax, ps in zip(axes_flat, param_sets):
        I_ = ps["I"]; alpha_ = ps["alpha"]; beta_ = ps["beta"]; sigma_ = ps["sigma"]
        label = ps.get("label", fr"$\sigma={sigma_:.2f}$")
        sigma_c_ = _phase_sigma_c(I_, alpha_)
        print(f"  {label}  (sigma/sigma_c={sigma_/sigma_c_:.2f})")

        # Theory
        try:
            tau_th, C_th, _ = theory_phase_autocorr(
                I=I_, alpha=alpha_, sigma=sigma_, beta=beta_,
                tau_max=tau_max, dtau=dt, **theory_kwargs,
            )
        except Exception as e:
            print(f"  theory failed: {e}")
            tau_th, C_th = None, None

        # Collect the pre-computed simulation results for this param set.
        C_runs = []
        tau_s = None
        for _ in range(max(1, sim_reps)):
            tau_run, C_run = next(result_iter)
            tau_s = tau_run
            C_runs.append(C_run)
        C_s = np.mean(C_runs, axis=0)

        C_s_norm = C_s / C_s[0] if C_s[0] > 0 else C_s
        has_theory = C_th is not None and C_th[0] > 1e-2
        if has_theory:
            C_th_norm = C_th / C_th[0]
            ax.plot(tau_th, C_th_norm, "r--", lw=2,
                    label=_phase_theory_label(theory_kwargs), zorder=3)
        ax.plot(tau_s, C_s_norm, "b", lw=1.5, alpha=0.85, label="Sim", zorder=2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(
            xlabel=r"$\tau$", ylabel=r"$C_{uu}(\tau) / C_{uu}(0)$",
            ylim=(-0.2, 1.05),
            title=label, xlim=(0, tau_max),
        )
        ax.legend(fontsize=8)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle(
        "Phase network: sim vs theory — parameter dependence",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_corr_params.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


# -----------------------------------------------------------------------------
# 4. Sim vs theory: N convergence
# -----------------------------------------------------------------------------

def plot_phase_corr_N(
    N_vals=None,
    sigma=None,
    I=1.0,
    alpha=1.0,
    beta=1.0,
    T=5000.0,
    dt=0.02,
    tau_max=80.0,
    plot_dir=None,
    n_jobs=1,
    theory_kwargs=None,
):
    """Sim vs theory C_uu(tau) for different N (finite-size convergence).

    Three panels: subcritical / near-critical / chaotic regimes.
    Each panel overlays all N values; theory shown when available.
    """
    import os
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    theory_kwargs = {} if theory_kwargs is None else dict(theory_kwargs)

    if N_vals is None:
        N_vals = (64, 128, 256, 512, 1024)

    sigma_c = _phase_sigma_c(I, alpha)

    if sigma is None:
        sigma_list = [1.1 * sigma_c, 1.3 * sigma_c, 1.5 * sigma_c]
        regime_labels = [
            fr"near-critical ($g=1.1$)",
            fr"supercritical ($g=1.3$)",
            fr"supercritical ($g=1.5$)",
        ]
    else:
        sigma_list = [sigma]
        regime_labels = [fr"$\sigma={sigma:.2f}$"]

    ncols = len(sigma_list)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_vals)))

    # Build all sim jobs so they can be dispatched in parallel.
    ss = np.random.SeedSequence()
    all_jobs = []
    for sig in sigma_list:
        for N in N_vals:
            seed = int(ss.spawn(1)[0].generate_state(1)[0])
            all_jobs.append(dict(
                N=N, I=I, alpha=alpha, sigma=sig, beta=beta,
                T=T, dt=dt, tau_max=tau_max, n_probe=N,
                seed=seed,
            ))

    print(f"  Dispatching {len(all_jobs)} sims (n_jobs={n_jobs}) …")
    all_results = _run_jobs_parallel(all_jobs, n_jobs)

    for ax, sig, rlabel in zip(axes, sigma_list, regime_labels):
        print(f"  N-convergence: sigma={sig:.2f} (g={sig/sigma_c:.2f})")

        # Theory — only plot when a valid SCS fixed point was found
        try:
            tau_th, C_th, _ = theory_phase_autocorr(
                I=I, alpha=alpha, sigma=sig, beta=beta, tau_max=tau_max, dtau=dt,
                **theory_kwargs,
            )
            if C_th[0] > 1e-2:
                C_th_norm = C_th / C_th[0]
                ax.plot(tau_th, C_th_norm, "k--", lw=2.5, zorder=6,
                        label=_phase_theory_label(theory_kwargs))
        except Exception as e:
            print(f"  theory failed: {e}")

        n_sigma = len(sigma_list)
        sig_idx = sigma_list.index(sig)
        for i, (N, color) in enumerate(zip(N_vals, colors)):
            tau_s, C_s = all_results[sig_idx * len(N_vals) + i]
            print(f"    N={N}")
            C_s_norm = C_s / C_s[0] if C_s[0] > 0 else C_s
            ax.plot(tau_s, C_s_norm, lw=1.3, color=color, alpha=0.85, label=f"N={N}")

        ax.axhline(0, color="k", lw=0.5)
        ax.set(
            xlabel=r"$\tau$", ylabel=r"$C_{uu}(\tau) / C_{uu}(0)$",
            ylim=(-0.2, 1.05),
            title=rlabel, xlim=(0, tau_max),
        )
        ax.legend(fontsize=8)

    plt.suptitle(
        fr"Phase network: finite-size convergence  ($I={I}$, $\alpha={alpha}$, $\sigma_c={sigma_c:.2f}$)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_corr_N.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    # Save plots to data/plots directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving plots to: {plot_dir}")

    # 1. Rate network
    # sigma < 1 -> fixed point, sigma > 1 -> chaos for tanh
    plot_rate_network(sigma=1.5, N=512, C0_guess=0.65, plot_dir=plot_dir)

    # 2. Binary network: shape of correlation functions
    plot_binary_network(sigma_vals=(0.5, 0.8, 0.95), N=800, plot_dir=plot_dir)

    # 3. Two-timescale exponential fit
    plot_two_timescale_fit(sigma=0.8, N=800, plot_dir=plot_dir)

    # 4. Finite-size convergence
    plot_binary_network_N_convergence(sigma=0.8, N_vals=(128, 300, 800, 1600), plot_dir=plot_dir)

    # 5. Clipped-gain theory vs simulation
    plot_clipped_vs_linear(sigma_vals=(0.7, 1.0, 1.3), N=800, plot_dir=plot_dir)
