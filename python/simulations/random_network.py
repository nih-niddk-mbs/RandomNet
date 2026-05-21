"""
Simulations to test the 2PI effective action theory for random neural networks:
        1. Rate neuron network          -- Sompolinsky-Crisanti-Sommers equation
        2. Phase neuron network         -- phase-reset spiking with synaptic drive u
        3. Binary neuron network        -- exact two-exponential formula

Run each section independently. Requires numpy, scipy, matplotlib.
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
    if lam:
        W -= W.mean(axis=1, keepdims=True)
    np.fill_diagonal(W, 0)
    return W


# -----------------------------------------------------------------------------
# 1. RATE NEURON NETWORK
#    Model: du_i/dt = -u_i + sum_j W_ij f(u_j)
#    Theory: C''(tau) = C(tau) - Q(tau), Q(tau) = <f(u(0))f(u(tau))>
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
    tau = np.arange(max_lag) * dt
    return tau, C


def theory_rate_autocorr(
    C0=None,
    sigma=1.5,
    tau_max=50,
    dtau=0.01,
    f=np.tanh,
    n_quad=20,
    C0_bounds=(0.05, 5.0),
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
    """
    ntau = int(tau_max / dtau)
    tau = np.arange(ntau) * dtau
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    gh_w2 = np.outer(gh_w, gh_w)

    def Q_func(C_tau, C0_val):
        """Deterministic quadrature for E[f(x)f(y)] with correlated Gaussian x,y."""
        if C0_val <= 0:
            return 0.0
        rho = float(np.clip(C_tau / C0_val, -0.999999, 0.999999))
        scale = np.sqrt(2.0 * C0_val)
        x = scale * gh_x[:, None]
        y = scale * (rho * gh_x[:, None] + np.sqrt(1.0 - rho**2) * gh_x[None, :])
        vals = f(x) * f(y)
        return float(np.sum(gh_w2 * vals) / np.pi)

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

        # Look for a sign change in the energy endpoint.
        for i in range(len(candidates) - 1):
            if values[i] == 0:
                return float(candidates[i])
            if values[i] * values[i + 1] < 0:
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

        # Fallback: choose the candidate closest to the self-consistency root.
        return float(candidates[np.argmin(np.abs(values))])

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


def plot_rate_network(sigma=1.5, N=512, C0_guess=0.8):
    """Compare simulation vs theory for rate network."""
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
    plt.savefig("rate_network_test.png", dpi=150)
    plt.show()


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
        return alpha * np.maximum(I + u_, 0.0) ** (1.0 / alpha)

    nb = int(burn / dt)
    for _ in range(nb):
        phi += F(u) * dt
        spikes = phi >= np.pi
        phi[spikes] -= 2.0 * np.pi
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            drive = W @ spikes.astype(float)
        u += dt * (-beta * u + beta * drive)
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
        spikes = phi >= np.pi
        phi[spikes] -= 2.0 * np.pi
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            drive = W @ spikes.astype(float)
        u += dt * (-beta * u + beta * drive)
        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
        U_probe[t] = u[probe_idx]
        if return_spike:
            R_pop[t] = np.mean(spikes.astype(float)) / dt

    max_lag = int(tau_max / dt)
    C = np.mean([autocorr(U_probe[:, i], max_lag) for i in range(n_probe)], axis=0)
    tau = np.arange(max_lag) * dt
    if not return_spike:
        return tau, C

    C_spk = autocorr(R_pop, max_lag)
    return tau, C, C_spk


def theory_phase_autocorr(
    I=1.0,
    alpha=1.0,
    sigma=7.0,
    beta=1.0,
    C0=None,
    tau_max=50,
    dtau=0.01,
    n_quad=24,
):
    """
    Reduced phase 2PI closure mapped to the rate-theory solver.

    For this model, the closure has the same scalar structure as the rate case,
    with effective gain g(u) = rho * F(u), rho = 1/(2*pi).
    """
    rho = 1.0 / (2.0 * np.pi)

    def g(u):
        return rho * alpha * np.maximum(I + u, 0.0) ** (1.0 / alpha)

    Fprime0 = float(np.maximum(I, 1e-10) ** (1.0 / alpha - 1.0))
    sigma_c = 1.0 / (rho * Fprime0)

    # Gauss-Hermite nodes for centering g.
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)

    def mu_g(C0_val):
        """E[g(u)] under u ~ N(0, C0_val) via GH quadrature."""
        if C0_val <= 0:
            return g(0.0)
        s = np.sqrt(2.0 * float(C0_val))
        return float(np.sum(gh_w * g(s * gh_x)) / np.sqrt(np.pi))

    # With row-sum corrected W, only the centered gain enters the SCS closure.
    # Pass g_c = g - E[g] to theory_rate_autocorr.  Because the mean depends on
    # C0, we do one warm-up pass with raw g to get an approximate C0, then
    # center and solve properly.
    tau_scale = max(float(beta), 1e-10)

    # Pass 1: rough C0 with raw g (gives wrong amplitude but usable C0 estimate)
    _, C_warm = theory_rate_autocorr(
        C0=C0,
        sigma=sigma,
        tau_max=tau_max * tau_scale,
        dtau=dtau * tau_scale,
        f=g,
        n_quad=n_quad,
    )
    C0_est = float(C_warm[0])
    mean_g = mu_g(C0_est)

    def g_centered(u):
        return g(u) - mean_g

    # Pass 2: solve with centered gain
    tau_s, C_s = theory_rate_autocorr(
        C0=None,
        sigma=sigma,
        tau_max=tau_max * tau_scale,
        dtau=dtau * tau_scale,
        f=g_centered,
        n_quad=n_quad,
    )
    return tau_s / tau_scale, C_s, sigma_c


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
        return alpha * np.maximum(I + u, 0.0) ** (1.0 / alpha)

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
):
    """Compare phase-model spike autocorrelation from simulation and theory."""
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
    plt.savefig("phase_spike_correlation_test.png", dpi=150)
    plt.show()


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
):
    """Compare phase-network simulation and reduced-theory autocorrelations."""
    rho = 1.0 / (2.0 * np.pi)
    Fprime0 = float(np.maximum(I, 1e-10) ** (1.0 / alpha - 1.0))
    sigma_c = 1.0 / (rho * Fprime0)
    print(f"Phase model: I={I}, alpha={alpha}, sigma_c={sigma_c:.3f}")

    if sigma_vals is None:
        sigma_vals = [0.75 * sigma_c, 0.95 * sigma_c, 1.1 * sigma_c]

    fig, axes = plt.subplots(1, len(sigma_vals), figsize=(5 * len(sigma_vals), 4))
    if len(sigma_vals) == 1:
        axes = [axes]

    for ax, sigma in zip(axes, sigma_vals):
        g_val = sigma / sigma_c
        print(f"\n-- phase sigma={sigma:.3f} (g={g_val:.2f}) --")

        if sigma < sigma_c:
            tau_th, C_th, _ = theory_phase_autocorr(
                I=I,
                alpha=alpha,
                sigma=sigma,
                beta=beta,
                tau_max=tau_max,
                dtau=dtau,
            )
        else:
            tau_th, C_th = None, None
            print("  above transition: no monotone decaying branch")

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

        def norm(x):
            x0 = float(x[0])
            return x / x0 if abs(x0) > 1e-12 else x

        ax.plot(tau_s, norm(C_s), "b", lw=1.5, alpha=0.85, label=f"Sim ({sim_reps} runs)")
        if C_th is not None:
            ax.plot(tau_th, norm(C_th), "r--", lw=2, label="2PI reduced")
        ax.axhline(0, color="k", lw=0.5)
        ax.set(
            xlabel=r"$\tau$",
            ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
            title=fr"$\sigma={sigma:.2f}$, $g={g_val:.2f}$",
            xlim=(0, tau_max),
        )
        ax.legend(fontsize=8)

    plt.suptitle(
        fr"Phase neuron network: $I={I}$, $\alpha={alpha}$, $\sigma_c={sigma_c:.2f}$",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("phase_network_test.png", dpi=150)
    plt.show()


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
    tau = np.arange(max_lag) * dt
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
    """
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
    s = sigma * np.sqrt(max(C_uu_0, 0.0))
    u = s * gh_x * np.sqrt(2.0)
    fu = np.maximum(f0 + f1 * u, 0.0)
    integrand = 2.0 * fu * mu / (fu + mu)
    return float(np.dot(gh_w, integrand) / np.sqrt(np.pi))


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

        # Effective coefficients from the current C_uu(0).
        gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
        s = sigma * np.sqrt(C_uu_0)
        u = s * gh_x * np.sqrt(2.0)
        fu = np.maximum(f0 + f1 * u, 0.0)
        nbar_eff = float(np.dot(gh_w, fu / (fu + mu)) / np.sqrt(np.pi))
        c1_eff = float(
            np.dot(
                gh_w,
                f1 * (fu > 0).astype(float) * mu * fu / (fu + mu) ** 2,
            )
            / np.sqrt(np.pi)
        )
        D0_eff = _D0_clipped(C_uu_0, f0, f1, mu, sigma, n_quad=n_quad)

        Q = np.array([
            _Q_clipped_correct(C_uu[k], C_uu_0, f0, f1, mu, sigma, n_quad=n_quad)
            for k in range(ntau)
        ])
        Q_hat = fft(Q)

        # Coupled 3-component update in frequency space.
        C_uu_hat = b2 * Q_hat / (b2 + w2)
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
    s = sigma * np.sqrt(C_uu_0)
    u = s * gh_x * np.sqrt(2.0)
    fu = np.maximum(f0 + f1 * u, 0.0)
    nbar_eff = float(np.dot(gh_w, fu / (fu + mu)) / np.sqrt(np.pi))
    c1_eff = float(
        np.dot(
            gh_w,
            f1 * (fu > 0).astype(float) * mu * fu / (fu + mu) ** 2,
        )
        / np.sqrt(np.pi)
    )
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
    sim_method="tau-leap",
    sim_cache_path=None,
    force_resim=False,
):
    """Compare clipped simulation with clipped-vs-linear theory predictions."""
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
                sigma, beta, mu, f0, f1, tau_max=20, dtau=0.05
            )
        else:
            tau_lin, Cnn_lin = None, None

        tau_clip, Cnn_clip, _Cuu_clip, g_eff = theory_binary_clipped(
            sigma, beta, mu, f0, f1, tau_max=20, dtau=0.05
        )

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
                clip_rate_on=True,
                method=sim_method,
            )
            cached[sigma] = (tau_s, Cnn_s, Cuu_s)
            cache_updated = True

        def norm(x):
            x0 = float(x[0])
            return x / x0 if abs(x0) > 1e-12 else x

        ax.plot(tau_s, norm(Cnn_s), "b", lw=1.5, alpha=0.8, label="Sim (clipped)")
        ax.plot(
            tau_clip,
            norm(Cnn_clip),
            "r--",
            lw=2,
            label=fr"2PI clipped ($g_{{eff}}={g_eff:.2f}$)",
        )
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
            xlim=(0, 15),
        )
        ax.legend(fontsize=8)

    plt.suptitle("Clipped vs linear gain: binary 2PI", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("clipped_vs_linear.png", dpi=150)
    plt.show()

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
):
    """
    For each sigma, compare simulation vs theory.
    Also shows the g=1 transition.
    """
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
    plt.tight_layout()
    plt.savefig("binary_network_test.png", dpi=150)
    plt.show()


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
):
    """
    Show that the simulated correlation function is well fit by two exponentials,
    with decay rates matching theory predictions kappa+ and kappa-.
    """
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
    plt.savefig("two_timescale_fit.png", dpi=150)
    plt.show()


# -----------------------------------------------------------------------------
# 4. CONVERGENCE WITH NETWORK SIZE
#    Plot correlation functions for binary network at different N values
#    against theory to visualize finite-size effects
# -----------------------------------------------------------------------------

def plot_binary_network_N_convergence(sigma=0.8, N_vals=(128, 300, 800, 1600),
                                       beta=1.0, mu=1.0, f0=0.5, f1=1.0,
                                       clip_rate_on=True, sim_method="tau-leap"):
    """
    Compare simulation vs theory for binary network at multiple network sizes.
    Shows how finite-size effects diminish as N increases.
    """
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
    plt.tight_layout()
    plt.savefig("binary_network_N_convergence.png", dpi=150)
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Rate network
    # sigma < 1 -> fixed point, sigma > 1 -> chaos for tanh
    plot_rate_network(sigma=1.5, N=512, C0_guess=0.65)

    # 2. Binary network: shape of correlation functions
    plot_binary_network(sigma_vals=(0.5, 0.8, 0.95), N=800)

    # 3. Two-timescale exponential fit
    plot_two_timescale_fit(sigma=0.8, N=800)

    # 4. Finite-size convergence
    plot_binary_network_N_convergence(sigma=0.8, N_vals=(128, 300, 800, 1600))

    # 5. Clipped-gain theory vs simulation
    plot_clipped_vs_linear(sigma_vals=(0.7, 1.0, 1.3), N=800)
