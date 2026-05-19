"""
Simulations to test the 2PI effective action theory for two random neural networks:
  1. Phase (rate) neuron network  -- tests the Sompolinsky-Crisanti-Sommers equation
  2. Binary neuron network        -- tests the exact two-exponential formula

Run each section independently. Requires numpy, scipy, matplotlib.
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from scipy.linalg import expm

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
# 1. PHASE / RATE NEURON NETWORK
#    Model: du_i/dt = -u_i + sum_j W_ij f(u_j)
#    Theory: C''(tau) = C(tau) - Q(tau), Q(tau) = <f(u(0))f(u(tau))>
# -----------------------------------------------------------------------------

def sim_rate_network(
    N=512, sigma=1.5, T=2000.0, dt=0.05, f=np.tanh, lam=1, burn=200, rng=rng
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

    # average autocorrelation over neurons
    max_lag = int(50 / dt)
    C = np.mean([autocorr(U[:, i], max_lag) for i in range(min(N, 64))], axis=0)
    tau = np.arange(max_lag) * dt
    return tau, C


def theory_rate_autocorr(
    C0=None,
    beta=1.0,
    sigma=1.5,
    tau_max=50,
    dtau=0.01,
    f=np.tanh,
    n_quad=20,
    C0_bounds=(0.05, 5.0),
):
    """
     Solve the SCS equation:
       C''(tau) = beta^2 [C(tau) - sigma^2 Q(tau)]
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

        # Since C decays monotonically, integrate d tau / dC = 1 / (beta sqrt(H(C))).
        C_desc = C_grid[::-1]
        H_desc = H_grid[::-1]
        speed = beta * np.sqrt(H_desc)
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
    tau_th, C_th = theory_rate_autocorr(C0_guess, sigma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(tau_sim, C_sim, "b", lw=1.5, label="Simulation")
    ax.plot(tau_th, C_th / C_th[0], "r--", lw=2, label="SCS theory")
    ax.set(
        xlabel=r"$\\tau$",
        ylabel=r"$C_{11}(\\tau)/C_{11}(0)$",
        title=f"Rate network  $\\sigma={sigma}$",
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
        xlabel=r"$\\omega / 2\\pi$",
        ylabel="Power spectrum",
        title="Power spectrum",
        xlim=(0, 2),
    )
    ax.legend()

    plt.suptitle("Phase/Rate network: SCS test", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("rate_network_test.png", dpi=150)
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
    rng=rng,
):
    """
    Gillespie-inspired Euler simulation of the binary network.

    f(u) = f0 + f1*u (linear gain, clipped to be non-negative)
    State: n_i in {0,1}, u_i continuous.

    At each step:
      - Each neuron independently flips with probability rate*dt
      - u_i updated via Euler for the ODE
    """
    W = make_weights(N, sigma, lam, rng)
    n = rng.integers(0, 2, N).astype(float)
    u = np.zeros(N)

    def rate_on(u_):
        return np.maximum(f0 + f1 * u_, 0.0)

    nb = int(burn / dt)
    for _ in range(nb):
        r_on = rate_on(u) * (1 - n)
        r_off = mu * n
        flip_on = rng.random(N) < r_on * dt
        flip_off = rng.random(N) < r_off * dt
        n += flip_on.astype(float) - flip_off.astype(float)
        n = np.clip(n, 0, 1)
        u += dt * (-beta * u + beta * (W @ n))

    nt = int(T / dt)
    N_rec = np.zeros((nt, N))
    U_rec = np.zeros((nt, N))
    for t in range(nt):
        r_on = rate_on(u) * (1 - n)
        r_off = mu * n
        flip_on = rng.random(N) < r_on * dt
        flip_off = rng.random(N) < r_off * dt
        n += flip_on.astype(float) - flip_off.astype(float)
        n = np.clip(n, 0, 1)
        u += dt * (-beta * u + beta * (W @ n))
        N_rec[t] = n
        U_rec[t] = u

    max_lag = int(30 / dt)
    Cnn = np.mean([autocorr(N_rec[:, i], max_lag) for i in range(min(N, 64))], axis=0)
    Cuu = np.mean([autocorr(U_rec[:, i], max_lag) for i in range(min(N, 64))], axis=0)
    tau = np.arange(max_lag) * dt
    return tau, Cnn, Cuu


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
    # Steady state
    gamma = mu + f0
    n_bar = f0 / gamma
    c1 = f1 * mu / gamma  # effective gain f'(0)(1-n_bar)
    D0 = 2 * n_bar * (1 - n_bar) * gamma  # intrinsic noise
    g = c1 * sigma / gamma  # dimensionless coupling

    print(f"  gamma={gamma:.3f}, n_bar={n_bar:.3f}, c1={c1:.3f}, g={g:.3f}, D0={D0:.3f}")
    if g >= 1:
        print(f"  WARNING: g={g:.3f} >= 1, network is above transition -> oscillatory.")

    # Pole locations
    disc = (gamma**2 - beta**2) ** 2 + 4 * c1**2 * beta**2 * sigma**2
    kp2 = 0.5 * ((gamma**2 + beta**2) + np.sqrt(disc))
    km2 = 0.5 * ((gamma**2 + beta**2) - np.sqrt(disc))

    tau = np.arange(0, tau_max, dtau)

    if km2 >= 0:  # g < 1: two real exponentials
        kp, km = np.sqrt(kp2), np.sqrt(km2)
        denom = kp * km * (km2 - kp2)  # note kp > km so this is negative
        Ap = D0 * 0.5 * (beta**2 - kp2) / (kp * (km2 - kp2))
        Am = D0 * 0.5 * (beta**2 - km2) / (km * (kp2 - km2))
        Cnn = Ap * np.exp(-kp * tau) + Am * np.exp(-km * tau)
        # C_uu via convolution in frequency: multiply spectral weight beta^2*sigma^2/(beta^2+w^2)
        # In time domain: C_uu = beta*sigma^2/2 * [Ap/kp * (e^{-kp|tau|} - e^{-beta|tau|})/(beta-kp)
        #                                          + Am/km * similar] for beta != kp, km
        def conv_exp(A, k):
            if abs(beta - k) < 1e-8:
                return A * beta * sigma**2 * tau * np.exp(-k * tau)
            return A * beta**2 * sigma**2 / (beta**2 - k**2) * (
                np.exp(-k * tau) - k / beta * np.exp(-beta * tau)
            )

        Cuu = conv_exp(Ap, kp) + conv_exp(Am, km)
    else:  # g > 1: oscillatory
        kp = np.sqrt(kp2)
        kr = np.sqrt(-km2)  # imaginary part
        Cnn = (D0 / kp) * np.exp(-kp * tau) * np.cos(kr * tau)
        Cuu = np.zeros_like(Cnn)  # not derived here

    return tau, Cnn, Cuu, g


def plot_binary_network(sigma_vals=(0.5, 0.8, 0.95), N=800, beta=1.0, mu=1.0, f0=0.5, f1=1.0):
    """
    For each sigma, compare simulation vs theory.
    Also shows the g=1 transition.
    """
    fig, axes = plt.subplots(2, len(sigma_vals), figsize=(5 * len(sigma_vals), 8))

    for col, sigma in enumerate(sigma_vals):
        print(f"\n-- sigma={sigma} --")
        tau_th, Cnn_th, Cuu_th, g = theory_binary_autocorr(sigma, beta, mu, f0, f1)

        print("  Simulating ...")
        tau_s, Cnn_s, Cuu_s = sim_binary_network(
            N=N, sigma=sigma, beta=beta, mu=mu, f0=f0, f1=f1
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
            xlabel=r"$\\tau$",
            ylabel=r"$C_{nn}(\\tau)/C_{nn}(0)$",
            title=fr"$\\sigma={sigma},\ g={g:.2f}$",
            xlim=(0, 20),
        )
        ax.legend(fontsize=8)

        # C_uu
        ax = axes[1, col]
        ax.plot(tau_s, Cuu_s, "g", lw=1.5, label="Sim")
        ax.plot(tau_th, Cuu_th_n, "m--", lw=2, label="Theory")
        ax.set(
            xlabel=r"$\\tau$",
            ylabel=r"$C_{uu}(\\tau)/C_{uu}(0)$",
            title=fr"$C_{{uu}}$,  $\\sigma={sigma}$",
            xlim=(0, 20),
        )
        ax.legend(fontsize=8)

    plt.suptitle("Binary neuron network: 2PI theory test", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("binary_network_test.png", dpi=150)
    plt.show()


# -----------------------------------------------------------------------------
# 3. TRANSITION SCAN
#    Sweep sigma and measure C_nn(0) (variance) vs theory prediction.
#    Should diverge as g -> 1.
# -----------------------------------------------------------------------------

def scan_transition(
    N=600, beta=1.0, mu=1.0, f0=0.5, f1=1.0, sigma_vals=None, T=3000, dt=0.02
):
    """
    Measure C_nn(0) from simulation and theory across a range of sigma.
    The variance should diverge as g = c1*sigma/gamma -> 1.
    """
    if sigma_vals is None:
        # sigma_crit = gamma/c1 = (mu+f0)^2 / (f1*mu)
        gamma = mu + f0
        c1 = f1 * mu / gamma
        s_crit = gamma / c1
        sigma_vals = np.linspace(0.1, 0.93 * s_crit, 14)

    Cnn0_sim = []
    Cnn0_th = []
    g_vals = []

    for sigma in sigma_vals:
        tau_th, Cnn_th, _, g = theory_binary_autocorr(
            sigma, beta, mu, f0, f1, tau_max=0.001
        )
        Cnn0_th.append(Cnn_th[0])
        g_vals.append(g)

        tau_s, Cnn_s, _ = sim_binary_network(
            N=N, sigma=sigma, beta=beta, mu=mu, f0=f0, f1=f1, T=T, dt=dt
        )
        Cnn0_sim.append(Cnn_s[0])
        print(
            f"sigma={sigma:.3f}  g={g:.3f}  C_sim(0)={Cnn_s[0]:.4f}  C_th(0)={Cnn_th[0]:.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(g_vals, Cnn0_sim, "bo-", ms=5, label="Simulation")
    ax.plot(g_vals, Cnn0_th, "r--", lw=2, label="Theory")
    ax.axvline(1, color="k", ls=":", lw=1.5, label="$g=1$ (transition)")
    ax.set(
        xlabel="$g = c_1\\sigma/\\gamma$",
        ylabel="$C_{nn}(0)$",
        title="Variance vs coupling strength",
    )
    ax.legend()

    ax = axes[1]
    ax.semilogy(g_vals, Cnn0_sim, "bo-", ms=5, label="Simulation")
    ax.semilogy(g_vals, Cnn0_th, "r--", lw=2, label="Theory")
    ax.axvline(1, color="k", ls=":", lw=1.5)
    ax.set(
        xlabel="$g$",
        ylabel="$C_{nn}(0)$  [log scale]",
        title="Variance divergence at transition",
    )
    ax.legend()

    plt.suptitle("Binary network: variance scan across transition", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("binary_transition_scan.png", dpi=150)
    plt.show()


# -----------------------------------------------------------------------------
# 4. TWO-TIMESCALE FIT
#    Fit the simulated C_nn(tau) to the two-exponential form
#    A+ exp(-kappa+ tau) + A- exp(-kappa- tau) and compare with theory.
# -----------------------------------------------------------------------------

def fit_two_exponentials(tau, C):
    """Least-squares fit of C(tau) = A+ e^{-k+ tau} + A- e^{-k- tau}."""
    from scipy.optimize import curve_fit

    def model(t, Ap, kp, Am, km):
        return Ap * np.exp(-kp * t) + Am * np.exp(-km * t)

    C0 = C[0]
    p0 = [C0 * 0.5, 2.0, C0 * 0.5, 0.3]
    bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(model, tau, C, p0=p0, bounds=bounds, maxfev=5000)
    except Exception:
        popt = p0
    return popt


def plot_two_timescale_fit(sigma=0.8, N=800, beta=1.0, mu=1.0, f0=0.5, f1=1.0):
    """
    Show that the simulated correlation function is well fit by two exponentials,
    with decay rates matching theory predictions kappa+ and kappa-.
    """
    tau_th, Cnn_th, _, g = theory_binary_autocorr(sigma, beta, mu, f0, f1)
    tau_s, Cnn_s, _ = sim_binary_network(
        N=N, sigma=sigma, beta=beta, mu=mu, f0=f0, f1=f1
    )
    popt = fit_two_exponentials(tau_s[1:], Cnn_s[1:])
    Ap, kp_fit, Am, km_fit = popt

    # Theory kappa values
    gamma = mu + f0
    c1 = f1 * mu / gamma
    disc = (gamma**2 - beta**2) ** 2 + 4 * c1**2 * beta**2 * sigma**2
    kp_th = np.sqrt(0.5 * ((gamma**2 + beta**2) + np.sqrt(disc)))
    km_th = np.sqrt(0.5 * ((gamma**2 + beta**2) - np.sqrt(disc)))

    fit_curve = Ap * np.exp(-kp_fit * tau_s) + Am * np.exp(-km_fit * tau_s)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tau_s, Cnn_s, "b", lw=1.5, alpha=0.7, label="Simulation")
    ax.plot(
        tau_s,
        fit_curve,
        "r--",
        lw=2,
        label=fr"Fit: $\\kappa_+={kp_fit:.3f}$, $\\kappa_-={km_fit:.3f}$",
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set(
        xlabel=r"$\\tau$",
        ylabel=r"$C_{nn}(\\tau)$",
        title=fr"Two-timescale structure: $\\sigma={sigma}$, $g={g:.2f}$",
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
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Rate / phase network
    # sigma < 1 -> fixed point, sigma > 1 -> chaos for tanh
    plot_rate_network(sigma=1.5, N=512, C0_guess=0.65)

    # 2. Binary network: shape of correlation functions
    plot_binary_network(sigma_vals=(0.5, 0.8, 0.95), N=800)

    # 3. Variance divergence at transition
    scan_transition(N=600)

    # 4. Two-timescale exponential fit
    plot_two_timescale_fit(sigma=0.8, N=800)
