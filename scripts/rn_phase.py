"""Phase/spiking-network simulations, theory closures, and plots."""

import os

import numpy as np
import matplotlib.pyplot as plt

from rn_core import autocorr, default_results_dir, make_weights, rng

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
    synapse_update="exact",
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
    synapse_update = str(synapse_update).lower()
    if synapse_update not in ("exact", "euler"):
        raise ValueError("synapse_update must be 'exact' or 'euler'")

    def F(u_):
        return alpha * np.clip(I + u_, 0.0, 1e12) ** (1.0 / alpha)

    def spike_weights(phi_old, rate, spike_counts):
        """Return per-neuron spike counts, optionally filtered by event time."""
        if synapse_update == "euler":
            return spike_counts
        weighted = np.zeros_like(spike_counts, dtype=float)
        spiking = np.flatnonzero(spike_counts > 0)
        for j in spiking:
            count = int(spike_counts[j])
            if count <= 0 or rate[j] <= 0.0:
                continue
            thresholds = np.pi + 2.0 * np.pi * np.arange(count)
            crossing_times = (thresholds - phi_old[j]) / rate[j]
            crossing_times = np.clip(crossing_times, 0.0, dt)
            weighted[j] = np.sum(np.exp(-beta * (dt - crossing_times)))
        return weighted

    def update_u(u_, drive_):
        # Spikes are delta events: integrating beta*W*dN gives beta*W*count,
        # not beta*W*count*dt.  The exact option also applies exponential leak
        # over the step, including within-step decay from estimated spike times.
        if synapse_update == "exact":
            return np.exp(-beta * dt) * u_ + beta * drive_
        return u_ + (-beta * u_) * dt + beta * drive_

    nb = int(burn / dt)
    for _ in range(nb):
        phi_old = phi.copy()
        rate = F(u)
        phi = phi_old + rate * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        # Correct multi-spike: wrap phi into (-pi, pi) regardless of how many
        # cycles were completed in this step (F(u)*dt can exceed 2*pi for large u).
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        filtered_counts = spike_weights(phi_old, rate, spike_counts)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            drive = W @ filtered_counts
        u = update_u(u, drive)
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
        phi_old = phi.copy()
        rate = F(u)
        phi = phi_old + rate * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        filtered_counts = spike_weights(phi_old, rate, spike_counts)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            drive = W @ filtered_counts
        u = update_u(u, drive)
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
    solver="inflated_ic",
    kernel_omega=0.0,
    kernel_damping=0.0,
    kernel_scaled_by_beta=True,
    q_method="gh",
    n_qmc=2048,
    hermite_order=32,
    warn_on_no_branch=True,
):
    """
    Solve approximate 2PI closures for the phase neuron network with lambda=1.

    With row-sum correction (lambda=1), W@1=0 so E[u_i]=0 exactly, meaning
    C_11(tau->inf)=0 and C_eq=0. The centered SCS with gain g is directly correct:
    no g_shifted or C_eq machinery needed.

    The filtered shot-noise covariance is treated as a homogeneous contribution
    that inflates C(0); the smooth ODE is integrated with C'(0)=0.  This is the
    only active scalar phase closure in the code.

    A minimal generalized phase kernel can be included,

        L_C C = C'' + 2*kernel_damping*C'
                + (kernel_omega**2 - beta**2)*C,

    so the tau>0 equation is

        L_C C = -beta**2 * Q_smooth(C; C0).

    kernel_omega=kernel_damping=0 recovers the rate-like SCS reduction.  By
    default the generalized-kernel parameters are dimensionless multiples of
    beta, so kernel_omega=2 means omega=2*beta and kernel_damping=1 means
    damping=beta.  Set kernel_scaled_by_beta=False to pass raw time units.
    q_method="gh" uses tensor Gauss-Hermite quadrature for Q_smooth. q_method="qmc"
    uses common-random Sobol Gaussian samples, which is slower but more robust
    for hard rectification and strongly nonlinear gains. q_method="hermite" uses
    a 1-D Hermite expansion of g(u) and evaluates the centered covariance as a
    series in C(tau)/C(0), avoiding cancellation from subtracting E[g]^2.
    The smooth Gaussian closure averages over phases and therefore does not
    retain the near-threshold C33 advection peaks described in the notes.  Those
    peaks require a richer phase-density closure than this scalar C_uu solver.
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

        # Avoid unbounded cache growth during nonlinear solves and scans.
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

    def smooth_energy(C_val, C0_val, n_grid=384):
        """H(C; C0) = C^2 - 2 int_0^C Q_smooth(x; C0) dx."""
        C_val = float(C_val)
        C0_val = float(C0_val)
        if C_val <= 0.0 or C0_val <= 0.0:
            return np.nan
        C_grid = np.linspace(0.0, C_val, n_grid)
        Q_grid = np.array([Q_centered(c, C0_val) for c in C_grid])
        return float(C_val**2 - 2.0 * np.trapezoid(Q_grid, C_grid))

    ntau = int(tau_max / dtau)
    tau = np.arange(ntau) * dtau

    solver_key = str(solver).lower()
    if solver_key not in ("inflated_ic", "inflated-ic", "inflated"):
        raise ValueError("phase theory supports solver='inflated_ic'")

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
            return np.nan

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

    return tau, inflated_ic_solution(), sigma_c


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


def theory_phase_threshold_ringing(
    I=1.0,
    alpha=1.0,
    beta=1.0,
    sigma=3.0,
    tau_max=40.0,
    dtau=0.02,
    C0_total=None,
    n_quad=48,
    n_samples=4096,
    max_harmonic=None,
    peak_width=None,
):
    """Toy C33 threshold-return ringing theory for the subcritical phase model.

    This is not the scalar Gaussian SCS closure.  It approximates the
    single-neuron spike-train autocorrelation by averaging a threshold-return
    delta comb over a Gaussian distribution of quasi-static drives:

        C_ring(tau) ∝ E[F(u)^2 sum_m delta(tau - 2*pi*m/F(u))].

    The delta peaks are plotted with a narrow Gaussian width.  This is meant as
    a diagnostic of the C33 advection mechanism below/near criticality, where
    the scalar phase closure intentionally smooths this structure away.
    """
    tau = np.arange(0.0, tau_max, dtau)
    rho = 1.0 / (2.0 * np.pi)

    def F(u):
        return alpha * np.clip(I + u, 0.0, 1e12) ** (1.0 / alpha)

    if C0_total is None:
        # Subcritical variance floor from filtered shot noise, solved by a
        # simple fixed point C = beta*sigma^2*rho*E[F(u)]/2.
        gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)

        def mean_F(C):
            if C <= 0.0:
                return float(F(0.0))
            return float(np.dot(gh_w, F(np.sqrt(2.0 * C) * gh_x)) / np.sqrt(np.pi))

        C = max(1e-12, 0.5 * beta * sigma**2 * rho * float(F(0.0)))
        for _ in range(100):
            C_new = max(1e-12, 0.5 * beta * sigma**2 * rho * mean_F(C))
            if abs(C_new - C) <= 1e-9 * max(1.0, C):
                C = C_new
                break
            C = 0.5 * C + 0.5 * C_new
        C0_total = C

    try:
        from scipy.special import ndtri

        p = (np.arange(int(n_samples)) + 0.5) / float(n_samples)
        z = ndtri(p)
        weights = np.full_like(z, 1.0 / len(z), dtype=float)
    except Exception:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(n_quad)
        z = np.sqrt(2.0) * gh_x
        weights = gh_w / np.sqrt(np.pi)
    u_nodes = np.sqrt(max(float(C0_total), 0.0)) * z
    rates = F(u_nodes)
    valid = rates > 1e-12
    rates = rates[valid]
    weights = weights[valid]

    if peak_width is None:
        peak_width = max(5.0 * dtau, 0.12 / max(float(beta), 1e-12))
    if max_harmonic is None:
        min_period = 2.0 * np.pi / max(float(np.max(rates)), 1e-12)
        max_harmonic = int(np.ceil(tau_max / max(min_period, 1e-12))) + 1

    C = np.zeros_like(tau)
    for w, rate in zip(weights, rates):
        period = 2.0 * np.pi / rate
        amp = w * (rho * rate) ** 2
        for m in range(1, max_harmonic + 1):
            center = m * period
            if center > tau_max + 4.0 * peak_width:
                break
            C += amp * np.exp(-0.5 * ((tau - center) / peak_width) ** 2)

    # Remove an approximate baseline so the curve is visually comparable to a
    # centered autocorrelation.  The remaining positive peaks mark returns.
    if len(C) > 0:
        tail = C[int(0.7 * len(C)):] if len(C) > 10 else C
        C = C - float(np.median(tail))
    return tau, C, float(C0_total)


def filter_spike_cov_to_drive(tau, Q_tau, beta=1.0):
    """Filter an even spike/input covariance through beta exp(-beta t)."""
    tau = np.asarray(tau, dtype=float)
    Q_tau = np.asarray(Q_tau, dtype=float)
    if len(tau) < 3:
        return np.zeros_like(Q_tau)
    dt = float(np.median(np.diff(tau)))
    # Even extension: [0, +tau, ..., reflected positive lags].
    Q_even = np.concatenate([Q_tau, Q_tau[-2:0:-1]])
    omega = 2.0 * np.pi * np.fft.fftfreq(len(Q_even), d=dt)
    C_hat = (float(beta) ** 2) * np.fft.fft(Q_even) / (float(beta) ** 2 + omega**2)
    return np.real(np.fft.ifft(C_hat))[: len(Q_tau)]


def fit_spike_return_comb(tau, Cspk, period_guess=None, max_harmonic=None):
    """Fit spike autocovariance with a damped Gaussian return comb.

    Model:
        baseline + amp * sum_m exp(-decay*m*T)
                     exp[-(tau-m*T)^2/(2*width^2)]

    This is an empirical fit to the threshold-return structure.  It is useful
    for asking whether the fitted spike covariance, after synaptic filtering,
    can reproduce C_uu.
    """
    from scipy.optimize import curve_fit

    tau = np.asarray(tau, dtype=float)
    Cspk = np.asarray(Cspk, dtype=float)
    mask = np.isfinite(tau) & np.isfinite(Cspk) & (tau > 0.0)
    t = tau[mask]
    y = Cspk[mask]
    if len(t) < 20:
        return np.zeros_like(tau), {}

    # Smooth only for initial guesses; fit uses the raw covariance.
    dt = float(np.median(np.diff(tau)))
    y_scale = max(float(np.nanmax(np.abs(y))), 1e-12)
    y_norm = y / y_scale

    if period_guess is None:
        # Pick the first prominent positive peak after zero.
        search = (t > 1.0) & (t < min(float(t[-1]), 12.0))
        if np.any(search):
            period_guess = float(t[search][np.argmax(y_norm[search])])
        else:
            period_guess = 2.0 * np.pi
    period_guess = max(float(period_guess), 4.0 * dt)

    if max_harmonic is None:
        max_harmonic = int(np.ceil(float(t[-1]) / period_guess)) + 2

    def model(tt, amp, period, width, decay, baseline):
        out = np.full_like(tt, baseline, dtype=float)
        period = max(period, 4.0 * dt)
        width = max(width, dt)
        for m in range(1, max_harmonic + 1):
            center = m * period
            if center > float(t[-1]) + 5.0 * width:
                break
            out += amp * np.exp(-decay * m) * np.exp(-0.5 * ((tt - center) / width) ** 2)
        return out

    p0 = [
        max(float(np.nanmax(y_norm)), 1e-3),
        period_guess,
        max(3.0 * dt, 0.20),
        0.15,
        float(np.nanmedian(y_norm)),
    ]
    bounds = (
        [0.0, max(4.0 * dt, 0.2 * period_guess), dt, 0.0, -1.0],
        [5.0, min(float(t[-1]), 2.5 * period_guess), max(2.0, period_guess), 4.0, 1.0],
    )
    try:
        popt, _ = curve_fit(model, t, y_norm, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        popt = np.array(p0, dtype=float)

    fitted = np.zeros_like(tau, dtype=float)
    fitted[mask] = y_scale * model(t, *popt)
    fitted[~mask] = y_scale * model(tau[~mask], *popt) if np.any(~mask) else fitted[~mask]
    info = dict(
        amp=float(popt[0] * y_scale),
        period=float(popt[1]),
        width=float(popt[2]),
        decay=float(popt[3]),
        baseline=float(popt[4] * y_scale),
    )
    return fitted, info


def _sim_phase_probe_correlations(
    N=256,
    I=1.0,
    alpha=1.0,
    sigma=3.0,
    beta=1.0,
    T=1200.0,
    dt=0.02,
    tau_max=30.0,
    burn=300.0,
    n_probe=96,
    job_rng=rng,
):
    """Simulate phase network and return per-neuron C_uu and spike autocovariance."""
    W = make_weights(N, sigma, 1, job_rng)
    phi = job_rng.uniform(-np.pi, np.pi, N)
    u = np.zeros(N)
    n_probe = int(max(1, min(N, n_probe)))
    probe_idx = np.arange(n_probe)

    def F(u_):
        return alpha * np.clip(I + u_, 0.0, 1e12) ** (1.0 / alpha)

    def step(phi_, u_):
        phi_old = phi_.copy()
        rate = F(u_)
        phi_new = phi_old + rate * dt
        spike_counts = np.floor((phi_new + np.pi) / (2.0 * np.pi)).astype(float)
        spiking = spike_counts > 0
        phi_new[spiking] = ((phi_new[spiking] + np.pi) % (2.0 * np.pi)) - np.pi

        weighted = np.zeros_like(spike_counts, dtype=float)
        for j in np.flatnonzero(spiking):
            count = int(spike_counts[j])
            if count <= 0 or rate[j] <= 0.0:
                continue
            thresholds = np.pi + 2.0 * np.pi * np.arange(count)
            crossing_times = np.clip((thresholds - phi_old[j]) / rate[j], 0.0, dt)
            weighted[j] = np.sum(np.exp(-beta * (dt - crossing_times)))
        u_new = np.exp(-beta * dt) * u_ + beta * (W @ weighted)
        if not np.all(np.isfinite(u_new)):
            u_new = np.nan_to_num(u_new, nan=0.0, posinf=1e6, neginf=-1e6)
        return phi_new, u_new, spike_counts

    for _ in range(int(burn / dt)):
        phi, u, _ = step(phi, u)

    nt = int(T / dt)
    max_lag = int(tau_max / dt)
    U = np.zeros((nt, n_probe))
    S = np.zeros((nt, n_probe))
    for k in range(nt):
        phi, u, spike_counts = step(phi, u)
        U[k] = u[probe_idx]
        S[k] = spike_counts[probe_idx] / dt

    Cuu = np.mean([autocorr(U[:, i], max_lag) for i in range(n_probe)], axis=0)
    Cspk = np.mean([autocorr(S[:, i], max_lag) for i in range(n_probe)], axis=0)
    tau = np.arange(max_lag) * dt
    return tau, Cuu, Cspk


def infer_Q_from_Cuu(tau, Cuu, beta=1.0, smooth_window=41, polyorder=3):
    """Infer effective spike-input covariance Q from C''=beta^2(C-Q).

    The derivative is regularized with a Savitzky-Golay filter when SciPy is
    available.  This is a diagnostic estimate for figures, not a replacement
    for the microscopic Q closure.
    """
    tau = np.asarray(tau, dtype=float)
    Cuu = np.asarray(Cuu, dtype=float)
    if len(tau) < 5:
        return np.zeros_like(Cuu)
    dt = float(np.median(np.diff(tau)))
    try:
        from scipy.signal import savgol_filter

        win = int(smooth_window)
        win = min(win, len(Cuu) - (1 - len(Cuu) % 2))
        if win % 2 == 0:
            win -= 1
        win = max(win, polyorder + 2 + (polyorder + 2) % 2)
        if win >= len(Cuu):
            win = len(Cuu) - 1 if len(Cuu) % 2 == 0 else len(Cuu)
        if win < polyorder + 2:
            Cpp = np.gradient(np.gradient(Cuu, dt), dt)
        else:
            C_s = savgol_filter(Cuu, win, polyorder, mode="interp")
            Cpp = savgol_filter(Cuu, win, polyorder, deriv=2, delta=dt, mode="interp")
            # Keep the original equal-time value but use the smoothed curvature.
            Cuu = C_s
    except Exception:
        Cpp = np.gradient(np.gradient(Cuu, dt), dt)
    return Cuu - Cpp / max(float(beta) ** 2, 1e-12)


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
        plot_dir = default_results_dir()
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


def phase_shot_renormalized_criticality(
    I=1.0,
    alpha=1.0,
    beta=1.0,
    n_quad=96,
    C_bounds=(1e-10, 1e4),
    n_scan=240,
    integration="quad",
):
    """
    Static smooth-feedback threshold dressed by shot-noise variance.

    This solves the coupled equations from the shot-noise-inflated linear
    theory:

        1 = sigma_c^2 * rho^2 * <F'(u)>_C^2
        C = beta * sigma_c^2 * rho * <F(u)>_C / 2

    with u~N(0,C).  The result is a static criterion for the smooth Gaussian
    closure, not the operational finite-branch threshold of a full nonlinear
    solve.  Returns a dict so callers can compare it to the old smooth
    threshold and to branch-existence scans.
    """
    from scipy.integrate import quad
    from scipy.optimize import brentq

    rho_phase = 1.0 / (2.0 * np.pi)
    integration = str(integration).lower()
    gh_x, gh_w = np.polynomial.hermite.hermgauss(int(n_quad))
    gh_w = gh_w / np.sqrt(np.pi)
    exponent = 1.0 / float(alpha)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    def moments(C):
        C = float(max(C, 0.0))
        if C <= 0.0:
            x = max(float(I), 0.0)
            mean_F = float(alpha) * x ** exponent
            mean_Fp = x ** (exponent - 1.0) if x > 0.0 else 0.0
            return mean_F, mean_Fp

        if integration in ("quad", "adaptive"):
            s = np.sqrt(C)
            z0 = -float(I) / s

            def normal_pdf(z):
                return inv_sqrt_2pi * np.exp(-0.5 * z * z)

            def f_integrand(z):
                x = float(I) + s * z
                return float(alpha) * x ** exponent * normal_pdf(z)

            def fp_integrand(z):
                x = float(I) + s * z
                return x ** (exponent - 1.0) * normal_pdf(z)

            mean_F = quad(f_integrand, z0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)[0]
            mean_Fp = quad(fp_integrand, z0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)[0]
            return float(mean_F), float(mean_Fp)

        u = np.sqrt(2.0 * C) * gh_x
        x = I + u
        active = x > 0.0
        x_pos = np.where(active, x, 1.0)
        F = np.where(active, float(alpha) * x_pos ** exponent, 0.0)
        Fp = np.where(active, x_pos ** (exponent - 1.0), 0.0)
        return float(np.dot(gh_w, F)), float(np.dot(gh_w, Fp))

    def sigma_from_C(C):
        _mean_F, mean_Fp = moments(C)
        if mean_Fp <= 0.0 or not np.isfinite(mean_Fp):
            return np.nan
        return 1.0 / (rho_phase * mean_Fp)

    def balance(C):
        mean_F, mean_Fp = moments(C)
        if mean_Fp <= 0.0 or not np.isfinite(mean_Fp):
            return np.nan
        sigma_c = 1.0 / (rho_phase * mean_Fp)
        shot_var = 0.5 * float(beta) * sigma_c**2 * rho_phase * mean_F
        return float(C - shot_var)

    lo, hi = map(float, C_bounds)
    grid = np.unique(
        np.sort(
            np.concatenate(
                [
                    np.linspace(lo, min(hi, 20.0), max(4, n_scan // 2)),
                    np.logspace(np.log10(lo), np.log10(hi), max(4, n_scan // 2)),
                ]
            )
        )
    )
    vals = np.array([balance(c) for c in grid])
    finite = np.isfinite(vals)
    grid, vals = grid[finite], vals[finite]

    roots = []
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            roots.append(float(grid[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            roots.append(float(brentq(balance, grid[i], grid[i + 1], maxiter=100)))

    if not roots:
        return dict(
            sigma_smooth=_phase_sigma_c(I, alpha),
            sigma_shot=np.nan,
            g_shot=np.nan,
            C_shot=np.nan,
            mean_F=np.nan,
            mean_Fp=np.nan,
            roots=[],
        )

    # Use the smallest positive root, continuously connected to the onset.
    C_star = float(min(r for r in roots if r > 0.0))
    mean_F, mean_Fp = moments(C_star)
    sigma_shot = sigma_from_C(C_star)
    sigma_smooth = _phase_sigma_c(I, alpha)
    return dict(
        sigma_smooth=sigma_smooth,
        sigma_shot=sigma_shot,
        g_shot=sigma_shot / sigma_smooth,
        C_shot=C_star,
        mean_F=mean_F,
        mean_Fp=mean_Fp,
        roots=roots,
    )


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
        plot_dir = default_results_dir()
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
    show_shot_renormalized=True,
):
    """Plot smooth, shot-renormalized, and branch-existence criticalities."""
    import os

    if plot_dir is None:
        plot_dir = default_results_dir()
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

    if show_shot_renormalized:
        shot_rows = [
            phase_shot_renormalized_criticality(I=I, alpha=a, beta=beta)
            for a in alphas
        ]
        sigma_shot = np.array([r["sigma_shot"] for r in shot_rows], dtype=float)
        g_shot = np.array([r["g_shot"] for r in shot_rows], dtype=float)
    else:
        sigma_shot = g_shot = None

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    axes[0].plot(
        alphas, sigma_smooth, "ko-", lw=1.8,
        label=r"smooth-feedback theory $\sigma_c^{\rm smooth}$",
    )
    if sigma_shot is not None:
        axes[0].plot(
            alphas, sigma_shot, "ms-", lw=1.8,
            label=r"shot-renormalized theory $\sigma_c^{\rm shot}$",
        )
    axes[0].plot(
        alphas[~censored], sigma_branch[~censored], "ro-", lw=1.8,
        label=r"scalar branch theory $\sigma_\ast$",
    )
    if np.any(censored):
        axes[0].plot(alphas[censored], sigma_branch[censored], "r^", ms=8,
                     label=r"scalar branch lower bound")
    axes[0].set(xlabel=r"$\alpha$", ylabel=r"critical $\sigma$")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        alphas[~censored], g_branch[~censored], "ro-", lw=1.8,
        label=r"scalar branch theory $g_\ast$",
    )
    if g_shot is not None:
        axes[1].plot(
            alphas, g_shot, "ms-", lw=1.8,
            label=r"shot-renormalized theory",
        )
    if np.any(censored):
        axes[1].plot(
            alphas[censored], g_branch[censored], "r^", ms=8,
            label=r"scalar branch lower bound",
        )
    axes[1].axhline(
        1.0, color="k", ls=":", lw=1.2,
        label=r"smooth-feedback theory $g=1$",
    )
    axes[1].set(
        xlabel=r"$\alpha$",
        ylabel=r"$g_\ast=\sigma_\ast/\sigma_c^{\rm smooth}$",
        ylim=(
            0,
            max(
                1.2,
                np.nanmax(
                    np.concatenate(
                        [
                            g_branch[np.isfinite(g_branch)],
                            g_shot[np.isfinite(g_shot)] if g_shot is not None else np.array([]),
                        ]
                    )
                )
                * 1.15
                if (
                    np.any(np.isfinite(g_branch))
                    or (g_shot is not None and np.any(np.isfinite(g_shot)))
                )
                else 1.2,
            ),
        ),
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    plt.suptitle(
        "Phase criticality: smooth, shot-renormalized, and finite-branch theory",
        fontsize=13, fontweight="bold",
    )
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
    burn=300.0,
    scale_slow_beta_time=True,
    plot_dir=None,
):
    """Compare phase-model spike autocorrelation from simulation and theory."""
    import os
    if plot_dir is None:
        plot_dir = default_results_dir()
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

    slow_factor = _beta_time_factor(beta) if scale_slow_beta_time else 1.0
    tau_s, _Cuu_s, Cspk_s = sim_phase_network(
        N=N,
        I=I,
        alpha=alpha,
        sigma=sigma,
        beta=beta,
        T=T * slow_factor,
        dt=dt,
        burn=burn * slow_factor,
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


def plot_phase_subcritical_ringing(
    I=1.0,
    alpha=1.0,
    beta=1.0,
    sigma=None,
    N=256,
    T=1200.0,
    dt=0.02,
    tau_max=30.0,
    burn=300.0,
    n_probe=96,
    plot_dir=None,
):
    """Show below-critical threshold-return ringing in per-neuron spike trains.

    The left panel is the per-neuron spike-train autocorrelation, averaged over
    probe neurons, with a toy C33 threshold-return comb overlaid.  The right
    panel shows C_uu for the same simulation and the scalar inflated-IC theory.
    This separates the microscopic phase-density ringing from the smoothed
    scalar drive closure.
    """
    import os
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    sigma_c = _phase_sigma_c(I, alpha)
    if sigma is None:
        sigma = 0.55 * sigma_c

    tau, Cuu, Cspk = _sim_phase_probe_correlations(
        N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
        T=T, dt=dt, tau_max=tau_max, burn=burn, n_probe=n_probe,
    )

    tau_ring, C_ring, C_floor = theory_phase_threshold_ringing(
        I=I, alpha=alpha, beta=beta, sigma=sigma, tau_max=tau_max, dtau=dt,
    )
    tau_th, C_th, _ = theory_phase_autocorr(
        I=I, alpha=alpha, beta=beta, sigma=sigma,
        tau_max=tau_max, dtau=max(dt, 0.05),
        solver="inflated_ic", q_method="hermite", n_quad=48, hermite_order=32,
        warn_on_no_branch=False,
    )

    def norm_after_zero(x):
        if len(x) <= 1:
            return x
        scale = np.nanmax(np.abs(x[1:]))
        return x / scale if scale > 1e-12 else x

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(tau[1:], norm_after_zero(Cspk)[1:], color="C0", lw=1.4,
                 label="simulation: per-neuron spikes")
    axes[0].plot(tau_ring[1:], norm_after_zero(C_ring)[1:], color="C3", ls="--", lw=2.0,
                 label=r"$C_{33}$ threshold-return comb")
    axes[0].set(
        xlabel=r"$\tau$",
        ylabel="normalized spike autocovariance",
        title=fr"Below critical: $\sigma/\sigma_c={sigma/sigma_c:.2f}$",
        xlim=(0, tau_max),
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    Cuu_norm = Cuu / max(abs(Cuu[0]), 1e-12)
    axes[1].plot(tau, Cuu_norm, color="C0", lw=1.5, label=r"simulation $C_{uu}$")
    if np.all(np.isfinite(C_th)):
        axes[1].plot(tau_th, C_th / max(abs(C_th[0]), 1e-12),
                     color="C2", ls="--", lw=2.0, label="scalar inflated-IC theory")
    axes[1].set(
        xlabel=r"$\tau$",
        ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
        title=fr"Drive covariance; shot floor $C_{{sn}}\approx{C_floor:.2f}$",
        xlim=(0, tau_max),
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    plt.suptitle("Phase model below criticality: C33 ringing versus scalar closure",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_subcritical_ringing.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")


def plot_phase_transition_ringing_sweep(
    I=1.0,
    alpha=1.0,
    beta=1.0,
    g_vals=(0.5, 0.8, 1.0, 1.2, 1.5),
    N=192,
    T=900.0,
    dt=0.02,
    tau_max=25.0,
    burn=300.0,
    n_probe=96,
    plot_dir=None,
):
    """Sweep below-to-above criticality using spike autocovariance and C_uu.

    Top row: per-neuron spike-train autocovariance, where the threshold-return
    C33 ringing should be visible below criticality.  The black dashed curve is
    Q inferred from the measured drive covariance via C''=beta^2(C-Q), and the
    red dotted curve is a fitted C33 return comb.

    Bottom row: measured drive covariance and the synaptically filtered fitted
    spike covariance.
    """
    import os
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    sigma_c = _phase_sigma_c(I, alpha)
    ncols = len(g_vals)
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 6.2), sharex=True)
    if ncols == 1:
        axes = np.array(axes).reshape(2, 1)

    def norm_tail(x):
        if len(x) <= 1:
            return x
        scale = np.nanmax(np.abs(x[1:]))
        return x / scale if scale > 1e-12 else x

    for col, g_val in enumerate(g_vals):
        sigma = float(g_val) * sigma_c
        print(f"  transition sweep: g={g_val:.2f}, sigma={sigma:.3f}")
        tau_s, Cuu_s, Cspk_s = _sim_phase_probe_correlations(
            N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
            T=T, dt=dt, tau_max=tau_max, burn=burn, n_probe=n_probe,
            job_rng=np.random.default_rng(1000 + col),
        )

        Q_inferred = infer_Q_from_Cuu(tau_s, Cuu_s, beta=beta)
        tau_ring, C_ring_raw, C_floor = theory_phase_threshold_ringing(
            I=I, alpha=alpha, beta=beta, sigma=sigma,
            tau_max=tau_max, dtau=dt, n_quad=48,
        )
        # Use the C33 comb as a period-informed prior, but fit the actual
        # simulated spike autocovariance.  This avoids presenting a rough
        # quadrature comb as a quantitative theory curve.
        period_guess = 2.0 * np.pi / max(1e-12, alpha * max(I, 1e-12) ** (1.0 / alpha))
        Cspk_fit, fit_info = fit_spike_return_comb(tau_s, Cspk_s, period_guess=period_guess)
        Cuu_ring = filter_spike_cov_to_drive(tau_s, sigma**2 * Cspk_fit, beta=beta)
        Cuu_shot_basis = np.exp(-float(beta) * tau_s)
        X = np.column_stack([Cuu_shot_basis, Cuu_ring])
        try:
            from scipy.optimize import nnls

            coeffs, _ = nnls(X, Cuu_s)
        except Exception:
            coeffs, *_ = np.linalg.lstsq(X, Cuu_s, rcond=None)
        Cuu_from_fit = X @ coeffs

        ax = axes[0, col]
        ax.plot(tau_s[1:], norm_tail(Cspk_s)[1:], color="C0", lw=1.2, label="sim spikes")
        ax.plot(tau_s[1:], norm_tail(Q_inferred)[1:],
                color="k", ls="--", lw=1.4, label=r"inferred $Q=C-C''/\beta^2$")
        ax.plot(tau_s[1:], norm_tail(Cspk_fit)[1:],
                color="C3", ls=":", lw=1.8, label=r"fitted $C_{33}$ comb")
        ax.axhline(0, color="0.2", lw=0.5)
        ax.set_title(fr"$g=\sigma/\sigma_c={g_val:.1f}$")
        ax.set_xlim(0, tau_max)
        ax.set_ylim(-0.45, 1.05)
        if col == 0:
            ax.set_ylabel("spike autocov.")
            ax.legend(fontsize=7)

        ax = axes[1, col]
        ax.plot(tau_s, Cuu_s / max(abs(Cuu_s[0]), 1e-12),
                color="C0", lw=1.2, label=r"sim $C_{uu}$")
        if np.nanmax(np.abs(Cuu_from_fit)) > 1e-12:
            ax.plot(tau_s, Cuu_from_fit / max(abs(Cuu_from_fit[0]), 1e-12),
                    color="C3", ls=":", lw=1.8, label=r"shot + filtered comb")
        ax.axhline(0, color="0.2", lw=0.5)
        ax.set_xlabel(r"$\tau$")
        ax.set_ylim(-0.45, 1.05)
        if col == 0:
            ax.set_ylabel(r"$C_{uu}/C_{uu}(0)$")
            ax.legend(fontsize=7)

    plt.suptitle("Phase transition sweep: fitted spike Q and Cuu prediction",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_transition_ringing_sweep.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
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
    burn=300.0,
    sim_reps=3,
    plot_dir=None,
    theory_kwargs=None,
    scale_slow_beta_time=True,
):
    """Compare phase-network simulation and reduced-theory autocorrelations."""
    import os
    if plot_dir is None:
        plot_dir = default_results_dir()
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
    theory_label = "2PI inflated IC"
    if abs(omega_label) > 0.0 or abs(damping_label) > 0.0:
        suffix = r"\times\beta" if scaled_label else ""
        theory_label = fr"{theory_label} ($\omega={omega_label:g}{suffix}$, $\gamma={damping_label:g}{suffix}$)"

    if sigma_vals is None:
        sigma_vals = [0.75 * sigma_c, 0.95 * sigma_c, 1.1 * sigma_c]

    slow_factor = _beta_time_factor(beta) if scale_slow_beta_time else 1.0
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
                T=T * slow_factor,
                dt=dt,
                burn=burn * slow_factor,
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
    burn=300.0,
    sim_reps=2,
    plot_dir=None,
    theory_variants=None,
    scale_slow_beta_time=True,
):
    """Overlay simulation with several approximate phase-network theories."""
    import os

    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    sigma_c = _phase_sigma_c(I, alpha)
    if sigma is None:
        sigma = 1.3 * sigma_c

    if theory_variants is None:
        theory_variants = [
            dict(
                label="inflated IC",
                kwargs=dict(solver="inflated_ic", q_method="hermite", n_quad=48, hermite_order=32),
                style=dict(color="C2", ls="--", lw=2.2),
            ),
            dict(
                label=r"inflated IC + gen. kernel",
                kwargs=dict(
                    solver="inflated_ic", kernel_omega=2.0, kernel_damping=1.0,
                    q_method="hermite", n_quad=48, hermite_order=32,
                ),
                style=dict(color="C1", ls="-.", lw=2.0),
            ),
            dict(
                label=r"inflated IC + weak gen.",
                kwargs=dict(
                    solver="inflated_ic", kernel_omega=1.0, kernel_damping=0.5,
                    q_method="hermite", n_quad=48, hermite_order=32,
                ),
                style=dict(color="C4", ls=(0, (5, 2)), lw=2.0),
            ),
        ]

    C_runs = []
    tau_s = None
    slow_factor = _beta_time_factor(beta) if scale_slow_beta_time else 1.0
    for _ in range(max(1, int(sim_reps))):
        tau_run, C_run = sim_phase_network(
            N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
            T=T * slow_factor, dt=dt, burn=burn * slow_factor,
            tau_max=tau_max, n_probe=N,
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
    burn=300.0,
    sim_reps=1,
    plot_dir=None,
    theory_variants=None,
    scale_slow_beta_time=True,
):
    """Grid of phase-network examples comparing simulation to theory variants."""
    import os

    if plot_dir is None:
        plot_dir = default_results_dir()
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
                label="inflated IC",
                kwargs=dict(solver="inflated_ic", q_method="hermite", n_quad=48, hermite_order=32),
                style=dict(color="C2", ls="--", lw=1.8),
            ),
            dict(
                label=r"infl. IC + gen.",
                kwargs=dict(
                    solver="inflated_ic", kernel_omega=2.0, kernel_damping=1.0,
                    q_method="hermite", n_quad=48, hermite_order=32,
                ),
                style=dict(color="C1", ls="-.", lw=1.8),
            ),
            dict(
                label=r"infl. IC + weak gen.",
                kwargs=dict(
                    solver="inflated_ic", kernel_omega=1.0, kernel_damping=0.5,
                    q_method="hermite", n_quad=48, hermite_order=32,
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
        slow_factor = _beta_time_factor(beta) if scale_slow_beta_time else 1.0
        for _ in range(max(1, int(sim_reps))):
            tau_run, C_run = sim_phase_network(
                N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
                T=T * slow_factor, dt=dt, burn=burn * slow_factor,
                tau_max=tau_max, n_probe=N,
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
    burn=300.0,
    min_T=400.0,
    min_burn=400.0,
    sim_reps=1,
    plot_dir=None,
    theory_kwargs=None,
    scale_slow_beta_time=True,
):
    """Check phase theory and simulation on the scaled time axis beta*tau.

    Here tau_max is interpreted as the maximum beta*tau shown in every panel,
    so the raw simulated/theory lag window is tau_max/beta for each beta.
    """
    import os

    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    if theory_kwargs is None:
        theory_kwargs = dict(
            solver="inflated_ic",
            kernel_omega=2.0,
            kernel_damping=1.0,
            kernel_scaled_by_beta=True,
            q_method="hermite",
            n_quad=48,
            hermite_order=32,
        )
    else:
        theory_kwargs = dict(theory_kwargs)

    sigma_c = _phase_sigma_c(I, alpha)
    sigma = g_val * sigma_c

    beta_vals = tuple(float(b) for b in beta_vals)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    ax_raw, ax_scaled = axes
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(beta_vals)))
    max_raw_tau = tau_max / max(min(beta_vals), 1e-12)

    for beta, color in zip(beta_vals, colors):
        print(f"  beta scaling: beta={beta:g}, sigma={sigma:.3f}, g={g_val:.2f}")
        C_runs = []
        tau_s = None
        slow_factor = _beta_time_factor(beta) if scale_slow_beta_time else 1.0
        raw_tau_max = tau_max / max(float(beta), 1e-12)
        T_run = max(T * slow_factor, float(min_T))
        burn_run = max(burn * slow_factor, float(min_burn))
        for _ in range(max(1, int(sim_reps))):
            tau_run, C_run = sim_phase_network(
                N=N, I=I, alpha=alpha, sigma=sigma, beta=beta,
                T=T_run, dt=dt, burn=burn_run,
                tau_max=raw_tau_max, n_probe=N,
            )
            tau_s = tau_run
            C_runs.append(C_run)
        C_s = np.mean(C_runs, axis=0)
        C_s_norm = C_s / max(abs(C_s[0]), 1e-12)
        ax_raw.plot(tau_s, C_s_norm, color=color, lw=1.8,
                    label=fr"sim $\beta={beta:g}$")
        ax_scaled.plot(beta * tau_s, C_s_norm, color=color, lw=1.8,
                       label=fr"sim $\beta={beta:g}$")

        try:
            tau_th, C_th, _ = theory_phase_autocorr(
                I=I, alpha=alpha, sigma=sigma, beta=beta,
                tau_max=raw_tau_max, dtau=dtau, **theory_kwargs,
            )
            C_th_norm = C_th / max(abs(C_th[0]), 1e-12)
            ax_raw.plot(tau_th, C_th_norm, color=color, ls="--", lw=2.0,
                        label=fr"theory $\beta={beta:g}$")
            ax_scaled.plot(beta * tau_th, C_th_norm, color=color, ls="--", lw=2.0,
                           label=fr"theory $\beta={beta:g}$")
        except Exception as err:
            print(f"    theory failed for beta={beta:g}: {err}")

    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
        ax.set(ylim=(-0.35, 1.1))
        ax.legend(fontsize=8, ncol=1)

    ax_raw.set(
        xlabel=r"$\tau$",
        ylabel=r"$C_{uu}(\tau)/C_{uu}(0)$",
        title="Raw time",
        xlim=(0, max_raw_tau),
    )
    ax_scaled.set(
        xlabel=r"$\beta\tau$",
        title="Scaled time",
        xlim=(0, tau_max),
    )

    plt.suptitle(
        fr"Phase beta diagnostic: raw vs scaled time, $\alpha={alpha}$, $g={g_val}$",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_beta_scaling_diagnostic.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
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
    synapse_update="exact",
    max_recorded_spikes_per_bin=None,
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
    synapse_update = str(synapse_update).lower()
    if synapse_update not in ("exact", "euler"):
        raise ValueError("synapse_update must be 'exact' or 'euler'")

    def F(u_):
        return alpha * np.clip(I + u_, 0.0, 1e12) ** (1.0 / alpha)

    def spike_weights(phi_old, rate, spike_counts):
        if synapse_update == "euler":
            return spike_counts
        weighted = np.zeros_like(spike_counts, dtype=float)
        spiking = np.flatnonzero(spike_counts > 0)
        for j in spiking:
            count = int(spike_counts[j])
            if count <= 0 or rate[j] <= 0.0:
                continue
            thresholds = np.pi + 2.0 * np.pi * np.arange(count)
            crossing_times = (thresholds - phi_old[j]) / rate[j]
            crossing_times = np.clip(crossing_times, 0.0, dt)
            weighted[j] = np.sum(np.exp(-beta * (dt - crossing_times)))
        return weighted

    def update_u(u_, drive_):
        if synapse_update == "exact":
            return np.exp(-beta * dt) * u_ + beta * drive_
        return u_ + (-beta * u_) * dt + beta * drive_

    nb = int(burn / dt)
    for _ in range(nb):
        phi_old = phi.copy()
        rate = F(u)
        phi = phi_old + rate * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        drive = W @ spike_weights(phi_old, rate, spike_counts)
        u = update_u(u, drive)
        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)

    nt = int(T / dt)
    t = np.arange(nt) * dt
    idx = np.arange(n_show)
    U = np.zeros((nt, n_show))
    spk_times = [[] for _ in range(n_show)]

    for step in range(nt):
        phi_old = phi.copy()
        rate = F(u)
        phi = phi_old + rate * dt
        spike_counts = np.floor((phi + np.pi) / (2.0 * np.pi)).astype(float)
        spikes = spike_counts > 0
        phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
        drive = W @ spike_weights(phi_old, rate, spike_counts)
        u = update_u(u, drive)
        if not np.all(np.isfinite(u)):
            u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
        U[step] = u[idx]
        for k in range(n_show):
            count = int(spike_counts[idx[k]])
            if count:
                if max_recorded_spikes_per_bin is not None:
                    count = min(count, int(max_recorded_spikes_per_bin))
                spk_times[k].extend([t[step]] * count)

    return t, U, [np.array(st) for st in spk_times]


def _phase_sigma_c(I, alpha):
    rho = 1.0 / (2.0 * np.pi)
    Fprime0 = float(np.maximum(I, 1e-10) ** (1.0 / alpha - 1.0))
    return 1.0 / (rho * Fprime0)


def _beta_time_factor(beta):
    """Simulation-time multiplier for slow synapses.

    The u-filter relaxation time is O(1/beta).  Comparisons at beta < 1 need
    proportionally longer burn-in and recording windows to estimate the same
    number of effective correlation times.
    """
    return max(1.0, 1.0 / max(float(beta), 1e-12))


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
    synapse_update="exact",
    max_recorded_spikes_per_bin=1,
    plot_dir=None,
):
    """Plot u(t) traces for several neurons across sigma values.

    One column per sigma value; each panel shows n_show overlaid traces.
    """
    import os
    if plot_dir is None:
        plot_dir = default_results_dir()
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
            synapse_update=synapse_update,
            max_recorded_spikes_per_bin=max_recorded_spikes_per_bin,
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
    synapse_update="exact",
    max_recorded_spikes_per_bin=1,
    plot_dir=None,
):
    """Spike raster (neuron index vs time) + population rate for several sigma.

    Two rows per sigma: raster (top), smoothed population firing rate (bottom).
    """
    import os
    if plot_dir is None:
        plot_dir = default_results_dir()
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
            synapse_update=synapse_update,
            max_recorded_spikes_per_bin=max_recorded_spikes_per_bin,
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


def _worker_init(scripts_dir):
    """Initializer for spawned worker processes.

    Adds the scripts directory to sys.path so that rn_phase and friends are
    importable, and forces the non-interactive Agg matplotlib backend before
    any pyplot import can open a display.
    """
    import sys, os
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass


def _run_jobs_parallel(jobs, n_jobs):
    """Run a list of job-dicts via _phase_sim_job, returning results in order.

    n_jobs=1  → sequential (no extra processes).
    n_jobs>1  → ProcessPoolExecutor with that many workers.
    n_jobs=-1 → use os.cpu_count() workers.

    On macOS, 'fork' is unsafe when NumPy/BLAS has active threads and will
    cause worker processes to be killed (BrokenProcessPool).  We use 'spawn'
    on macOS instead; the _worker_init initializer re-adds the scripts path so
    _phase_sim_job is importable in the fresh child interpreter.  Linux keeps
    'fork' for speed.  Windows always uses 'spawn' (default).
    """
    import os, sys
    from concurrent.futures import ProcessPoolExecutor

    if n_jobs == 1:
        return [_phase_sim_job(dict(j)) for j in jobs]

    workers = os.cpu_count() if n_jobs == -1 else int(n_jobs)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    import multiprocessing
    if sys.platform == "darwin":
        mp_ctx = multiprocessing.get_context("spawn")
        init, init_args = _worker_init, (scripts_dir,)
    elif sys.platform != "win32":
        mp_ctx = multiprocessing.get_context("fork")
        init, init_args = None, ()
    else:
        mp_ctx = None  # Windows default (spawn)
        init, init_args = _worker_init, (scripts_dir,)

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_ctx,
            initializer=init,
            initargs=init_args,
        ) as ex:
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
    burn=300.0,
    sim_reps=2,
    plot_dir=None,
    n_jobs=1,
    theory_kwargs=None,
    scale_slow_beta_time=True,
):
    """Sim vs theory C_uu(tau) for different parameter combinations.

    param_sets : list of dicts with keys I, alpha, beta, sigma (absolute values),
                 and optional 'label'. Theory is shown only when sigma < sigma_c.
    """
    import os
    if plot_dir is None:
        plot_dir = default_results_dir()
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
        beta_ps = float(ps["beta"])
        slow_factor = max(1.0, 1.0 / max(beta_ps, 1e-12)) if scale_slow_beta_time else 1.0
        for _ in range(max(1, sim_reps)):
            seed = int(ss.spawn(1)[0].generate_state(1)[0])
            all_jobs.append(dict(
                N=N, I=ps["I"], alpha=ps["alpha"], sigma=ps["sigma"],
                beta=ps["beta"], T=T * slow_factor, burn=burn * slow_factor,
                dt=dt, tau_max=tau_max, n_probe=N,
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
    burn=300.0,
    plot_dir=None,
    n_jobs=1,
    theory_kwargs=None,
    scale_slow_beta_time=True,
):
    """Sim vs theory C_uu(tau) for different N (finite-size convergence).

    Three panels: subcritical / near-critical / chaotic regimes.
    Each panel overlays all N values; theory shown when available.
    """
    import os
    if plot_dir is None:
        plot_dir = default_results_dir()
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
    slow_factor = max(1.0, 1.0 / max(float(beta), 1e-12)) if scale_slow_beta_time else 1.0
    for sig in sigma_list:
        for N in N_vals:
            seed = int(ss.spawn(1)[0].generate_state(1)[0])
            all_jobs.append(dict(
                N=N, I=I, alpha=alpha, sigma=sig, beta=beta,
                T=T * slow_factor, burn=burn * slow_factor,
                dt=dt, tau_max=tau_max, n_probe=N,
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
