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
    solver="cusp",
    q_method="gh",
    n_qmc=2048,
    hermite_order=32,
    warn_on_no_branch=True,
):
    """
    Solve the scalar Gaussian 2PI closure for the phase network with lambda=1.

    With row-sum correction (lambda=1), W@1=0 so E[u_i]=0 exactly, meaning
    C_11(tau->inf)=0 and C_eq=0. The centered SCS with gain g is directly correct:
    no g_shifted or C_eq machinery needed.

    The same-spike term is imposed through the exact cusp condition

        C'(0+) = -beta**2 * sigma**2 * mean_rate / 2,

    and C(0) is fixed by the corresponding energy equation.  For positive lag,
    the smooth off-diagonal closure obeys

        C'' = beta**2 * (C - sigma**2 * Cov[g(u(0)), g(u(tau))]).

    q_method="gh" uses tensor Gauss-Hermite quadrature for Q_smooth. q_method="qmc"
    uses common-random Sobol Gaussian samples, which is slower but more robust
    for hard rectification and strongly nonlinear gains. q_method="hermite" uses
    a 1-D Hermite expansion of g(u) and evaluates the centered covariance as a
    series in C(tau)/C(0), avoiding cancellation from subtracting E[g]^2.
    The smooth Gaussian closure averages over phases and therefore does not
    retain the near-threshold C33 advection peaks described in the notes.  Those
    peaks require a richer phase-density closure than this scalar C_uu solver.
    warn_on_no_branch controls whether missing cusp branches are printed; scans
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

    beta_val = max(float(beta), 1e-10)
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
    if solver_key not in ("cusp", "same_spike", "same-spike"):
        raise ValueError("phase theory supports solver='cusp'")

    def solve_cusp_c0():
        if C0 is not None:
            guess_val = float(C0)
            if guess_val > 0.0:
                return guess_val

        def balance(C_total):
            C_total = float(C_total)
            if C_total <= 0.0:
                return np.nan
            cusp_energy = 0.25 * beta_val**2 * sigma**4 * mu_g(C_total) ** 2
            return smooth_energy(C_total, C_total) - cusp_energy

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
            if values[i] < 0.0 < values[i + 1]:
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
        if warn_on_no_branch:
            print("  no finite cusp-closure branch found")
        return np.nan

    def cusp_solution():
        C_total0 = solve_cusp_c0()
        if not np.isfinite(C_total0):
            return np.full_like(tau, np.nan, dtype=float)

        # Use the conserved energy to select the decaying stationary branch.
        # Direct integration of the second-order equation eventually excites its
        # exponentially growing companion after C has become numerically tiny.
        n_grid = 4096
        positive = np.unique(
            np.concatenate(
                (
                    np.geomspace(C_total0 * 1e-12, C_total0, n_grid),
                    np.linspace(C_total0 / n_grid, C_total0, n_grid),
                )
            )
        )
        C_grid = np.concatenate(([0.0], positive))
        Q_grid = np.array([Q_centered(c, C_total0) for c in C_grid])
        integral = np.zeros_like(C_grid)
        integral[1:] = np.cumsum(
            0.5 * (Q_grid[1:] + Q_grid[:-1]) * np.diff(C_grid)
        )
        energy = C_grid**2 - 2.0 * integral
        energy_scale = max(C_total0**2, 1.0)
        if np.min(energy) < -1e-7 * energy_scale:
            if warn_on_no_branch:
                print("  cusp energy has no real decaying branch")
            return np.full_like(tau, np.nan, dtype=float)
        speed = beta_val * np.sqrt(np.maximum(energy[1:], 1e-30))

        C_desc = positive[::-1]
        speed_desc = speed[::-1]
        dC = -np.diff(C_desc)
        segment_speed = 0.5 * (speed_desc[:-1] + speed_desc[1:])
        tau_desc = np.concatenate(
            ([0.0], np.cumsum(dC / np.maximum(segment_speed, 1e-30)))
        )
        return np.interp(tau, tau_desc, C_desc, left=C_total0, right=0.0)

    return tau, cusp_solution(), sigma_c


def plot_phase_operational_criticality(
    beta=1.0,
    alpha_vals=(1.0, 1.25, 1.5, 2.0, 3.0),
    I_vals=(0.25, 0.5, 1.0, 1.5, 2.0),
    theory_kwargs=None,
    plot_dir=None,
    **_unused,
):
    """Plot the smooth transition and cusp covariance at that transition.

    The cusp term generates a finite covariance on both sides of the smooth
    instability, so branch existence is not used as a transition criterion.
    """
    if plot_dir is None:
        plot_dir = default_results_dir()
    os.makedirs(plot_dir, exist_ok=True)

    kwargs = {} if theory_kwargs is None else dict(theory_kwargs)
    kwargs.setdefault("solver", "cusp")
    kwargs.setdefault("q_method", "hermite")
    kwargs.setdefault("n_quad", 48)
    kwargs.setdefault("hermite_order", 32)
    kwargs.setdefault("tau_max", 0.2)
    kwargs.setdefault("dtau", 0.1)
    kwargs.setdefault("warn_on_no_branch", False)

    alphas = np.asarray(alpha_vals, dtype=float)
    inputs = np.asarray(I_vals, dtype=float)
    sigma_smooth = np.array(
        [[_phase_sigma_c(input_value, alpha) for input_value in inputs] for alpha in alphas]
    )
    C0_at_transition = np.full_like(sigma_smooth, np.nan)

    for ia, alpha in enumerate(alphas):
        for ii, (input_value, sigma_c) in enumerate(zip(inputs, sigma_smooth[ia])):
            print(
                f"  cusp covariance at transition: I={input_value:g}, "
                f"alpha={alpha:g}, sigma_c={sigma_c:.4g}"
            )
            _tau, C, _ = theory_phase_autocorr(
                I=input_value,
                alpha=alpha,
                sigma=sigma_c,
                beta=beta,
                **kwargs,
            )
            if len(C) and np.isfinite(C[0]):
                C0_at_transition[ia, ii] = C[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(alphas)))
    for alpha, values, color in zip(alphas, sigma_smooth, colors):
        axes[0].plot(inputs, values, "o-", color=color, lw=1.8, label=fr"$\alpha={alpha:g}$")
    axes[0].set(
        xlabel=r"$I$",
        ylabel=r"predicted critical $\sigma_c$",
        title=r"Smooth-feedback transition: $\sigma_c=[\rho F'(0)]^{-1}$",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    for alpha, values, color in zip(alphas, C0_at_transition, colors):
        axes[1].plot(
            inputs,
            values,
            "o-",
            color=color,
            lw=1.8,
            label=fr"$\alpha={alpha:g}$",
        )
    axes[1].set(
        xlabel=r"$I$",
        ylabel=r"cusp closure $C_{11}(0)$ at $\sigma_c$",
        title=fr"Same-spike covariance at transition ($\beta={beta:g}$)",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    outpath = os.path.join(plot_dir, "phase_operational_criticality.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved to {outpath}")
    plt.close("all")
    return dict(
        alpha=alphas,
        I=inputs,
        sigma_smooth=sigma_smooth,
        beta=float(beta),
        C0_at_transition=C0_at_transition,
    )


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
                label="cusp closure",
                kwargs=dict(solver="cusp", q_method="hermite", n_quad=48, hermite_order=32),
                style=dict(color="C3", ls="--", lw=2.2),
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
                label="cusp closure",
                kwargs=dict(solver="cusp", q_method="hermite", n_quad=48, hermite_order=32),
                style=dict(color="C3", ls="--", lw=1.8),
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

    plt.suptitle("Phase network: cusp closure across parameters", fontsize=13, fontweight="bold")
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
            solver="cusp",
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
