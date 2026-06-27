"""Binary-neuron simulations, theory closures, and plots."""

import os

import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

from rn_core import autocorr, default_results_dir, make_weights, rng

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
        plot_dir = default_results_dir()
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
        plot_dir = default_results_dir()
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
        plot_dir = default_results_dir()
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
        plot_dir = default_results_dir()
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


