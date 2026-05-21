import importlib.util
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf


def load_module():
    spec = importlib.util.spec_from_file_location(
        "simmod", Path(__file__).with_name("random_network.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_module()
    out_dir = Path(__file__).resolve().parents[1] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a larger default rate-network size to reduce finite-size bias.
    rate_N = 1024

    # Shared binary-network parameters. gamma, n_bar, c1 are determined by these,
    # so expose them once here rather than implicitly fixing them in each call.
    binary_params = dict(beta=1.0, mu=1.0, f0=0.5, f1=1.0)

    # 1) Single binary sigma plot
    sigma = 0.8
    tau_th, Cnn_th, _, g = mod.theory_binary_autocorr(
        sigma=sigma, tau_max=20, dtau=0.001, **binary_params
    )
    tau_s, Cnn_s, _ = mod.sim_binary_network(
        N=300,
        sigma=sigma,
        **binary_params,
        T=800.0,
        dt=0.02,
        lam=1,
        burn=200,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tau_s, Cnn_s / max(Cnn_s[0], 1e-12), lw=1.8, label="Simulation")
    ax.plot(
        tau_th,
        Cnn_th / max(Cnn_th[0], 1e-12),
        "--",
        lw=2.2,
        label=f"Theory (g={g:.2f})",
    )
    ax.set_xlabel("tau")
    ax.set_ylabel("C_nn(tau) / C_nn(0)")
    ax.set_title("Binary network: sigma=0.8")
    ax.set_xlim(0, 12)
    ax.legend()
    fig.tight_layout()
    p1 = out_dir / "quick_binary_sigma_08.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # 2) Multi-sigma binary comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    for sigma in (0.5, 0.8, 0.95):
        tau_th, Cnn_th, _, g = mod.theory_binary_autocorr(
            sigma=sigma, tau_max=20, dtau=0.001, **binary_params
        )
        tau_s, Cnn_s, _ = mod.sim_binary_network(
            N=300,
            sigma=sigma,
            **binary_params,
            T=800.0,
            dt=0.02,
            lam=1,
            burn=200,
        )

        ax.plot(
            tau_s,
            Cnn_s / max(Cnn_s[0], 1e-12),
            lw=1.5,
            label=f"Sim sigma={sigma:.2f}",
        )
        ax.plot(
            tau_th,
            Cnn_th / max(Cnn_th[0], 1e-12),
            "--",
            lw=2.0,
            label=f"Theory sigma={sigma:.2f}",
        )

    ax.set_xlabel("tau")
    ax.set_ylabel("C_nn(tau) / C_nn(0)")
    ax.set_title("Binary network: simulation vs theory")
    ax.set_xlim(0, 12)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p2 = out_dir / "quick_binary_multi_sigma.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # 3) Chaotic-rate comparison with fixed parameters.
    # Average a couple of simulation runs to reduce tail noise, while keeping the
    # theory branch fixed and reproducible.
    chaotic_sigma = 2.0
    rate_params = dict(N=rate_N, sigma=chaotic_sigma, T=1500.0, dt=0.05, burn=250, lam=1)
    rate_reps = 2
    tau_s = None
    C_runs = []
    for _ in range(rate_reps):
        tau_run, C_run = mod.sim_rate_network(**rate_params)
        tau_s = tau_run
        C_runs.append(C_run / max(C_run[0], 1e-12))
    C_s = np.mean(C_runs, axis=0)
    C_s_std = np.std(C_runs, axis=0)
    tau_t, C_t = mod.theory_rate_autocorr(C0=None, sigma=chaotic_sigma, tau_max=20, dtau=0.01)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tau_s, C_s, lw=1.8, label=f"Simulation mean ({rate_reps} runs)")
    ax.fill_between(
        tau_s,
        np.clip(C_s - C_s_std, -1.0, 1.0),
        np.clip(C_s + C_s_std, -1.0, 1.0),
        color="C0",
        alpha=0.15,
        linewidth=0,
    )
    ax.plot(tau_t, C_t / max(C_t[0], 1e-12), "--", lw=2.2, label="Theory")
    ax.set_xlabel("tau")
    ax.set_ylabel("C(tau) / C(0)")
    ax.set_title(f"Chaotic rate network: sigma={chaotic_sigma}")
    ax.set_xlim(0, 12)
    ax.legend()
    fig.tight_layout()
    p3 = out_dir / "quick_rate_sigma_20.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    # 3b) Phase-neuron model comparison (kept separate from the rate model).
    phase_I = 1.0
    phase_alpha = 1.0
    phase_beta = 1.0
    _, _, phase_sigma_c = mod.theory_phase_autocorr(
        I=phase_I,
        alpha=phase_alpha,
        sigma=1.0,
        beta=phase_beta,
        tau_max=5,
        dtau=0.02,
    )
    phase_sigma = 0.6 * phase_sigma_c
    tau_ps, C_ps = mod.sim_phase_network(
        N=512,
        I=phase_I,
        alpha=phase_alpha,
        sigma=phase_sigma,
        beta=phase_beta,
        T=1800.0,
        dt=0.02,
        burn=250,
    )
    tau_pt, C_pt, _ = mod.theory_phase_autocorr(
        I=phase_I,
        alpha=phase_alpha,
        sigma=phase_sigma,
        beta=phase_beta,
        tau_max=30,
        dtau=0.02,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tau_ps, C_ps / max(C_ps[0], 1e-12), lw=1.8, label="Simulation")
    ax.plot(tau_pt, C_pt / max(C_pt[0], 1e-12), "--", lw=2.2, label="Theory")
    ax.set_xlabel("tau")
    ax.set_ylabel("C_uu(tau) / C_uu(0)")
    ax.set_title(
        f"Phase neuron network: sigma={phase_sigma:.2f} (sigma_c={phase_sigma_c:.2f})"
    )
    ax.set_xlim(0, 20)
    ax.legend()
    fig.tight_layout()
    p3b = out_dir / "quick_phase_sigma_subcritical.png"
    fig.savefig(p3b, dpi=160)
    plt.close(fig)

    # 3c) Phase-neuron spike-correlation comparison from phi-rate closure.
    tau_pspk_s, _Cuu_pspk, Cspk_s = mod.sim_phase_network(
        N=512,
        I=phase_I,
        alpha=phase_alpha,
        sigma=phase_sigma,
        beta=phase_beta,
        T=1800.0,
        dt=0.02,
        burn=250,
        return_spike=True,
    )
    tau_pspk_t, Cspk_t, _ = mod.theory_phase_spike_autocorr(
        I=phase_I,
        alpha=phase_alpha,
        sigma=phase_sigma,
        beta=phase_beta,
        tau_max=30,
        dtau=0.02,
    )

    # Drop zero lag to avoid finite-dt spike-count peak dominating normalization.
    tau_pspk_s = tau_pspk_s[1:]
    Cspk_s = Cspk_s[1:]
    tau_pspk_t = tau_pspk_t[1:]
    Cspk_t = Cspk_t[1:]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tau_pspk_s, Cspk_s / max(Cspk_s[0], 1e-12), lw=1.8, label="Simulation")
    ax.plot(
        tau_pspk_t,
        Cspk_t / max(Cspk_t[0], 1e-12),
        "--",
        lw=2.2,
        label="Theory (phi-rate)",
    )
    ax.set_xlabel("tau")
    ax.set_ylabel("C_spk(tau) / C_spk(0+)")
    ax.set_title(
        f"Phase spike correlation: sigma={phase_sigma:.2f} (sigma_c={phase_sigma_c:.2f})"
    )
    ax.set_xlim(0, 20)
    ax.legend()
    fig.tight_layout()
    p3c = out_dir / "quick_phase_spike_sigma_subcritical.png"
    fig.savefig(p3c, dpi=160)
    plt.close(fig)

    # 4) Chaotic-rate sigma sweep with longer traces and a wider lag window.
    chaotic_sigmas = (1.8, 2.0, 2.4)
    fig, axes = plt.subplots(1, len(chaotic_sigmas), figsize=(5.2 * len(chaotic_sigmas), 4.5), sharey=True)

    for ax, chaotic_sigma in zip(np.atleast_1d(axes), chaotic_sigmas):
        rate_params = dict(N=rate_N, sigma=chaotic_sigma, T=2200.0, dt=0.05, burn=300, lam=1)
        rate_reps = 2
        tau_s = None
        C_runs = []
        for _ in range(rate_reps):
            tau_run, C_run = mod.sim_rate_network(**rate_params)
            tau_s = tau_run
            C_runs.append(C_run / max(C_run[0], 1e-12))
        C_s = np.mean(C_runs, axis=0)
        C_s_std = np.std(C_runs, axis=0)
        tau_t, C_t = mod.theory_rate_autocorr(C0=None, sigma=chaotic_sigma, tau_max=30, dtau=0.01)

        ax.plot(tau_s, C_s, lw=1.8, label=f"Sim mean ({rate_reps} runs)")
        ax.fill_between(
            tau_s,
            np.clip(C_s - C_s_std, -1.0, 1.0),
            np.clip(C_s + C_s_std, -1.0, 1.0),
            color="C0",
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(tau_t, C_t / max(C_t[0], 1e-12), "--", lw=2.0, label="Theory")
        ax.set_title(f"Chaotic rate: sigma={chaotic_sigma}")
        ax.set_xlabel("tau")
        ax.set_xlim(0, 25)
        ax.grid(alpha=0.15)
        if ax is axes[0]:
            ax.set_ylabel("C(tau) / C(0)")
        ax.legend(fontsize=8)

    fig.suptitle("Chaotic rate network sigma sweep", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p4 = out_dir / "quick_rate_sigma_sweep.png"
    fig.savefig(p4, dpi=160)
    plt.close(fig)

    # 5) Multiple link-function examples under the same network setup.
    link_examples = [
        ("tanh", np.tanh),
        ("erf(x/sqrt(2))", lambda x: erf(x / np.sqrt(2.0))),
        ("softsign", lambda x: x / (1.0 + np.abs(x))),
    ]
    link_sigma = 1.8
    fig, axes = plt.subplots(1, len(link_examples), figsize=(5.2 * len(link_examples), 4.5), sharey=True)

    for ax, (link_name, link_fn) in zip(np.atleast_1d(axes), link_examples):
        rate_params = dict(
            N=rate_N,
            sigma=link_sigma,
            T=3200.0,
            dt=0.05,
            burn=500,
            lam=1,
            f=link_fn,
            n_probe=256,
        )
        rate_reps = 4
        C_runs = []
        tau_s = None
        for _ in range(rate_reps):
            tau_run, C_run = mod.sim_rate_network(**rate_params)
            tau_s = tau_run
            C_runs.append(C_run / max(C_run[0], 1e-12))
        C_s = np.mean(C_runs, axis=0)
        C_s_std = np.std(C_runs, axis=0)
        tau_t, C_t = mod.theory_rate_autocorr(
            C0=None,
            sigma=link_sigma,
            tau_max=25,
            dtau=0.01,
            f=link_fn,
        )

        ax.plot(tau_s, C_s, lw=1.8, label=f"Sim mean ({rate_reps} runs)")
        ax.fill_between(
            tau_s,
            np.clip(C_s - C_s_std, -1.0, 1.0),
            np.clip(C_s + C_s_std, -1.0, 1.0),
            color="C0",
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(tau_t, C_t / max(C_t[0], 1e-12), "--", lw=2.0, label="Theory")
        ax.set_title(f"{link_name}\n(sigma={link_sigma}, N={rate_params['N']}, reps={rate_reps})")
        ax.set_xlabel("tau")
        ax.set_xlim(0, 20)
        ax.grid(alpha=0.15)
        if ax is axes[0]:
            ax.set_ylabel("C(tau) / C(0)")
        ax.legend(fontsize=8)

    fig.suptitle("Rate network: link-function comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p5 = out_dir / "quick_rate_link_functions.png"
    fig.savefig(p5, dpi=160)
    plt.close(fig)

    # 6) Network-size invariance check (rate network).
    # If the scaling is correct, normalized correlations should collapse as N grows.
    N_values = (512, 1024, 2048)
    sigma_inv = 1.8
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for N in N_values:
        tau_s, C_s = mod.sim_rate_network(
            N=N,
            sigma=sigma_inv,
            T=2200.0,
            dt=0.05,
            burn=300,
            lam=1,
            f=np.tanh,
            n_probe=min(N, 256),
        )
        ax.plot(tau_s, C_s / max(C_s[0], 1e-12), lw=1.8, label=f"N={N}")

    ax.set_xlabel("tau")
    ax.set_ylabel("C(tau) / C(0)")
    ax.set_title(f"Rate network size invariance check (sigma={sigma_inv})")
    ax.set_xlim(0, 20)
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    p6 = out_dir / "rate_size_invariance.png"
    fig.savefig(p6, dpi=160)
    plt.close(fig)

    print(str(p1))
    print(str(p2))
    print(str(p3))
    print(str(p4))
    print(str(p5))
    print(str(p6))
    print(str(p3b))
    print(str(p3c))


if __name__ == "__main__":
    main()
