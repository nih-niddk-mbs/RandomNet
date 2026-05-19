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
        "simmod", Path(__file__).with_name("two_pi_random_network_tests.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_module()
    out_dir = Path(__file__).resolve().parents[1] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Single binary sigma plot
    sigma = 0.8
    tau_th, Cnn_th, _, g = mod.theory_binary_autocorr(
        sigma=sigma, beta=1.0, mu=1.0, f0=0.5, f1=1.0, tau_max=20, dtau=0.001
    )
    tau_s, Cnn_s, _ = mod.sim_binary_network(
        N=300,
        sigma=sigma,
        beta=1.0,
        mu=1.0,
        f0=0.5,
        f1=1.0,
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
            sigma=sigma, beta=1.0, mu=1.0, f0=0.5, f1=1.0, tau_max=20, dtau=0.001
        )
        tau_s, Cnn_s, _ = mod.sim_binary_network(
            N=300,
            sigma=sigma,
            beta=1.0,
            mu=1.0,
            f0=0.5,
            f1=1.0,
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
    rate_params = dict(N=512, sigma=chaotic_sigma, T=1500.0, dt=0.05, burn=250, lam=1)
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

    # 4) Chaotic-rate sigma sweep with longer traces and a wider lag window.
    chaotic_sigmas = (1.8, 2.0, 2.4)
    fig, axes = plt.subplots(1, len(chaotic_sigmas), figsize=(5.2 * len(chaotic_sigmas), 4.5), sharey=True)

    for ax, chaotic_sigma in zip(np.atleast_1d(axes), chaotic_sigmas):
        rate_params = dict(N=512, sigma=chaotic_sigma, T=2200.0, dt=0.05, burn=300, lam=1)
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
        rate_params = dict(N=384, sigma=link_sigma, T=1600.0, dt=0.05, burn=250, lam=1, f=link_fn)
        rate_reps = 2
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
        ax.set_title(f"{link_name}\n(sigma={link_sigma})")
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

    print(str(p1))
    print(str(p2))
    print(str(p3))
    print(str(p4))
    print(str(p5))


if __name__ == "__main__":
    main()
