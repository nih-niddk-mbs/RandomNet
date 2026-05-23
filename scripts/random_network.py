"""
Compatibility facade for RandomNet simulations and theory closures.

The implementation is split across:
  - rn_core.py    shared helpers
  - rn_rate.py    rate-network simulation/theory/plots
  - rn_phase.py   phase/spiking simulation/theory/plots
  - rn_binary.py  binary-neuron simulation/theory/plots

Existing scripts can continue to import from random_network.py.
"""

from rn_core import *  # noqa: F401,F403
from rn_rate import *  # noqa: F401,F403
from rn_phase import *  # noqa: F401,F403
from rn_binary import *  # noqa: F401,F403


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, "..", "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving plots to: {plot_dir}")

    # 1. Rate network
    plot_rate_network(sigma=1.5, N=512, C0_guess=0.65, plot_dir=plot_dir)

    # 2. Binary network: shape of correlation functions
    plot_binary_network(sigma_vals=(0.5, 0.8, 0.95), N=800, plot_dir=plot_dir)

    # 3. Two-timescale exponential fit
    plot_two_timescale_fit(sigma=0.8, N=800, plot_dir=plot_dir)

    # 4. Finite-size convergence
    plot_binary_network_N_convergence(
        sigma=0.8,
        N_vals=(128, 300, 800, 1600),
        plot_dir=plot_dir,
    )

    # 5. Clipped-gain theory vs simulation
    plot_clipped_vs_linear(sigma_vals=(0.7, 1.0, 1.3), N=800, plot_dir=plot_dir)
