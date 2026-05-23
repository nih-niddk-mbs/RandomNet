# RandomNet

Simulation and theory experiments for randomly connected neural networks,
including rate, phase/spiking, and binary-neuron models.

The current focus is comparing simulations with several 2PI/Gaussian-closure
approximations.  Theory details and the meaning of the different criticality
measures are summarized in [docs/theory_closures.md](docs/theory_closures.md).

## Repository Layout

```text
RandomNet/
├── scripts/
│   ├── random_network.py      # Compatibility facade and legacy __main__
│   ├── rn_core.py             # Shared RNG, weights, autocorrelation helpers
│   ├── rn_rate.py             # Rate-network simulation/theory/plots
│   ├── rn_phase.py            # Phase-network simulation/theory/plots
│   ├── rn_binary.py           # Binary-neuron simulation/theory/plots
│   └── make_quick_plots.py    # Small binary/rate plotting script
├── data/
│   └── plots/                 # Generated figures
├── docs/
│   └── theory_closures.md     # Notes on closures and criticality
├── tests/                     # Development diagnostics
├── julia/                     # Reference/older Julia implementation
├── environment.yml
└── pyproject.toml
```

## Environment

Recommended Conda environment:

```bash
conda env create -f environment.yml
conda activate randomnet-py
```

On this workstation, the environment is commonly run via:

```bash
/Users/carsonc/miniconda3/bin/conda run -n randomnet-py python ...
```

## Common Commands

Compile-check the main scripts:

```bash
python -m py_compile scripts/random_network.py run_phase_plots.py
```

Generate the phase comparison plots:

```bash
python run_phase_plots.py
```

Generate quick binary/rate plots:

```bash
python scripts/make_quick_plots.py
```

Run the full legacy plotting script:

```bash
python scripts/random_network.py
```

All plots should be written under `data/plots/`.

## Main Theory Entrypoints

- `rn_rate.theory_rate_autocorr`: rate-network SCS theory.
- `rn_phase.theory_phase_autocorr`: phase-network closures, including finite-difference,
  inflated-initial-condition, and Hermite covariance options.
- `rn_phase.phase_operational_criticality`: branch-existence criticality for a selected
  phase closure.
- `rn_binary.theory_binary_autocorr`: exact linear binary-neuron theory.
- `rn_binary.theory_binary_clipped_integral`: clipped binary-neuron integral closure.

See [docs/theory_closures.md](docs/theory_closures.md) before interpreting
plots, because several approximations are intentionally kept on the table.

For backward compatibility, these names are also re-exported by
`scripts/random_network.py`.
