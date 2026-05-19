# RandomNet

Code to simulate and analyze randomly connected neural networks.

## Repository layout

- `julia/`: primary Julia source files (`RandomNet.jl` and related modules)
- `src/`: Julia package entrypoint shim for compatibility with Julia package loading
- `Project.toml`, `Manifest.toml`: Julia environment and dependencies
- `python/simulations/`: standalone Python simulation scripts for 2PI experiments

## Python simulations

The script `python/simulations/two_pi_random_network_tests.py` runs the 2PI test simulations and plotting workflows.

### Conda environment (recommended)

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate randomnet-py
```

Run:

```bash
python python/simulations/two_pi_random_network_tests.py
```

Install Python dependencies:

```bash
python -m pip install -r python/requirements.txt
```

Run:

```bash
python python/simulations/two_pi_random_network_tests.py
```
