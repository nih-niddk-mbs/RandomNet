# RandomNet

Code to simulate and analyze randomly connected neural networks.

## Repository Structure

```
RandomNet/
├── randomnet/              # Main Python package
├── scripts/                # Simulation and analysis scripts
│   ├── random_network.py   # Core simulation code
│   └── make_quick_plots.py
├── data/
│   ├── plots/             # All visualization outputs
│   └── results/           # Simulation results and cache
├── tests/                 # Test and debugging scripts
├── julia/                 # Julia implementation (reference)
├── environment.yml        # Conda environment specification
└── pyproject.toml         # Python package configuration
```

## Setup

### Create Conda Environment

```bash
conda env create -f environment.yml -p ./miniconda/envs/randomnet
conda activate randomnet
```

Alternatively, if using existing miniconda installation:

```bash
~/miniconda/bin/conda env create -f environment.yml
source ~/miniconda/etc/profile.d/conda.sh
conda activate randomnet
```

## Running Simulations

All simulation results and plots are saved to `data/plots/`:

```bash
cd scripts
python random_network.py
```

## File Organization

- **randomnet/**: Python package modules for network simulations
- **scripts/**: Executable simulation and plotting scripts
- **data/**: All numerical results and visualizations
- **tests/**: Development and debugging scripts  
- **julia/**: Reference Julia implementation (not actively maintained)
	`theory_phase_autocorr`, `plot_phase_network`).

### Conda environment (recommended)

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate randomnet-py
```

Run:

```bash
python python/simulations/random_network.py
```

Install Python dependencies:

```bash
python -m pip install -r python/requirements.txt
```

Run:

```bash
python python/simulations/random_network.py
```
