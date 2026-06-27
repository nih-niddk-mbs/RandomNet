# RandomNet Repository Reorganization

**Date**: May 21, 2026

**Update, June 27, 2026**: Generated figures and simulation caches now live
outside the repository in the configured RandomNet results directory, defaulting
to `~/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/randomnet/`.
The older `data/plots/` references below describe the historical
reorganization, not the current output policy.

## Summary of Changes

### Directory Structure
- **Before**: Scattered Python files, plots, and scripts throughout the repository
- **After**: Clean, organized Python project structure

### New Layout
```
RandomNet/
├── randomnet/              # Python package (empty, ready for modules)
├── scripts/                # Executable simulation scripts
│   ├── random_network.py   # Core simulation (moved from python/simulations/)
│   ├── make_quick_plots.py
│   └── simulations_old/    # Backup of old simulation directory
├── data/
│   ├── plots/              # ALL visualization outputs (28 PNG files)
│   └── results/            # Simulation results (ready for .npz, .npy files)
├── tests/                  # All test and debug scripts
├── julia/                  # Julia code (kept for reference)
├── environment.yml         # Conda environment specification
├── pyproject.toml         # Python package configuration
└── README.md              # Updated with new structure and setup instructions
```

### Files Moved
- **random_network.py**: `python/simulations/` → `scripts/`
- **make_quick_plots.py**: `python/simulations/` → `scripts/`
- **All PNG files** (28 files): `python/plots/` + root → `data/plots/`
- **Test scripts** (15 files): root directory → `tests/`
- **Old directories removed**: `python/`, `src/` (Julia compatibility shim)

### Configuration Updates
- **environment.yml**: Updated with proper conda environment spec
- **pyproject.toml**: Created for Python package configuration
- **.gitignore**: Enhanced with proper entries for Python projects
- **README.md**: Updated with new structure and conda setup instructions
- **random_network.py**: Updated all plot save paths to `data/plots/`

## Setup Instructions

### Using Conda
```bash
# Create environment from file
conda env create -f environment.yml

# Activate
conda activate randomnet

# Run simulations
cd scripts
python random_network.py
```

### All Plots Saved To
`/Users/carsonc/github/RandomNet/data/plots/`

## Key Improvements
1. ✅ Single location for all plots: `data/plots/`
2. ✅ Clean root directory (no scattered .py or .png files)
3. ✅ Proper Python package structure
4. ✅ Julia files organized and kept (not deleted)
5. ✅ Proper conda environment configuration
6. ✅ Results directory ready for simulation outputs
7. ✅ All paths relative and portable

## Status
✅ **Reorganization Complete** - Repository is now a clean, functional Python project
