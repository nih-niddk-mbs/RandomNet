# RandomNet

Code used to reproduce the figures in the RandomNet paper. Simulation results,
figure files, and the paper source live in a configurable external results
directory rather than in this repository.

## Contents

- `scripts/rn_core.py`: row-sum-corrected random weights and correlation helpers.
- `scripts/rn_rate.py`: rate-network simulation and SCS closure.
- `scripts/rn_binary.py`: sigmoid binary-network simulation, dynamic DMFT,
  and the controlled affine-tangent benchmark.
- `scripts/rn_phase.py`: threshold-reset simulation, stationary and full
  two-time event-DMFT closures, general phase laws, transformed theta/QIF,
  LIF event sampling, and tangent chaos diagnostics.
- `scripts/rn_lif_fp.py`: density-level LIF--OU Fokker--Planck generator,
  threshold-flux propagator quadrature, and a one-mode closure diagnostic.
- `scripts/rn_phase_2pi.py`: the homogeneous Hartree/Wick phase approximation,
  its feedback-stability calculation, and the rejected fixed-kernel
  diagnostic retained for regression testing.
- `scripts/make_paper_figures.py`: the single entry point for all paper figures.
- `scripts/rn_chaos.py`: Lyapunov and replica scans plus transformed-theta and
  LIF activity and covariance comparisons.
- `scripts/make_chaos_figures.py`: independent driver for the separate
  `randnetchaos` manuscript.
- `scripts/benchmark_lif_fp.py`: prescribed-drive quadrature validation and
  Markov-mode closure benchmark.
- `tests/`: regression tests for row-sum correction, the same-spike cusp, and
  threshold-return peaks.

## Numerical Methods

The rate theory uses deterministic tensor Gauss-Hermite quadrature for the
Gaussian gain moment, bisection for the physical nonzero energy root, and
first-integral reconstruction of the monotone covariance branch. There is no
sampling noise in this calculation.

The binary DMFT solver samples only stationary Gaussian drive paths. For each
drive, it integrates the conditional two-state master equation and its
two-time propagator analytically on every constant-rate interval. The returned
state spectrum is the sum of the spectrum of the conditional means and the
exact conditional Bernoulli covariance. Binary jump histories are not sampled.
Common Gaussian paths are retained during fixed-point iteration, and the
diagnostics report the spectral residual and
`conditional_method="exact_master_equation"`.

The stationary phase event-DMFT solver uses common-random-number Fourier
synthesis, deterministic phase advection, complete threshold event counts,
and exact within-step filtering of every event time. The two-time solver instead
factorizes and iterates the full covariance matrix and reports a Frobenius
residual. The Hartree/Wick calculation discretizes the causal propagator
equations and uses a matrix-free Arnoldi calculation for its feedback
stability. Convergence should be checked separately in time step, path length,
path count, phase-grid size, and fixed-point tolerance.

The chaos calculations use a general one-dimensional law
`phi_dot = g(phi, u)`. Built-in models include the legacy power law, the
transformed theta neuron, and a leaky integrate-and-fire threshold/reset law;
a circular custom law can be passed as a callable. For `I > 0`, the theta
transformation is

```text
tan(theta/2) = sqrt(I) tan(phi/2)
phi_dot = 2 sqrt(I) + (1 + cos(phi)) u / sqrt(I).
```

The maximal Lyapunov calculation differentiates the theta and exact LIF maps
and every within-step event time analytically. The tangent two-replica calculation
power-iterates the Frechet derivative of the event-DMFT covariance map.
Neither calculation diagnoses chaos from covariance decay alone.

For a prescribed Ornstein--Uhlenbeck drive, the density-level effective-action
saddle is evaluated without drive-path sampling by `rn_lif_fp`. A conservative
finite-volume generator evolves the joint LIF voltage and drive density. Its
threshold probability flux is reinserted at reset, and propagating the
stationary event-flux measure gives the two-event covariance. Synaptic
filtering is the deterministic quadrature

```text
C_y(tau) = beta mean_rate exp(-beta |tau|) / 2
           + integral beta exp(-beta |tau-s|) Q_regular(s) ds / 2.
```

This is exact for a prescribed OU covariance up to state-grid and lag
quadrature error. Projecting the self-consistent ringing covariance onto one
OU exponential is retained only as a diagnostic: the projection converges in
its two parameters but discards most of the covariance norm. A general
path-free closure therefore requires a structure-preserving multi-mode
Gaussian generator, such as a Hermite/spectral discretization; the one-mode
diagnostic is not used as paper theory.

The same generator also supplies a first-return approximation through
`solve_lif_ou_first_return_renewal`. Reset reinjection is removed, the
stationary spike-conditioned drive distribution is propagated to its next
threshold crossing, and the resulting FPT density is inserted into the
renewal equation. This preserves the complete marginal interval law while
discarding serial dependence between successive intervals. Comparing it with
the reinjecting propagator therefore measures the renewal approximation itself,
not a mismatch between two single-neuron discretizations.

`compute_lif_ou_return_map` integrates the same absorbing propagator over all
return times. The resulting drive-state matrix is `P = -D L_abs^{-1} B`; its
subleading eigenvalue measures memory retained per spike, and
`-mean_interval/log(abs(lambda_2))` estimates the physical correlation time.

For prescribed threshold-flux kernel `Q`, the causal Gaussian 2PI equations
give `C11 = beta**2 * sigma**2 * R Q R.T`. The implemented phase calculation
closes this equation by synthesizing stationary Gaussian drives, advecting
the phase through each drive, measuring the complete event spectrum, and
filtering that spectrum back into the 2PI drive-covariance equation. This
retains both the same-spike contribution and the distinct threshold returns
visible in `C33`. The same-spike term fixes

```text
C11'(0+) = -beta^2 sigma^2 mean_rate / 2,
```

The stationary event-DMFT solver obeys this cusp directly. A separate smooth scalar
solver determines `C11(0)` from the associated cusp energy equation and is
shown only where the paper compares it with the return-resolving calculation.
No phenomenological return kernel is fitted. The trajectory average supplies
`Q[C11]` nonperturbatively, but it does not independently integrate every
mixed and density propagator in the four-field Gaussian 2PI system.

The next approximation removes stationarity during the closure. It iterates
the complete matrices `C11(t,s)` and `Q(t,s)`, measures `C33(v,t;v',s)` from
the same conditional phase paths, and takes lag averages only after the
two-time iteration. Use `solver="twotime_dmft"`. This preserves coherent
threshold returns that the stationary projection broadens too early.

The homogeneous Hartree/Wick approximation closes the bilocal spike-kernel
saddle with the Gaussian density propagator. It is the complete homogeneous
Gaussian/Wick truncation, but not an exact solution of the untruncated 2PI
effective action. Its feedback map preserves covariance positivity and
reports the instability of the homogeneous branch. Use
`solver="gaussian_2pi"` through `theory_phase_autocorr`. The old fixed-`Q`
event-diagonal splice remains available only as `solver="fixed_q"`; it is
excluded from paper figures because it can violate covariance positivity.

## Installation

Create the environment once:

```bash
conda env create -f environment.yml
conda activate randomnet-py
```

All computational modules are ordinary Python files in `scripts/`. From the
repository root, make them importable with:

```bash
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

No result path is built into the repository. Functions that write files accept
`plot_dir`; the aggregate workflow also accepts `--plot-dir` or reads
`RANDOMNET_RESULTS_DIR`.

## Run Individual Functions

Simulations and theories can be imported and called independently. They return
NumPy arrays and do not write files. For example:

```python
import numpy as np

from rn_rate import sim_rate_network, theory_rate_autocorr
from rn_binary import sim_binary_network, theory_binary_sigmoid_dmft
from rn_phase import (
    maximal_lyapunov_phase_network,
    sim_phase_network,
    theory_phase_autocorr,
)

# Rate model: simulation and SCS theory.
tau_sim, C_sim = sim_rate_network(
    N=256,
    sigma=1.5,
    T=200.0,
    burn=50.0,
    rng=np.random.default_rng(1),
)
tau_theory, C_theory = theory_rate_autocorr(sigma=1.5)

# Binary model: matched sigmoid simulation and dynamic DMFT.
tau, Cnn_sim, Cuu_sim = sim_binary_network(
    N=256,
    sigma=2.0,
    T=200.0,
    burn=50.0,
    rng=np.random.default_rng(2),
)
tau_theory, Cnn_theory, Cuu_theory, sigma_critical = (
    theory_binary_sigmoid_dmft(
        sigma=2.0,
        tau_max=20.0,
    )
)

# Phase model: simulation and the two event-DMFT closure levels.
tau, Cuu_sim = sim_phase_network(
    N=128,
    sigma=7.0,
    T=200.0,
    burn=50.0,
    rng=np.random.default_rng(3),
)
tau_twotime, Cuu_twotime, sigma_critical = theory_phase_autocorr(
    sigma=7.0,
    solver="twotime_dmft",
    tau_max=15.0,
)
tau_stationary, Cuu_stationary, _ = theory_phase_autocorr(
    sigma=7.0,
    solver="density",
    tau_max=15.0,
)

# Transformed theta/QIF model and a direct chaos diagnostic.
tau_theta, Cuu_theta = sim_phase_network(
    N=128,
    I=1.0,
    sigma=1.0,
    T=200.0,
    burn=50.0,
    phase_model="theta",
    rng=np.random.default_rng(4),
)
lambda_max = maximal_lyapunov_phase_network(
    N=128,
    I=1.0,
    sigma=1.0,
    phase_model="theta",
)
```

The plotting functions are independent as well. Give them an external output
directory explicitly:

```python
from pathlib import Path

from rn_rate import plot_rate_network
from rn_phase import plot_phase_density_correlation

output_dir = Path("/path/to/randomnet-results")

plot_rate_network(
    N=384,
    T=500.0,
    burn=100.0,
    sim_reps=1,
    plot_dir=output_dir,
)
plot_phase_density_correlation(
    plot_dir=output_dir,
)
```

The public functions available in each module can be inspected interactively:

```bash
PYTHONPATH=scripts python -c 'import rn_phase; help(rn_phase.theory_phase_autocorr)'
```

The principal entry points are:

- `rn_core`: `make_weights`, `autocorr`, and `default_results_dir`.
- `rn_rate`: `sim_rate_network`, `theory_rate_autocorr`, and
  `plot_rate_network`.
- `rn_binary`: `sigmoid_rate`, `sim_binary_network`,
  `theory_binary_sigmoid_dmft`, `theory_binary_sigmoid_tangent`, and the
  binary plotting functions. `theory_binary_autocorr` is the formal affine
  benchmark used by the tangent calculation.
- `rn_phase`: `sim_phase_network`, `theory_phase_autocorr`,
  `theory_phase_density_autocorr`, `theory_phase_twotime_dmft`,
  `maximal_lyapunov_phase_network`, `phase_replica_stability_dmft`, and the
  phase plotting and finite-size functions.
- `rn_chaos`: Lyapunov/replica scans and generic theta/LIF figure functions.
- `rn_lif_fp`: `solve_lif_ou_fokker_planck`,
  `solve_lif_ou_first_return_renewal`, `compute_lif_ou_return_map`,
  `validate_lif_ou_fokker_planck`, and `solve_projected_lif_ou_dmft`.
- `rn_phase_2pi`: Gaussian/Hartree, fixed-kernel, propagator, and stability
  solvers used by the phase comparisons.

## Reproduce Figures

Run the complete reduced-size smoke workflow from the repository root:

```bash
python scripts/make_paper_figures.py --profile quick
```

Quick output goes to the platform's system temporary directory in a
`randomnet-quick` folder, so it cannot overwrite publication figures or
caches. The driver prints the resolved path. Use `--plot-dir PATH` to keep a
quick run elsewhere.

Run the publication calculations with:

```bash
python scripts/make_paper_figures.py --profile paper
```

The separate chaos manuscript has its own output names and never overwrites
the main paper figures:

```bash
python scripts/make_chaos_figures.py \
  --profile quick \
  --output-dir /path/to/randnetchaos \
  --jobs 2

python scripts/make_chaos_figures.py \
  --profile paper \
  --output-dir /path/to/randnetchaos \
  --jobs 10

python scripts/benchmark_lif_fp.py \
  --profile paper \
  --output-dir /path/to/randnetchaos
```

By default, publication output is written directly to:

```text
~/randomnet-results/
```

Set `RANDOMNET_RESULTS_DIR` or pass `--plot-dir PATH` to choose another external
root:

```bash
export RANDOMNET_RESULTS_DIR=/path/to/randomnet-results
```

Generate only selected figure groups when a full run is unnecessary:

```bash
python scripts/make_paper_figures.py \
    --profile quick \
    --figures rate binary phase-density \
    --plot-dir /path/to/randomnet-results
```

Use `python scripts/make_paper_figures.py --help` for the complete group list.

## Optional External Driver

The repository can be driven by a Python file stored anywhere, including in
the external results directory. The external file owns machine-specific paths,
resource choices, and run selection; the repository remains portable.

Here is a complete external driver:

```python
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Configure these two paths for the machine running the calculations.
REPOSITORY = Path("/path/to/RandomNet")
RESULTS = Path(__file__).resolve().parent

sys.path.insert(0, str(REPOSITORY / "scripts"))

from make_paper_figures import (
    expand_figure_groups,
    make_figures,
)

parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=("quick", "paper"), default="quick")
parser.add_argument("--figures", nargs="+", default=["all"])
parser.add_argument("--jobs", type=int, default=4)
args = parser.parse_args()

make_figures(args.profile, expand_figure_groups(args.figures), RESULTS, args.jobs)
```

For example, run that file from any directory with:

```bash
python /path/to/randomnet-results/run_randomnet.py \
    --profile paper \
    --figures all \
    --jobs 10
```

The driver can instead import any individual function from `rn_rate`,
`rn_binary`, `rn_phase`, or `rn_phase_2pi` and combine calculations as needed.
Machine-specific repository and result paths belong in this external driver or
the environment, never in the repository.

Publication figures are numbered in manuscript order: rate calibration
(`fig01`), sigmoid binary results (`fig02`--`fig04`), the phase closure hierarchy
(`fig05`), two-time and stationary event-DMFT comparisons
(`fig06`--`fig10`), the smooth-feedback transition estimate and scalar-cusp
covariance (`fig11`), and phase activity (`fig12`--`fig13`).

The manuscript source, compiled PDF, numbered figures, reusable simulation
caches, and numerical tables all live directly in the external `randomnet`
directory. Temporary raw plots and working directories are removed after
their numbered figures are published. A quick run is a workflow check; only
`--profile paper` uses the network sizes, replicate counts, and resolutions
reported in the manuscript captions and finite-size table.

The binary microscopic model uses the bounded positive rate
`rate_max * expit((u - theta) / delta)` in both simulations and theory. The
dynamic DMFT solver evaluates the original conditional two-state master
equation and propagator exactly for each sampled Gaussian drive, then closes
its covariance self-consistently. It neither samples binary jumps nor inserts
an effective telegraph process. The affine pole formula is shown only as the
tangent limit of this same sigmoid at zero drive.

Regenerate the finite-size extrapolation used in the phase-density validity
table with:

```bash
python scripts/make_paper_figures.py --profile paper --figures phase-validity
```

Regenerate the Hartree/Wick comparison with:

```bash
python scripts/make_paper_figures.py --profile paper --figures phase-2pi
```

Run the regression tests with:

```bash
pytest -q
```
