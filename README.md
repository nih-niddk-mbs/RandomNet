# RandomNet

Code used to reproduce the figures in the RandomNet paper. Simulation results,
figure files, and the paper source live outside this repository in OneDrive.

## Contents

- `scripts/rn_core.py`: row-sum-corrected random weights and correlation helpers.
- `scripts/rn_rate.py`: rate-network simulation and SCS closure.
- `scripts/rn_binary.py`: binary-network simulation and analytic closures.
- `scripts/rn_phase.py`: phase-reset simulation and cusp-based scalar 2PI closure.
- `scripts/make_paper_figures.py`: the single entry point for all paper figures.
- `tests/`: regression tests for row-sum correction and the same-spike cusp.

The phase closure follows the paper's decomposition of the spike covariance into
same-spike and distinct-spike pieces. The same-spike term fixes

```text
C11'(0+) = -beta^2 sigma^2 mean_rate / 2,
```

and `C11(0)` is solved from the associated cusp energy equation. The removed
inflated-initial-condition and phenomenological-kernel closures are not part of
the reproduction code.

## Environment

```bash
conda env create -f environment.yml
conda activate randomnet-py
```

On this workstation:

```bash
/Users/carsonc/miniconda3/bin/conda run -n randomnet-py python \
  scripts/make_paper_figures.py --profile quick
```

Use `--profile paper` for publication runs. By default outputs are written to:

```text
~/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/randomnet/paper/
```

Set `RANDOMNET_RESULTS_DIR` to change the external output root. The paper driver
also accepts `--plot-dir` for a one-off destination.

Run the regression tests with:

```bash
python -m pytest -q
```
