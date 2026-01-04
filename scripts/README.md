# Scripts Directory

This directory contains execution scripts for running simulations and analyses.

## Execution Scripts

- `run.py` - Run batch trajectory simulations
- `run_set.py` - Run parameter set simulations
- `run_giulio.py` - Reference implementation runner

## Multiprocessing Scripts

- `mp_run.py` - Multiprocessing runner for damping estimation
- `mp_run_force.py` - Multiprocessing runner for force detection
- `mp_run_freq.py` - Multiprocessing runner for frequency estimation
- `mp_run_freq0.py` - Alternative frequency estimation runner

## Utility Scripts

- `inspecting_force.py` - Force detection inspection utilities
- `script_to_move_data.py` - Data management utilities

## Usage

Most scripts should be run from the repository root directory:

```bash
# From repository root
python scripts/run.py --seed 10 --mode damping --dt 1e-4 --total_time 50
```

Some scripts may require the `numerics` package to be importable, which is available when running from the root directory.

