# Trapped-Ion RL Control

This project focuses on **model-free reinforcement learning for trapped-ion quantum control**. The RL agent (PPO/TF-Agents) learns red/blue sideband pulse sequences to prepare **bosonic cat states** of the motional mode. Rewards are computed from sampled **Wigner-function parity measurements**, while **fidelity** is tracked for evaluation and final reporting.

The implementation follows a measurement-based reward design, compares reward against fidelity during evaluation, and visualizes the final phase-space distributions and pulse sequences.

## Requirements
The project was tested with the conda environment defined in `qcrl-server-tf240.yml` (TensorFlow 2.4.0 and TF-Agents 0.6.0). CPU-only setups work, but training may be slower.

## Installation
Create and activate the conda environment:
```sh
conda env create -f qcrl-server-tf240.yml
conda activate qcrl-server  # or your env name if you created a custom one
```
Install the package in editable mode:
```sh
pip install -e .
```

## Run the trapped-ion cat-state example
Open two terminals (local or VSCode remote), and run both from the same repo checkout (don’t mix local and remote).

In both terminals:
```sh
cd /path/to/quantum_control_rl_server
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qcrl-server  # or your env name if you created a custom one
# If you're on a headless/remote machine, this avoids Matplotlib config issues:
export MPLCONFIGDIR=/tmp/matplotlib
```

Terminal A (server):
```sh
cd examples/trapped_ion_cat
python trapped_ion_cat_training_server.py
```

Terminal B (client):
```sh
cd examples/trapped_ion_cat
python trapped_ion_cat_client.py
```

After training finishes, generate plots from the latest `.h5` file:
```sh
cd examples/trapped_ion_cat
python parse_trapped_ion_cat_data.py
```

Optional (plot final pulse sequences):
```sh
cd examples/trapped_ion_cat
python plot_trapped_ion_cat_pulses.py
```

## Outputs
Plots and evaluation artifacts are saved under `examples/trapped_ion_cat/outputs/` (for example: `training_curve.png`, `eval_fidelity_curve.png`, `pulse_sequences.png`, `wigner_target_vs_final.png`, and `final_fidelity.txt`). These files are tracked in this repository.

## Code map (trapped-ion focus)

Core RL server (`quantum_control_rl_server/`):
- `PPO.py`: PPO training loop (TF-Agents) used by the training server.
- `dynamic_episode_driver_sim_env.py`: builds a simulated TF-Agents driver and wraps the environment for batch collection.
- `tf_env.py`: base TF-Agents environment; communicates actions and rewards via remote client.
- `tf_env_wrappers.py`: action wrapper (scaling, scripted actions, optional residual learning).
- `remote_env_tools.py`: TCP/IP pickle socket utilities for server/client communication.
- `h5log.py`: writes actions, rewards, and policy distributions into `.h5` logs.
- `version_helper.py`: TF-Agents version compatibility shim.
- `__init__.py`: package marker.

Trapped-ion example (`examples/trapped_ion_cat/`):
- `trapped_ion_cat_training_server.py`: defines the RL task (actions, scales, PPO settings) and launches the server.
- `trapped_ion_cat_client.py`: runs the simulation, computes measurement-based rewards, logs fidelity, and writes final plots.
- `trapped_ion_cat_sim_function.py`: QuTiP simulation of red/blue sideband dynamics, cat-state targets, Wigner sampling, and reward computation.
- `parse_trapped_ion_cat_data.py`: parses `.h5` logs and `eval_fidelity.csv`, then generates training/eval plots.
- `plot_trapped_ion_cat_pulses.py`: reads final policy parameters from `.h5` and plots pulse sequences.

Data/artifacts:
- `examples/trapped_ion_cat/*.h5`: training logs written by the server.
- `examples/trapped_ion_cat/eval_fidelity.csv`: evaluation fidelity history written by the client.
- `examples/trapped_ion_cat/outputs/`: stored plots and final-state artifacts.

Repo support files:
- `setup.py`: package metadata for editable installs.
- `requirements.txt`: Python package list (if you prefer pip over conda).
- `qcrl-server-tf240.yml`: conda environment specification used in this project.


## License
See `LICENSE`.
