# Trapped-Ion GKP Example

This directory is a GKP-target variant of the trapped-ion characteristic-function RL pipeline.

## Main scripts

- `trapped_ion_gkp_training_server.py`: PPO training server.
- `trapped_ion_gkp_client.py`: remote simulation client and final refinement.
- `trapped_ion_gkp_sim_function.py`: trapped-ion simulator and GKP target/characteristic utilities.
- `run_with_logs.sh`: one-command local launcher (server + client + plots).
- `parse_trapped_ion_gkp_data.py`: training/eval curve plotting.
- `plot_trapped_ion_gkp_pulses.py`: pulse sequence plotting.
- `make_characteristic_points_gif.py`: GIF of characteristic sampling points over epochs.

## Key GKP environment variables

- `GKP_DELTA` (default `0.301`)
- `GKP_LOGICAL` (default `0`; accepted: `0`, `1`, `plus`, `minus`)
- `GKP_SQUEEZE_R` (optional; default auto `-log(GKP_DELTA)`)
- `GKP_ENVELOPE_KAPPA` (optional; default auto `GKP_DELTA`)
- `GKP_LATTICE_TRUNC` (default `4`)

By default the GKP target follows the finite-energy mapping used in the papers:
- envelope parameter `delta`
- peak squeezing `r = -log(delta)`
- envelope operator scale `kappa = delta`

## Quick run

Run inside an activated environment on a compute node:

```bash
cd examples/trapped_ion_gkp
bash run_with_logs.sh
```

If your server/client dependencies are split across different environments,
launch with separate interpreters:

```bash
SERVER_PYTHON=/path/to/venv_tf/bin/python \
CLIENT_PYTHON=/path/to/venv_dq/bin/python \
POST_PYTHON=/path/to/venv_dq/bin/python \
bash run_with_logs.sh
```
