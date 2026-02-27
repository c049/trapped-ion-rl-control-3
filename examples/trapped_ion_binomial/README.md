# Trapped-Ion Binomial Example

This directory is the binomial-target variant of the trapped-ion characteristic-function RL pipeline.

## Main scripts

- `trapped_ion_binomial_training_server.py`: PPO training server.
- `trapped_ion_binomial_client.py`: remote simulation client and final refinement.
- `trapped_ion_binomial_sim_function.py`: trapped-ion simulator and binomial target/characteristic utilities.
- `run_with_logs.sh`: one-command local launcher (server + client + plots).
- `parse_trapped_ion_binomial_data.py`: training/eval curve plotting.
- `plot_trapped_ion_binomial_pulses.py`: pulse sequence plotting.
- `make_characteristic_points_gif.py`: GIF of characteristic sampling points over epochs.

## Binomial target options

- `BINOMIAL_CODE` (default `d3_z`)
  - `d3_z`: `(sqrt(3)|3> + |9>) / 2` (PRX 2022 appendix example)
  - `s2_plus`: `(|0> + sqrt(3)|6>) / 2`
  - `s1_plus`: `(|0> + |4>) / sqrt(2)`
  - backward-compatible aliases include `d3_minus`, `d3_plus`, `s2_z`
- `BINOMIAL_REL_PHASE` (optional relative phase on the second component)
- `N_BOSON` (default `30`)
- `CHAR_IMPORTANCE_POWER` (default `1.0`; sampling density proportional to `|chi_target|^power`)
- `CHAR_REWARD_OBJECTIVE_STAGE2` and `CHAR_REWARD_SWITCH_EPOCH` (optional objective schedule)
- `CHAR_REWARD_SWITCH_MIN_BEST_EVAL`, `CHAR_REWARD_STAGE2_PATIENCE_EVAL`,
  `CHAR_REWARD_STAGE2_MIN_GAIN`, `CHAR_REWARD_STAGE2_ALLOW_REVERT`
  (guard/rollback controls for objective switching)
- `CHAR_REWARD_AUTO_RESCALE=1` with `CHAR_REWARD_AUTO_RESCALE_TARGET_P90`
  (stabilize reward scale when overlap-style rewards become too large)
- `BINOMIAL_TARGET_TAIL_WARN` / `BINOMIAL_TARGET_TAIL_ERROR` (Fock-tail diagnostics)
- `ALLOW_LOW_N_BOSON=1` (override truncation guard intentionally)
- `FINAL_REFINE_ENABLE_AMP=1` (segment-level amplitude search in final local refinement)
- `FINAL_REFINE_FULL_STEPS=1` (optional step-level final refinement)

## Physical scale options

- `OMEGA_RABI_HZ` (default `2000`): sets `Omega_r = Omega_b = 2*pi*OMEGA_RABI_HZ`.
- `T_STEP` (default `1e-5` seconds): integration step per control time step.
- Keep `Omega` and `T_STEP` consistent:
  if `Omega` is scaled by factor `k`, scale `T_STEP` by `1/k` to preserve pulse-time scale.
- Optional time-via-amplitude regularization:
  `LEARN_AMP_R=1 LEARN_AMP_B=1` enables amplitude optimization in PPO policy;
  `AMP_OVERSHOOT_LAMBDA>0` penalizes amplitudes above `AMP_OVERSHOOT_BASE` (default `1.0`).
- Global duration-scale control (RL action, enabled by default):
  `LEARN_DURATION_SCALE=1` keeps a scalar action that rescales the full Hamiltonian in the rollout
  (`LEARN_DURATION_SCALE=0` disables it).
  Use `DURATION_INIT_SCALE`, `DURATION_ACTION_SCALE`, `DURATION_MIN_SCALE`, `DURATION_MAX_SCALE`.
  `DURATION_SCALE_PENALTY_LAMBDA>0` adds explicit time regularization on `duration_scale`
  (target controlled by `DURATION_SCALE_TARGET`; default penalizes only above target).

## Dephasing-robust options

- `ROBUST_TRAINING=1`: enable dephasing-robust training objective.
- `DEPHASE_MODEL` (default `quasi_static`): robust-training noise model (`quasi_static` or `stochastic`).
- `DEPHASE_DETUNING_FRAC` (default `0.05`): detuning scale as a fraction of Rabi rate.
  For `quasi_static`, samples are drawn from `delta in [-frac*Omega, +frac*Omega]`.
- `DEPHASE_NOISE_SAMPLES_TRAIN` / `DEPHASE_NOISE_SAMPLES_EVAL` / `DEPHASE_NOISE_SAMPLES_REFINE`:
  number of sampled dephasing trajectories for training, evaluation, and final refinement.
  For `DEPHASE_MODEL=quasi_static` with `DEPHASE_QUASI_SAMPLER=grid` and `DEPHASE_INCLUDE_NOMINAL=1`,
  odd counts are recommended so non-nominal samples form symmetric `+/-` pairs.
  `AUTO_ADJUST_QUASI_GRID_ODD_SAMPLES=1` auto-adjusts even counts by `+1`.
- `DEPHASE_INCLUDE_NOMINAL=1`: include a nominal sample (`delta=0`) in each sampled set.
- `DEPHASE_OBJECTIVE_INCLUDE_NOMINAL=1`: include the nominal sample in robust averaging (recommended with Gaussian weighting).
- `DEPHASE_QUASI_SAMPLER` (`grid` or `uniform`, default `grid`):
  quasi-static detuning sampler; `grid` uses evenly spaced detuning samples (resource-efficient with Gaussian weights).
- `DEPHASE_DETUNING_WEIGHTING` (`gaussian` or `uniform`, default `gaussian`):
  weighting used when aggregating robust rewards/fidelities across sampled noise trajectories.
  `gaussian` is recommended so large detunings do not dominate optimization.
- `DEPHASE_GAUSSIAN_SIGMA_FRAC` (default `0.50`):
  Gaussian weight width relative to `DEPHASE_DETUNING_FRAC * Omega`.
- `DEPHASE_STOCHASTIC_SIGMA_FRAC` (default `DEPHASE_DETUNING_FRAC`):
  per-time-step Gaussian std for `DEPHASE_MODEL=stochastic`.
- `DEPHASE_STOCHASTIC_STD_MODE` (`gamma_dt` or `frac_omega`, default `gamma_dt`):
  stochastic detuning-std rule.
  `gamma_dt` uses `sigma = sqrt(2*gamma/tau_c)` (teacher's formula).
  `frac_omega` uses `sigma = DEPHASE_STOCHASTIC_SIGMA_FRAC * Omega`.
- `DEPHASE_GAMMA` (default `18.0`): dephasing rate used when `DEPHASE_STOCHASTIC_STD_MODE=gamma_dt`.
- `DEPHASE_STOCHASTIC_CORRELATION` (`segment` or `step`, default `segment`):
  stochastic correlation time choice (`segment` matches one re-sample per pulse segment).
- `DEPHASE_STOCHASTIC_CLIP_STD` (default `3.0`):
  optional clip in units of stochastic std (set `0` to disable clipping).
- `DEPHASE_STOCHASTIC_SHARED_ACROSS_BATCH=1`:
  share the same stochastic trajectories across policy batch elements.
- Teacher-alignment guards (enabled by default):
  `STRICT_TEACHER_ALIGNMENT=1`, `FORCE_GAUSSIAN_WEIGHTING=1`,
  `FORCE_GRID_QUASI_SAMPLER=1`, `FORCE_STOCHASTIC_GAMMA_DT=1`,
  with Omega/t_step check controlled by
  `OMEGA_TSTEP_SCALE_REF_HZ`, `OMEGA_TSTEP_SCALE_REF_TSTEP`, `OMEGA_TSTEP_SCALE_TOL`.
- `ROBUST_NOMINAL_FID_FLOOR` (default `0.985`): nominal fidelity floor.
- `ROBUST_FLOOR_PENALTY` (default `0.0`): penalty multiplier for violating the floor.
  Keep at `0` for strict "optimize expected robust performance first" behavior.
- `DEPHASE_SWEEP_MAX_FRAC` / `DEPHASE_SWEEP_POINTS`:
  sweep range and resolution used for final fidelity-vs-dephasing curves.
- `ROBUST_COMPARE_BASELINE_NPZ`:
  optional baseline pulse file for robust-vs-nonrobust sweep comparison.
- `ROBUST_COMPARE_BASELINE_DURATION_SCALE` (optional):
  overrides baseline `duration_scale` used in robust-vs-baseline dephasing comparison.
  Useful when comparing to older baseline `.npz` files that do not store `duration_scale`.

## Outputs

- `outputs/final_fidelity.txt`: final evaluated fidelity.
- `outputs/final_pulses.npz`: final pulse waveforms (`phi_r`, `phi_b`, `amp_r`, `amp_b`)
  and optional `duration_scale` when duration control is enabled.
- `outputs/dephasing_sweep_robust.csv` + `outputs/dephasing_sweep_robust.png`:
  final pulse fidelity-vs-detuning curve.
- `outputs/dephasing_compare.csv` + `outputs/dephasing_compare.png`:
  robust-vs-baseline comparison (generated when `ROBUST_COMPARE_BASELINE_NPZ` is set).
- `eval_robust_metrics.csv`: robust evaluation summary (`R_rob`, `F_nom`, `F_rob`, `penalty`, `score`) per eval epoch.
- `outputs/final_robust_score.txt`: final robust score summary for the final pulse.
- `checkpoint/final_fidelity_best.txt` + `checkpoint/final_pulses_best.npz`: best-so-far run cache.
- `checkpoint/final_robust_score_best.txt` + `checkpoint/final_pulses_robust_best.npz`:
  best-so-far robust-score checkpoint in robust mode.

## Quick run

Run inside an activated environment on a compute node:

```bash
cd examples/trapped_ion_binomial
bash run_with_logs.sh
```

If your server/client dependencies are split across environments:

```bash
SERVER_PYTHON=/path/to/venv_tf/bin/python \
CLIENT_PYTHON=/path/to/venv_dq/bin/python \
POST_PYTHON=/path/to/venv_dq/bin/python \
bash run_with_logs.sh
```

Warm-start fine-tune (optional):

```bash
INIT_PULSES_NPZ=outputs/final_pulses.npz \
INIT_PULSE_BLEND=1.0 \
N_SEGMENTS=120 \
LEARN_AMP_R=1 LEARN_AMP_B=1 \
bash run_with_logs.sh
```

## Robust-vs-baseline workflow

1. Run non-robust baseline and keep its pulses:

```bash
ROBUST_TRAINING=0 \
bash run_with_logs.sh

cp outputs/final_pulses.npz checkpoint/nonrobust_baseline_pulses.npz
```

2. Run robust training and compare against the baseline pulses:

```bash
ROBUST_TRAINING=1 \
DEPHASE_MODEL=quasi_static \
OMEGA_RABI_HZ=2000 \
T_STEP=1e-5 \
LEARN_DURATION_SCALE=1 \
DEPHASE_DETUNING_FRAC=0.05 \
DEPHASE_NOISE_SAMPLES_TRAIN=7 \
DEPHASE_NOISE_SAMPLES_EVAL=13 \
DEPHASE_NOISE_SAMPLES_REFINE=17 \
ROBUST_NOMINAL_FID_FLOOR=0.985 \
ROBUST_FLOOR_PENALTY=0.0 \
ROBUST_COMPARE_BASELINE_NPZ=checkpoint/nonrobust_baseline_pulses.npz \
bash run_with_logs.sh
```

3. (Optional) switch to stochastic dephasing trajectories:

```bash
ROBUST_TRAINING=1 \
DEPHASE_MODEL=stochastic \
OMEGA_RABI_HZ=2000 \
T_STEP=1e-5 \
LEARN_DURATION_SCALE=1 \
DEPHASE_DETUNING_FRAC=0.05 \
DEPHASE_DETUNING_WEIGHTING=gaussian \
DEPHASE_STOCHASTIC_STD_MODE=gamma_dt \
DEPHASE_GAMMA=18.0 \
DEPHASE_STOCHASTIC_CORRELATION=segment \
DEPHASE_NOISE_SAMPLES_TRAIN=7 \
DEPHASE_NOISE_SAMPLES_EVAL=13 \
DEPHASE_NOISE_SAMPLES_REFINE=17 \
bash run_with_logs.sh
```
