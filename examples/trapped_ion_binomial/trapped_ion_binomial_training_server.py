#%%

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import numpy as np
import tensorflow as tf
from tf_agents import specs
from tf_agents.networks import actor_distribution_network

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from quantum_control_rl_server import PPO
from quantum_control_rl_server import remote_env_tools as rmt
from quantum_control_rl_server.h5log import h5log

root_dir = os.getcwd()
host_ip = os.environ.get("HOST", "127.0.0.1")
host_port = int(os.environ.get("PORT", "5555"))

FAST_SMOKE = os.environ.get("FAST_SMOKE", "0") == "1"
PPO_LR = float(os.environ.get("PPO_LR", "1.0e-4"))
PPO_ENTROPY_REG = float(os.environ.get("PPO_ENTROPY_REG", "5.0e-3"))
PPO_INIT_STD = float(os.environ.get("PPO_INIT_STD", "0.3"))
PPO_NUM_POLICY_UPDATES = int(os.environ.get("PPO_NUM_POLICY_UPDATES", "20"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "0"))
num_epochs = int(os.environ.get("NUM_EPOCHS", "300"))
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "160"))
PHASE_ACTION_SCALE = float(os.environ.get("PHASE_ACTION_SCALE", str(np.pi)))
AMP_ACTION_SCALE = float(os.environ.get("AMP_ACTION_SCALE", "1.0"))
LEARN_AMP_R = os.environ.get("LEARN_AMP_R", "0") == "1"
LEARN_AMP_B = os.environ.get("LEARN_AMP_B", "0") == "1"
LEARN_DURATION_SCALE = os.environ.get("LEARN_DURATION_SCALE", "1") == "1"
DURATION_INIT_SCALE = float(os.environ.get("DURATION_INIT_SCALE", "1.0"))
DURATION_ACTION_SCALE = float(os.environ.get("DURATION_ACTION_SCALE", "0.25"))
INIT_PULSES_NPZ = os.environ.get("INIT_PULSES_NPZ", "").strip()
INIT_PULSE_BLEND = float(os.environ.get("INIT_PULSE_BLEND", "1.0"))


def _parse_fc_layers(env_name, default):
    raw = os.environ.get(env_name, "").strip()
    if raw == "":
        return tuple(default)
    vals = tuple(int(v.strip()) for v in raw.split(",") if v.strip())
    if any(v <= 0 for v in vals):
        raise ValueError(f"{env_name} values must be positive integers, got {vals}")
    return vals


ACTOR_FC_LAYERS = _parse_fc_layers("ACTOR_FC_LAYERS", (50, 20))
VALUE_FC_LAYERS = _parse_fc_layers("VALUE_FC_LAYERS", ())

do_evaluation = True
eval_interval = int(os.environ.get("EVAL_INTERVAL", "20"))
eval_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "5"))
num_policy_updates = PPO_NUM_POLICY_UPDATES

learn_residuals = True
save_tf_style = False

n_steps = int(os.environ.get("N_STEPS", "120"))
n_segments = int(os.environ.get("N_SEGMENTS", "60"))
if n_steps <= 0 or n_segments <= 0:
    raise ValueError("N_STEPS and N_SEGMENTS must be positive.")
if n_steps % n_segments != 0:
    raise ValueError(
        f"N_STEPS ({n_steps}) must be divisible by N_SEGMENTS ({n_segments})."
    )
init_phi_r = list(np.zeros(n_segments))
init_phi_b = list(np.zeros(n_segments))
init_amp_r = list(np.ones(n_segments))
init_amp_b = list(np.ones(n_segments))
init_duration_scale = [DURATION_INIT_SCALE]

def _segment_phase_from_full(arr, n_segments):
    vals = np.asarray(arr, dtype=float).reshape(-1)
    if vals.size == n_segments:
        return vals.copy()
    if vals.size % n_segments == 0:
        seg_len = vals.size // n_segments
        reshaped = vals.reshape(n_segments, seg_len)
        return np.angle(np.mean(np.exp(1j * reshaped), axis=1))
    src = np.arange(vals.size, dtype=float)
    dst = (np.arange(n_segments, dtype=float) + 0.5) * (vals.size / n_segments) - 0.5
    dst = np.clip(dst, 0.0, max(float(vals.size - 1), 0.0))
    unwrapped = np.unwrap(vals)
    interp = np.interp(dst, src, unwrapped)
    return np.angle(np.exp(1j * interp))


def _segment_amp_from_full(arr, n_segments):
    vals = np.asarray(arr, dtype=float).reshape(-1)
    if vals.size == n_segments:
        return vals.copy()
    if vals.size % n_segments == 0:
        seg_len = vals.size // n_segments
        reshaped = vals.reshape(n_segments, seg_len)
        return np.mean(reshaped, axis=1)
    src = np.arange(vals.size, dtype=float)
    dst = (np.arange(n_segments, dtype=float) + 0.5) * (vals.size / n_segments) - 0.5
    dst = np.clip(dst, 0.0, max(float(vals.size - 1), 0.0))
    return np.interp(dst, src, vals)


if not np.isfinite(INIT_PULSE_BLEND) or INIT_PULSE_BLEND < 0.0 or INIT_PULSE_BLEND > 1.0:
    raise ValueError(f"INIT_PULSE_BLEND must be in [0,1], got {INIT_PULSE_BLEND}")
if not np.isfinite(DURATION_INIT_SCALE) or DURATION_INIT_SCALE <= 0.0:
    raise ValueError(
        f"DURATION_INIT_SCALE must be finite and > 0, got {DURATION_INIT_SCALE}"
    )
if not np.isfinite(DURATION_ACTION_SCALE) or DURATION_ACTION_SCALE <= 0.0:
    raise ValueError(
        f"DURATION_ACTION_SCALE must be finite and > 0, got {DURATION_ACTION_SCALE}"
    )

if INIT_PULSES_NPZ:
    pulse_path = (
        INIT_PULSES_NPZ
        if os.path.isabs(INIT_PULSES_NPZ)
        else os.path.join(root_dir, INIT_PULSES_NPZ)
    )
    if not os.path.exists(pulse_path):
        raise FileNotFoundError(f"INIT_PULSES_NPZ not found: {pulse_path}")
    with np.load(pulse_path) as pulse_data:
        req = ("phi_r", "phi_b", "amp_r", "amp_b")
        missing = [k for k in req if k not in pulse_data]
        if missing:
            raise ValueError(f"INIT_PULSES_NPZ missing keys {missing} in {pulse_path}")
        raw_phi_r = np.asarray(pulse_data["phi_r"], dtype=float).reshape(-1)
        raw_phi_b = np.asarray(pulse_data["phi_b"], dtype=float).reshape(-1)
        raw_amp_r = np.asarray(pulse_data["amp_r"], dtype=float).reshape(-1)
        raw_amp_b = np.asarray(pulse_data["amp_b"], dtype=float).reshape(-1)
        raw_duration_scale = np.asarray(
            pulse_data["duration_scale"] if "duration_scale" in pulse_data else [DURATION_INIT_SCALE],
            dtype=float,
        ).reshape(-1)
        init_phi_r_loaded = _segment_phase_from_full(raw_phi_r, n_segments)
        init_phi_b_loaded = _segment_phase_from_full(raw_phi_b, n_segments)
        init_amp_r_loaded = _segment_amp_from_full(raw_amp_r, n_segments)
        init_amp_b_loaded = _segment_amp_from_full(raw_amp_b, n_segments)
        init_duration_scale_loaded = float(raw_duration_scale[0]) if raw_duration_scale.size else DURATION_INIT_SCALE
    init_phi_r = list(
        (1.0 - INIT_PULSE_BLEND) * np.asarray(init_phi_r, dtype=float)
        + INIT_PULSE_BLEND * init_phi_r_loaded
    )
    init_phi_b = list(
        (1.0 - INIT_PULSE_BLEND) * np.asarray(init_phi_b, dtype=float)
        + INIT_PULSE_BLEND * init_phi_b_loaded
    )
    init_amp_r = list(
        (1.0 - INIT_PULSE_BLEND) * np.asarray(init_amp_r, dtype=float)
        + INIT_PULSE_BLEND * init_amp_r_loaded
    )
    init_amp_b = list(
        (1.0 - INIT_PULSE_BLEND) * np.asarray(init_amp_b, dtype=float)
        + INIT_PULSE_BLEND * init_amp_b_loaded
    )
    init_duration_scale = [
        float((1.0 - INIT_PULSE_BLEND) * init_duration_scale[0] + INIT_PULSE_BLEND * init_duration_scale_loaded)
    ]
    print(
        "Warm-start enabled: INIT_PULSES_NPZ=%s blend=%.3f "
        "(source lens: phi_r=%d phi_b=%d amp_r=%d amp_b=%d duration=%d -> segments=%d)"
        % (
            pulse_path,
            INIT_PULSE_BLEND,
            len(raw_phi_r),
            len(raw_phi_b),
            len(raw_amp_r),
            len(raw_amp_b),
            len(raw_duration_scale),
            n_segments,
        )
    )

phase_scale = list(np.ones(n_segments, dtype=float) * PHASE_ACTION_SCALE)
amp_scale = list(np.ones(n_segments, dtype=float) * AMP_ACTION_SCALE)

action_script = {
    "phi_r": [init_phi_r],
    "phi_b": [init_phi_b],
    "amp_r": [init_amp_r],
    "amp_b": [init_amp_b],
    "duration_scale": [init_duration_scale],
}

action_spec = {
    "phi_r": specs.TensorSpec(shape=[n_segments], dtype=tf.float32),
    "phi_b": specs.TensorSpec(shape=[n_segments], dtype=tf.float32),
    "amp_r": specs.TensorSpec(shape=[n_segments], dtype=tf.float32),
    "amp_b": specs.TensorSpec(shape=[n_segments], dtype=tf.float32),
    "duration_scale": specs.TensorSpec(shape=[1], dtype=tf.float32),
}

action_scale = {
    "phi_r": phase_scale,
    "phi_b": phase_scale,
    "amp_r": amp_scale,
    "amp_b": amp_scale,
    "duration_scale": [DURATION_ACTION_SCALE],
}

to_learn = {
    "phi_r": os.environ.get("LEARN_PHI_R", "1") == "1",
    "phi_b": os.environ.get("LEARN_PHI_B", "1") == "1",
    "amp_r": LEARN_AMP_R,
    "amp_b": LEARN_AMP_B,
    "duration_scale": LEARN_DURATION_SCALE,
}

if FAST_SMOKE:
    num_epochs = 4
    train_batch_size = 2
    eval_interval = 2
    eval_batch_size = 2
    num_policy_updates = 2

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

rl_params = {
    "random_seed": RANDOM_SEED,
    "host": host_ip,
    "port": host_port,
    "num_epochs": num_epochs,
    "train_batch_size": train_batch_size,
    "n_steps": n_steps,
    "n_segments": n_segments,
    "do_evaluation": do_evaluation,
    "eval_interval": eval_interval,
    "eval_batch_size": eval_batch_size,
    "learn_residuals": learn_residuals,
    "phase_action_scale": PHASE_ACTION_SCALE,
    "amp_action_scale": AMP_ACTION_SCALE,
    "learn_amp_r": LEARN_AMP_R,
    "learn_amp_b": LEARN_AMP_B,
    "learn_duration_scale": LEARN_DURATION_SCALE,
    "duration_init_scale": DURATION_INIT_SCALE,
    "duration_action_scale": DURATION_ACTION_SCALE,
    "init_pulses_npz": INIT_PULSES_NPZ,
    "init_pulse_blend": INIT_PULSE_BLEND,
    "actor_fc_layers": ACTOR_FC_LAYERS,
    "value_fc_layers": VALUE_FC_LAYERS,
    "action_script": action_script,
    "action_scale": action_scale,
    "to_learn": to_learn,
    "save_tf_style": save_tf_style,
}

log = h5log(root_dir, rl_params)

from quantum_control_rl_server import dynamic_episode_driver_sim_env

server_socket = rmt.Server()
(host, port) = (host_ip, host_port)
server_socket.bind((host, port))
server_socket.connect_client()

env_kwargs = eval_env_kwargs = {
    "T": 1,
}

reward_kwargs = {
    "reward_mode": "remote",
    "server_socket": server_socket,
    "epoch_type": "training",
}

reward_kwargs_eval = {
    "reward_mode": "remote",
    "server_socket": server_socket,
    "epoch_type": "evaluation",
}

collect_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    env_kwargs,
    reward_kwargs,
    train_batch_size,
    action_script,
    action_scale,
    action_spec,
    to_learn,
    learn_residuals,
    remote=True,
)

eval_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    eval_env_kwargs,
    reward_kwargs_eval,
    eval_batch_size,
    action_script,
    action_scale,
    action_spec,
    to_learn,
    learn_residuals,
    remote=True,
)

PPO.train_eval(
    root_dir=root_dir,
    random_seed=RANDOM_SEED,
    num_epochs=num_epochs,
    normalize_observations=True,
    normalize_rewards=False,
    discount_factor=1.0,
    lr=PPO_LR,
    lr_schedule=None,
    num_policy_updates=num_policy_updates,
    initial_adaptive_kl_beta=0.0,
    kl_cutoff_factor=0,
    importance_ratio_clipping=0.1,
    value_pred_loss_coef=0.005,
    gradient_clipping=1.0,
    entropy_regularization=PPO_ENTROPY_REG,
    log_prob_clipping=0.0,
    eval_interval=eval_interval,
    save_interval=2,
    checkpoint_interval=None,
    summary_interval=2,
    do_evaluation=do_evaluation,
    train_batch_size=train_batch_size,
    eval_batch_size=eval_batch_size,
    collect_driver=collect_driver,
    eval_driver=eval_driver,
    replay_buffer_capacity=15000,
    ActorNet=actor_distribution_network.ActorDistributionNetwork,
    zero_means_kernel_initializer=False,
    init_action_stddev=PPO_INIT_STD,
    actor_fc_layers=ACTOR_FC_LAYERS,
    value_fc_layers=VALUE_FC_LAYERS,
    use_rnn=False,
    actor_lstm_size=(12,),
    value_lstm_size=(12,),
    h5datalog=log,
    save_tf_style=save_tf_style,
    rl_params=rl_params,
)

#%%
