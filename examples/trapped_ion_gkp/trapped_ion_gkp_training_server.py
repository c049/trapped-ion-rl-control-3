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
host_ip = "127.0.0.1"

FAST_SMOKE = os.environ.get("FAST_SMOKE", "0") == "1"
PPO_LR = float(os.environ.get("PPO_LR", "1.0e-4"))
PPO_ENTROPY_REG = float(os.environ.get("PPO_ENTROPY_REG", "5.0e-3"))
PPO_INIT_STD = float(os.environ.get("PPO_INIT_STD", "0.3"))
PPO_NUM_POLICY_UPDATES = int(os.environ.get("PPO_NUM_POLICY_UPDATES", "20"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "0"))
num_epochs = int(os.environ.get("NUM_EPOCHS", "300"))
train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "160"))

do_evaluation = True
eval_interval = int(os.environ.get("EVAL_INTERVAL", "20"))
eval_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "5"))
num_policy_updates = PPO_NUM_POLICY_UPDATES

learn_residuals = True
save_tf_style = False

n_steps = 120
n_segments = 60
init_phi_r = list(np.zeros(n_segments))
init_phi_b = list(np.zeros(n_segments))
init_amp_r = list(np.ones(n_segments))
init_amp_b = list(np.ones(n_segments))

phase_scale = list(np.ones(n_segments, dtype=float) * np.pi)
amp_scale = list(np.ones(n_segments, dtype=float) * 1.0)

action_script = {
    "phi_r": [init_phi_r],
    "phi_b": [init_phi_b],
    "amp_r": [init_amp_r],
    "amp_b": [init_amp_b],
}

action_spec = {
    "phi_r": specs.TensorSpec(shape=[n_segments], dtype=tf.float32),
    "phi_b": specs.TensorSpec(shape=[n_segments], dtype=tf.float32),
    "amp_r": specs.TensorSpec(shape=[n_segments], dtype=tf.float32),
    "amp_b": specs.TensorSpec(shape=[n_segments], dtype=tf.float32),
}

action_scale = {
    "phi_r": phase_scale,
    "phi_b": phase_scale,
    "amp_r": amp_scale,
    "amp_b": amp_scale,
}

to_learn = {
    "phi_r": True,
    "phi_b": True,
    "amp_r": False,
    "amp_b": False,
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
    "num_epochs": num_epochs,
    "train_batch_size": train_batch_size,
    "do_evaluation": do_evaluation,
    "eval_interval": eval_interval,
    "eval_batch_size": eval_batch_size,
    "learn_residuals": learn_residuals,
    "action_script": action_script,
    "action_scale": action_scale,
    "to_learn": to_learn,
    "save_tf_style": save_tf_style,
}

log = h5log(root_dir, rl_params)

from quantum_control_rl_server import dynamic_episode_driver_sim_env

server_socket = rmt.Server()
(host, port) = (host_ip, 5555)
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
    actor_fc_layers=(50, 20),
    value_fc_layers=(),
    use_rnn=False,
    actor_lstm_size=(12,),
    value_lstm_size=(12,),
    h5datalog=log,
    save_tf_style=save_tf_style,
    rl_params=rl_params,
)

#%%
