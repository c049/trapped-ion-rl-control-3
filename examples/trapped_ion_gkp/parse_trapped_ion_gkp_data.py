import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

root_dir = os.getcwd()
h5_files = sorted(glob.glob(os.path.join(root_dir, "*.h5")))
if not h5_files:
    raise SystemExit("No .h5 files found in this directory.")

filename = h5_files[-1]

with h5py.File(filename, "r") as f:
    runs = [k for k in f.keys() if k.isdigit()]
    if not runs:
        raise SystemExit("No numeric run groups found in the h5 file.")
    run = sorted(runs, key=int)[-1]

    rl_params = dict(f[run]["rl_params"].attrs.items())
    train_rewards = f[run]["training"]["rewards"][()]
    eval_rewards = None
    if "evaluation" in f[run]:
        eval_rewards = f[run]["evaluation"]["rewards"][()]

epochs = np.arange(train_rewards.shape[0])
train_mean = np.mean(train_rewards, axis=1)
train_std = np.std(train_rewards, axis=1)

output_dir = os.path.join(root_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "training_curve.png")

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.plot(epochs, train_mean, label="train reward")
ax.fill_between(
    epochs,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.3,
)

if eval_rewards is not None:
    eval_interval = int(rl_params.get("eval_interval", 1))
    eval_epochs = np.arange(0, eval_rewards.shape[0]) * eval_interval
    eval_mean = np.mean(eval_rewards, axis=1)
    ax.plot(eval_epochs, eval_mean, "o", label="eval reward")

ax.set_xlabel("Epoch")
ax.set_ylabel("Characteristic reward")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(plot_path, dpi=150)
plt.close(fig)

fidelity_path = os.path.join(root_dir, "eval_fidelity.csv")
if os.path.exists(fidelity_path):
    data = np.loadtxt(fidelity_path, delimiter=",", skiprows=1)
    if data.size:
        fig_f, ax_f = plt.subplots(1, 1, figsize=(7, 4))
        ax_f.plot(data[:, 0], data[:, 1], "o-")
        ax_f.set_xlabel("Epoch")
        ax_f.set_ylabel("Eval fidelity")
        fig_f.tight_layout()
        fid_plot_path = os.path.join(output_dir, "eval_fidelity_curve.png")
        fig_f.savefig(fid_plot_path, dpi=150)
        plt.close(fig_f)
        print(f"Saved eval fidelity plot to {fid_plot_path}")

print(f"Saved training curve to {plot_path}")
print(f"Used data from {filename} (run {run})")
