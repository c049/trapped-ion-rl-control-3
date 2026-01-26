import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from quantum_control_rl_server.remote_env_tools import Client
from trapped_ion_cat_sim_function import (
    trapped_ion_cat_sim,
    wigner_grid,
    select_wigner_points,
)

logger = logging.getLogger("RL")
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

client_socket = Client()
(host, port) = ("127.0.0.1", 5555)
client_socket.connect((host, port))

N_STEPS = 120
N_SEGMENTS = 30
SEG_LEN = N_STEPS // N_SEGMENTS
T_STEP = 10.0
SAMPLE_EXTENT = 3.0

TRAIN_POINTS = select_wigner_points(
    alpha_cat=2.0,
    n_boson=20,
    extent=SAMPLE_EXTENT,
    grid_size=41,
    top_k=60,
    cat_parity="even",
)
EVAL_POINTS = select_wigner_points(
    alpha_cat=2.0,
    n_boson=20,
    extent=SAMPLE_EXTENT,
    grid_size=41,
    top_k=60,
    cat_parity="even",
)
FINAL_POINTS = select_wigner_points(
    alpha_cat=2.0,
    n_boson=20,
    extent=SAMPLE_EXTENT,
    grid_size=61,
    top_k=120,
    cat_parity="even",
)

done = False
eval_log_path = os.path.join(os.getcwd(), "eval_fidelity.csv")
while not done:
    message, done = client_socket.recv_data()
    logger.info("Received message from RL agent server.")
    logger.info("Time stamp: %f", time.time())

    if done:
        logger.info("Training finished.")
        break

    epoch_type = message["epoch_type"]

    if epoch_type == "final":
        logger.info("Final Epoch")
        locs = message["locs"]
        scales = message["scales"]
        for key in locs.keys():
            logger.info("locs[%s]:", key)
            logger.info(locs[key][0])
            logger.info("scales[%s]:", key)
            logger.info(scales[key][0])
        phi_r_final = np.repeat(np.array(locs["phi_r"][0]), SEG_LEN)
        phi_b_final = np.repeat(np.array(locs["phi_b"][0]), SEG_LEN)
        amp_r_final = np.repeat(np.array(locs["amp_r"][0]), SEG_LEN)
        amp_b_final = np.repeat(np.array(locs["amp_b"][0]), SEG_LEN)

        _, final_fidelity, rho_final, rho_target = trapped_ion_cat_sim(
            phi_r_final,
            phi_b_final,
            amp_r=amp_r_final,
            amp_b=amp_b_final,
            n_boson=20,
            omega=2 * np.pi * 0.002,
            t_step=T_STEP,
            alpha_cat=2.0,
            cat_parity="even",
            sample_points=FINAL_POINTS,
            n_shots=0,
            return_details=True,
        )

        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        fidelity_path = os.path.join(output_dir, "final_fidelity.txt")
        with open(fidelity_path, "w", encoding="utf-8") as f:
            f.write(f"{final_fidelity:.6f}\n")
        logger.info("Final fidelity %.6f", final_fidelity)
        logger.info("Saved final fidelity to %s", fidelity_path)

        grid = np.linspace(-SAMPLE_EXTENT, SAMPLE_EXTENT, 121)
        w_target = wigner_grid(rho_target, grid, grid)
        w_final = wigner_grid(rho_final, grid, grid)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(
            w_target,
            extent=[-SAMPLE_EXTENT, SAMPLE_EXTENT, -SAMPLE_EXTENT, SAMPLE_EXTENT],
            origin="lower",
            cmap="RdBu_r",
        )
        axes[0].set_title("Target cat Wigner")
        axes[0].set_xlabel("Re(alpha)")
        axes[0].set_ylabel("Im(alpha)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(
            w_final,
            extent=[-SAMPLE_EXTENT, SAMPLE_EXTENT, -SAMPLE_EXTENT, SAMPLE_EXTENT],
            origin="lower",
            cmap="RdBu_r",
        )
        axes[1].set_title("Final state Wigner")
        axes[1].set_xlabel("Re(alpha)")
        axes[1].set_ylabel("Im(alpha)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        wigner_path = os.path.join(output_dir, "wigner_target_vs_final.png")
        fig.savefig(wigner_path, dpi=150)
        plt.close(fig)
        logger.info("Saved Wigner plot to %s", wigner_path)

        done = True
        logger.info("Training finished.")
        break

    action_batch = message["action_batch"]
    batch_size = message["batch_size"]
    epoch = message["epoch"]

    phi_r_coeff = action_batch["phi_r"].reshape([batch_size, -1])
    phi_b_coeff = action_batch["phi_b"].reshape([batch_size, -1])
    amp_r_coeff = action_batch["amp_r"].reshape([batch_size, -1])
    amp_b_coeff = action_batch["amp_b"].reshape([batch_size, -1])
    phi_r = np.repeat(phi_r_coeff, SEG_LEN, axis=1)
    phi_b = np.repeat(phi_b_coeff, SEG_LEN, axis=1)
    amp_r = np.repeat(amp_r_coeff, SEG_LEN, axis=1)
    amp_b = np.repeat(amp_b_coeff, SEG_LEN, axis=1)

    logger.info("Start %s epoch %d", epoch_type, epoch)

    if epoch_type == "evaluation":
        n_shots = 0
        sample_points = EVAL_POINTS
    else:
        n_shots = 0
        sample_points = TRAIN_POINTS

    reward_data = np.zeros((batch_size))
    fidelity_data = None
    if epoch_type == "evaluation":
        fidelity_data = np.zeros((batch_size))
    for ii in range(batch_size):
        if epoch_type == "evaluation":
            reward, fidelity, _, _ = trapped_ion_cat_sim(
                phi_r[ii],
                phi_b[ii],
                amp_r=amp_r[ii],
                amp_b=amp_b[ii],
                n_boson=20,
                omega=2 * np.pi * 0.002,
                t_step=T_STEP,
                alpha_cat=2.0,
                cat_parity="even",
                sample_points=sample_points,
                n_shots=n_shots,
                return_details=True,
            )
            reward_data[ii] = reward
            fidelity_data[ii] = fidelity
        else:
            reward_data[ii] = trapped_ion_cat_sim(
                phi_r[ii],
                phi_b[ii],
                amp_r=amp_r[ii],
                amp_b=amp_b[ii],
                n_boson=20,
                omega=2 * np.pi * 0.002,
                t_step=T_STEP,
                alpha_cat=2.0,
                cat_parity="even",
                sample_points=sample_points,
                n_shots=n_shots,
            )

    R = np.mean(reward_data)
    std_R = np.std(reward_data)
    logger.info("Average reward %.3f", R)
    logger.info("STDev reward %.3f", std_R)
    if fidelity_data is not None:
        mean_fidelity = float(np.mean(fidelity_data))
        std_fidelity = float(np.std(fidelity_data))
        logger.info("Eval fidelity %.6f", mean_fidelity)
        write_header = not os.path.exists(eval_log_path)
        with open(eval_log_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("epoch,mean_fidelity,std_fidelity\n")
            f.write(f"{epoch},{mean_fidelity:.6f},{std_fidelity:.6f}\n")

    logger.info("Sending message to RL agent server.")
    logger.info("Time stamp: %f", time.time())
    client_socket.send_data(reward_data)
