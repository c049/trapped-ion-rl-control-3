# Model-Free Quantum Control with Reinforcement Learning
Fork of [v-sivak/quantum-control-rl](https://github.com/v-sivak/quantum-control-rl); in this version the server (RL agent) and client (experiment or sim) communicate over tcpip.  This may be a bottleneck for some applications, but it's very convenient for others.

This code was used in the following publications:

[**Model-Free Quantum Control with Reinforcement Learning**; Phys. Rev. X 12, 011059 (2022)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.011059)

[**Real-time quantum error correction beyond break-even**; Nature 616, 50–55 (2023)](https://www.nature.com/articles/s41586-023-05782-6)

[**High-fidelity, frequency-flexible two-qubit fluxonium gates with a transmon coupler**; arXiv:2304.06087 (2023)](https://arxiv.org/abs/2304.06087)

## Requirements
Requires a variety of packages, but for ease of use one should create the conda environment defined in qcrl-server-tf240.yml.  See the installation section for more details.  The included qcrl-server environment uses tensorflow v2.4.0 and tf-agents v0.6.0, which has been tested to work with CUDA v11.0 and cudnn v8.0.5 for GPU acceleration.  Without CUDA set up, this package will still work using the CPU, but this may limit performance depending on the application.

## Installation
To install this package, first clone this repository.  This package should be used with the conda environment defined in qcrl-server-tf240.yml.  To create this environment from the file, open an anaconda cmd prompt, cd into the repo directory, and run:
```sh
conda env create -f qcrl-server-tf240.yml
```
To install this package into this conda environment qcrl-server, first activate the environment using
```sh
conda activate qcrl-server
```
then cd into the repo directory and run:
```sh
pip install -e .
```

## CUDA Compatibility

The qcrl-server conda environment has been tested to work with CUDA v11.0 and cudnn v8.0.5. 

## Running the examples

Open two consoles, activate qcrl-server in both, and cd into the directory of the example you want to run in both (pi_pulse or pi_pulse_oct_style).  In one console run *_training_server.py, and in the other run *_client.py.



