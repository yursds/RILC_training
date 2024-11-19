# Reinforced Iterative Learning Control (RILC)

## Overview

This repository contains an example of how to build an environment with [Gymnasium](https://gymnasium.farama.org/) and [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html) to train an agent using reinforcement learning (RL) techniques implemented in [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/).\
In particular, the [Reinforced Iterative Learning Control (RILC)]() is implemented.

The repository is built with a [Docker container](https://docs.docker.com/desktop/) to ensure reproducibility. For more information follow [Docker Installation Preliminaries](#docker_install).

- The `_setup_` folder is used to finalize the *.urdf file and fix a [bug](https://github.com/Farama-Foundation/Gymnasium/pull/746) within MuJoCo installation in Gymnasium.
- The `classes` folder contains some submodules to facilitate the creation of robots (using [Pinocchio](https://github.com/stack-of-tasks/pinocchio/tree/master)), references, controllers, environments, and callbacks (during training).

- The `config` folder contains the main parameters to start the training by executing `train_rlilc.py`. 

- The `log` and `model` folders are populated during training, and the **rl_classic** and **rilc** models used for the [paper]() are trained already. 

- The trained policy can be tested with `test_rlilc.py`.

- The `utils` folder contains some useful functions to load .yaml files more easily.

Further information can be found in the respective folders.

## Docker Installation Preliminaries
1. Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/), (suggesting `apt` installation). 
    - Follow [post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/).
3. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Installation and First Setup

1. Clone repository and navigate to it:

    ```bash
    git clone https://github.com/yursds/RILC_training.git
    cd ./RILC_training
    ```

2. Build and Run Docker Image

    To build image `RILC_training`:
    ```bash
    ./.dockercontainer/build.bash
    ```
    To run image `RILC_training`:
    ```bash
    ./.dockercontainer/run.bash
    ```

    - Run the following command <ins>**only the first time after cloning**</ins> to resolve [bug](https://github.com/Farama-Foundation/Gymnasium/pull/746) and create the `leg_constrained.urdf` file for correct loading of robot:

        ```bash
        # Resolve bug in MuJoCo rendering
        python3 ./_setup_/change_mujoco_rendering.py
        # Creates absolute paths for the meshes
        python3 ./_setup_/gen_urdf.py
        ```
