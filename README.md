# Reinforced Iterative Learning Control (RILC)

## Overview

This repository contains an example of how to build an environment with [Gymnasium](https://gymnasium.farama.org/) and [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html) to train an agent using reinforcement learning (RL) techniques implemented in [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/).\
In particular, the [Reinforced Iterative Learning Control (RILC)]() is implemented.

The repository environment is managed directly natively. You can install the dependencies using [uv](https://docs.astral.sh/uv/).

- The `classes` folder contains some submodules to facilitate the creation of robots (using [Pinocchio](https://github.com/stack-of-tasks/pinocchio/tree/master)), references, controllers, environments, and callbacks (during training).

- The `scripts` folder contains benchmark scripts. The main training script `train_ppo.py` is located in the root directory.

- The `log` and `model` folders are populated during training, and the **rl_classic** and **rilc** models used for the [paper]() are trained already.

- The trained policy can be tested with `scripts/compare_controllers.py` or other benchmark scripts.

- The `utils` folder contains some useful functions to load .yaml files more easily.

Further information can be found in the respective folders.

## Prerequisites

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python dependency management.

## Installation and First Setup

1. Clone repository and navigate to it:

   ```bash
   git clone https://github.com/yursds/RILC_training.git
   cd ./RILC_training
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Run training:

   ```bash
   uv run train_ppo.py
   ```

4. Test policies:

   ```bash
   uv run test_rilc.py
   uv run test_noilc.py
   ```


