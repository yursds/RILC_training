# Reinforced Iterative Learning Control (RILC)

## Overview

This repository contains the implementation of the **Reinforced Iterative Learning Control (RILC)** method.
It provides an environment built with [Gymnasium](https://gymnasium.farama.org/) and [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html) to train an agent using reinforcement learning (RL) techniques via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/).

The repository environment and dependencies are managed natively using [uv](https://docs.astral.sh/uv/).

### Repository Structure

- The `classes` folder contains some submodules to facilitate the creation of robots (using [Pinocchio](https://github.com/stack-of-tasks/pinocchio/tree/master)), references, controllers, environments, and callbacks (during training).
- The `utils` folder contains some useful functions to load `.yaml` configuration files more easily.
- Training scripts (`train_ppo.py`, `train_ppo_continue.py`) and evaluation scripts (`test_rilc.py`, `test_noilc.py`) are located directly in the root directory.
- The `log` and `model` folders are generated and populated automatically during training runs to store telemetry and checkpoints.

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

   To train a new policy from scratch:
   ```bash
   uv run train_ppo.py
   ```
   To continue training an existing policy:
   ```bash
   uv run train_ppo_continue.py
   ```

4. Test policies:

   Once trained, you can evaluate the policies using the testing scripts:
   ```bash
   uv run test_rilc.py
   uv run test_noilc.py
   ```
