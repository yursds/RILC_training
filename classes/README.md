# Controllers, Robots, References, Environments, Callbacks

## Controllers

Simple implementations of ILC (Iterative Learning Control) and PD (Proportional-Derivative) controllers.

## Robots

Examples of generalizing main functions for controlling robotic systems through different classes. Additionally, a folder for a specific robot containing .urdf, .xml, and mesh files is provided. The primary libraries used for implementation are MuJoCo, Pinocchio, and Torch.

## References

This folder leverages the Torch library to create time-dependent references using automatic differentiation for the derivatives.

## Environments

The core of this project is to create an environment for training an agent with reinforcement learning (RL). The main library used is Gymnasium.

## Callbacks

To visualize specific data during RL training, a custom callback is implemented.
