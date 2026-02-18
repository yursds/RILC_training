# Algorithms and Implementation

This document details the mathematical formulation and implementation of the control strategies used in this project, specifically the Reinforced Iterative Learning Control (RILC) framework.

## 1. System Dynamics (Series Elastic Actuators)

The system is modeled as a 2n-DoFs under-actuated robot with Series Elastic Actuators (SEA). The dynamics are given by [35]:

$$
\left\{
\begin{array}{l}
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) + K(q-\theta) = 0 \\
B\ddot{\theta} + D\dot{\theta} - K(q-\theta) = u
\end{array}
\right. \quad (1)
$$

Where:

- $q, \dot{q}, \ddot{q} \in \mathbb{R}^n$ are the position, velocity, and acceleration vectors on the link-side.
- $\theta, \dot{\theta}, \ddot{\theta} \in \mathbb{R}^n$ are the position, velocity, and acceleration vectors on the motor-side.
- $M(q) \in \mathbb{R}^{n \times n}$ is the robot inertia matrix.
- $C(q,\dot{q}) \in \mathbb{R}^{n \times n}$ is the robot Coriolis matrix.
- $G(q) \in \mathbb{R}^n$ is the gravity vector.
- $K \in \mathbb{R}^{n \times n}$ is the stiffness matrix associated to SEA.
- $B, D \in \mathbb{R}^{n \times n}$ are the motor inertia matrix and damping matrix ($D \succ 0$).
- $u \in \mathbb{R}^n$ is the torque control input.

We consider a set $\mathcal{P} = \{P_1, \dots, P_k, \dots P_m\}$ where $P_k$ represents the $k$-th iterative process. Following the classic affine state-space form, we define the state $x_j \doteq [q_j^\top, \theta_j^\top, \dot{q}_j^\top, \dot{\theta}_j^\top]^\top \in \mathbb{R}^{4n}$ for continuous-time systems with $t \in [0, t_f]$:

$$
\begin{cases}
\dot{x}_j = f(x_j) + g(x_j)[\bar{u}_j + \tilde{u}_\nu(\xi_j)] \\
y_j = h(x_j) \\
\xi_j = \phi(y_d, y_{j-I}, \bar{u}_{j-I})
\end{cases} \quad (2)
$$

Where:

- $f(x_j) = \begin{bmatrix} \dot{q}_j \\ \dot{\theta}_j \\ -N_1(q_j, \theta_j, \dot{q}_j) \\ -N_2(q_j, \theta_j, \dot{\theta}_j) \end{bmatrix}$
- $g(x_j) = \begin{bmatrix} 0_{3n \times n} \\ B^{-1} \end{bmatrix}$
- $N_1(q_j, \theta_j, \dot{q}_j) \doteq M^{-1}(q_j)[C(q_j, \dot{q}_j)\dot{q}_j + G(q_j) + K(q_j - \theta_j)]$
- $N_2(q_j, \theta_j, \dot{\theta}_j) \doteq B^{-1}[D\dot{\theta}_j - K(q_j - \theta_j)]$
- $y_j = h(x_j) = S_\theta x_j$ is the output, with $S_\theta = [0_{n \times n}, I_n, 0_{n \times 2n}]$.

## 2. Iterative Learning Control (ILC)

The ILC component works in a feedforward fashion to track a desired trajectory $y_d(t)$. The update law for the feedforward term $\bar{u}_j(t)$ at iteration $j$ is:

$$
\bar{u}_{j+1}(t) = \bar{u}_j(t) + L_j(t) \sum_{i=0}^r \chi_i e_j^{(i)}(t)
$$

Where:

- $e_j^{(i)}(t) = y_d^{(i)}(t) - y_j^{(i)}(t)$ is the $i$-th derivative of the error.
- $L_j(t)$ is the learning gain.
- $\chi_i$ are tunable control gains.

This update guarantees asymptotic convergence of the tracking error under standard assumptions (Lipschitz continuity, strictly proper system, etc.).

## 3. Reinforcement Learning (RL) - PPO

We use Proximal Policy Optimization (PPO) to train a stochastic policy $\tilde{u}_\nu(\xi_j)$ parameterized by $\nu$. The policy is updated to maximize a reward function (minimize cost) using the gradient update:

$$
\nu \leftarrow \nu - \nabla_\nu {\bar{J}}(\cdot, \nu)
$$

The RL agent observes the state $\xi_j(t) = \phi(y_d, y_{j-I}, \bar{u}_{j-I})$ which includes the desired trajectory and past ILC iterations.

## 4. Reinforced Iterative Learning Control (RILC)

The RILC framework synergistically combines ILC and RL. The total control input applied to the system at iteration $j$ is:

$$
u_j(t) = \bar{u}_j(t) + \tilde{u}_\nu(\xi_j(t)) + u_{mb}(t) + u_{fb}(t)
$$

Where:

- $\bar{u}_j(t)$: ILC feedforward term (learns from repetition).
- $\tilde{u}_\nu(\xi_j(t))$: RL policy (learns to generalize and compensate for non-repetitive disturbances).
- $u_{mb}(t)$: Model-based term (e.g., gravity compensation).
- $u_{fb}(t)$: Feedback term (e.g., PD controller) for stabilization.

## 5. Norm Optimal ILC (NOILC)

Minimizes a quadratic cost function balancing error and input change:
$$J(u_{j+1}) = e_{j+1}^T Q e_{j+1} + (u_{j+1} - u_j)^T R (u_{j+1} - u_j)$$

Update law:
$$u_{j+1} = u_j + (G^T Q G + R)^{-1} G^T Q e_j$$

**Note on G Calculation**:
The Lifted System Matrix **G** is computed via **model-based linearization**. The non-linear dynamics (Equation 2) are linearized at each time step along the reference trajectory $y_d(t)$ (and corresponding state/input) to obtain time-varying matrices $A_k, B_k$. These are then assembled into the lifted matrix $G$:

$$
G = \begin{bmatrix}
CB_0 & 0 & \dots & 0 \\
CA_1B_0 & CB_1 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
CA_{N-1}\dots A_1B_0 & CA_{N-1}\dots A_2B_1 & \dots & CB_{N-1}
\end{bmatrix}
$$
