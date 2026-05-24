# Algorithms and Implementation

This document details the mathematical formulation and implementation of the control strategies used in this project, specifically the Reinforced Iterative Learning Control (RILC) framework and the Combined Iterative Learning Control (C-ILC) baseline.

## 1. System Dynamics (Series Elastic Actuators)

The system is modeled as a 2n-DoFs under-actuated robot with Series Elastic Actuators (SEA). The dynamics are given by [35]:

$$
\begin{array}{l}
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) + K(q-\theta) = 0 \\
B\ddot{\theta} + D\dot{\theta} - K(q-\theta) = u
\end{array}
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
\end{cases}
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
\bar{u}_{j+1}(t) = \bar{u}_j(t) + \mathfrak{L}_j(t) \chi_i e_j^{(i)}(t)
$$

Where:

- $e_j^{(i)}(t) = y_d^{(i)}(t) - y_j^{(i)}(t)$ is the $i$-th derivative of the error.
- $\mathfrak{L}_j(t)$ is the iteration and time-varying learning gain.
- $\chi_i \succ 0$ are tunable control gains, with Einstein summation $\chi_i e^{(i)}_j \doteq \sum_{i=0}^r \chi_i e^{(i)}_j$.

This update guarantees asymptotic convergence of the tracking error under standard assumptions (Lipschitz continuity, strictly proper system, etc.), provided $\|I_n - \mathfrak{L}_j \chi_r E(x_j)\| \leq \eta < 1$.

## 3. Norm Optimal ILC (NOILC)

Minimizes a quadratic cost function balancing error and input change:

$$
J_j = \sum_{t=0}^{t_{\textnormal{f}}} \bigl[ e_j(t)^\top Q e_j(t) + \delta\bar{u}^\textnormal{N}_j(t)^\top R \,\delta\bar{u}^\textnormal{N}_j(t) \bigr],
$$

where $\delta\bar{u}^\textnormal{N}_j(t) = \bar{u}^\textnormal{N}_j(t) - \bar{u}^\textnormal{N}_{j-1}(t)$.

Lifted-domain update law:

$$
\bar{u}^\textnormal{N}_{j+1} = \bar{u}^\textnormal{N}_j + \bigl[(G^\top Q G + R)^{-1} G^\top Q\bigr] e_j,
$$

where $G$ denotes the lifted system matrix obtained by linearizing the robot dynamics along the desired trajectory:

$$
G = \begin{bmatrix}
CB_0 & 0 & \dots & 0 \\
CA_1B_0 & CB_1 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
CA_{N-1}\dots A_1B_0 & CA_{N-1}\dots A_2B_1 & \dots & CB_{N-1}
\end{bmatrix}
$$

The non-linear dynamics (Equation 2) are linearized at each time step along the reference trajectory $y_d(t)$ (and corresponding state/input) to obtain time-varying matrices $A_k, B_k$, which are then assembled into $G$.

## 4. Combined ILC (C-ILC)

C-ILC (Tsurumoto et al., IFAC 2023) decomposes the feedforward control into two complementary components: a **basis function term** (task-flexible, transferable across trajectories) and a **residual ILC term** (high-performance, trajectory-specific). The total control is

$$
u_j^\textnormal{C} = u_j^w + \bar{u}_j,
$$

where $u_j^w = \Phi_0\, w_j$ is the basis term and $\bar{u}_j$ follows the standalone ILC update.

### Basis Functions

The basis matrix $\Phi_0 \in \mathbb{R}^{n N \times n_b}$ is built from the sampled trajectory evolution using physics-inspired features. For each joint the basis vector at a given time sample is

$$
\varphi(y) = [\ddot{y}_1,\; \ddot{y}_2,\; \dot{y}_1,\; \dot{y}_2,\;
              \sin(y_1),\; \sin(y_2),\; 1]^\top,
$$

yielding $n_b = 7n$ parameters. These features capture acceleration coupling, velocity damping, gravitational signatures, and a constant offset.

### Combined Update Laws

The two components are updated jointly at each iteration to ensure coherent evolution of the total feedforward signal:

**Basis weight update** (virtual error feedback):

$$
w_{j+1} = w_j + \Phi_0^\dagger (e_j + \bar{u}_j),
$$

where $\Phi_0^\dagger = (\Phi_0^\top\Phi_0 + \lambda I)^{-1}\Phi_0^\top$ is the regularized pseudoinverse, and $(e_j + \bar{u}_j)$ is the **virtual error**. Including $\bar{u}_j$ in the virtual error prevents the basis update from ignoring the residual ILC's existing compensation.

**Residual ILC update** (with basis correction):

$$
\bar{u}_{j+1} = \bar{u}_j + \chi_i e_j^{(i)} + (u_j^w - u_{j+1}^w),
$$

where the correction term $(u_j^w - u_{j+1}^w)$ compensates for the change in the basis component. This term is the key innovation of C-ILC: it prevents the ILC from "fighting" the basis function update. When $w$ changes, the residual ILC is automatically adjusted so that the **total** feedforward $u_{j+1}^\textnormal{C}$ evolves smoothly.

## 5. Reinforcement Learning (RL) — PPO

We use Proximal Policy Optimization (PPO) to train a stochastic policy $\tilde{u}_\nu(\xi_j)$ parameterized by $\nu$. The policy is updated to maximize a reward function (minimize cost) using the gradient update:

$$
\nu \leftarrow \nu - \nabla_\nu {\bar{J}}(\cdot, \nu)
$$

The RL agent observes the state $\xi_j(t)$ which includes the desired trajectory and past ILC iterations.

## 6. Reinforced Iterative Learning Control (RILC)

The RILC framework synergistically combines ILC and RL. The total control input applied to the system at iteration $j$ is:

$$
u_j(t) = \bar{u}_j(t) + \tilde{u}_\nu(\xi_j(t)) + u_{mb}(t) + u_{fb}(t)
$$

Where:

- $\bar{u}_j(t)$: ILC feedforward term (learns from repetition).
- $\tilde{u}_\nu(\xi_j(t))$: RL policy (learns to generalize and compensate for non-repetitive disturbances).
- $u_{mb}(t)$: Model-based term (e.g., gravity compensation).
- $u_{fb}(t)$: Feedback term (e.g., PD controller) for stabilization.

### Observation $\xi_j$

The observation $\xi_j \in \mathbb{R}^{n_\xi}$ is the measurable process variable that the RL policy uses to compute its action. It is defined as

$$
\xi_j(t) = \phi(y_d, y_{j-I}, \bar{u}_{j-I}),
$$

where $I \subseteq \{0, \dots, j\}$ is a set of past iteration indices, and $z_{j-I} = \{z_{j-i}\}, \forall i \in I$. In practice, $\xi_j$ includes:

- The desired trajectory $y_d(t)$ and its derivatives.
- The current measured output $y_j(t)$.
- Past tracking errors $e_{j-i}(t)$ for $i \in I$.
- Past ILC feedforward inputs $\bar{u}_{j-i}(t)$ for $i \in I$.
- The previous RL action $\tilde{u}_\nu(\xi_{j-1})$.

This composition enables the RL policy to learn a correction that accounts for both the current tracking state and the historical ILC behavior, facilitating cross-trajectory generalization.
