import os
import sys
import torch
import mujoco
from matplotlib import pyplot as plt
import numpy as np

# --- Plotting Style ---
def setup_plotting(font_size=18):
    """Setup consistent plotting style for all benchmarks"""
    plt.rc('font', family='serif', serif='Times')
    plt.rcParams["text.usetex"] = True
    plt.rc('text.latex', preamble=r'\usepackage[utf8]{inputenc} \usepackage{amsmath} \usepackage{amsfonts}')
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('grid', linestyle='-.', alpha=0.5)
    plt.rc('axes', grid=True)
    plt.rcParams['figure.constrained_layout.use'] = True

# --- Configuration Constants ---
# Assuming this file is in scripts/, so .. is project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLASSES_PATH = os.path.join(PROJECT_ROOT, 'classes')
URDF_PATH = os.path.join(CLASSES_PATH, 'robots/robot_models/softleg_urdf/urdf/leg_constrained.urdf')
MJC_PATH = os.path.join(CLASSES_PATH, 'robots/robot_models/softleg_urdf/mjc/scene_test.xml')

# RL Model Path
parent_str = "model"
dat_rilc_str = "rilc_16"
step_rilc_str = "best_model/best_model.zip"
MODEL_RILC_STR = os.path.join(PROJECT_ROOT, parent_str, dat_rilc_str, step_rilc_str)

dat_rl_str = "rl_classic_64"
step_rl_str = "best_model/best_model.zip"
MODEL_RL_STR = os.path.join(PROJECT_ROOT, parent_str, dat_rl_str, step_rl_str)


# --- Helper Functions ---
def minjerk(qi:torch.Tensor,qf:torch.Tensor,duration:float,t:float) -> list[torch.Tensor,torch.Tensor,torch.Tensor]:
    delta_q = qi-qf
    q_new   = qi + delta_q * (15*(t/duration)**4 - 6*(t/duration)**5 - 10*(t/duration)**3)
    dq_new  = delta_q * (60*(t**3)/(duration**4) - 30*((t**4)/(duration**5)) - 30*(t**2)/(duration**3))
    ddq_new = delta_q * (180*(t**2)/(duration**4) - 120*((t**3)/(duration**5)) - 60*(t/(duration**3)))
    return q_new, dq_new, ddq_new

def angle_normalize(x:torch.Tensor) -> torch.Tensor:
    sx = torch.sin(x)
    cx = torch.cos(x)
    x = torch.atan2(sx,cx)
    return x

def resample_u(u_old:torch.Tensor, u_new:torch.Tensor, num_step:int) -> torch.Tensor:
    du_step = (u_new-u_old)/num_step
    return du_step

# --- Linearization for NOILC ---
def get_linearized_matrices(robot, q, dq, u, dt, damping=True):
    nq = robot._dim_q
    nu = robot._dim_u
    nx = 2 * nq
    eps = 1e-2
    A = torch.zeros(nx, nx)
    B = torch.zeros(nx, nu)
    
    q_nom = q.clone()
    dq_nom = dq.clone()
    q_bak = robot.q.clone()
    dq_bak = robot.dq.clone()
    
    def get_next_state(q_in, dq_in, u_in):
        robot.setState(q=q_in, dq=dq_in)
        x_next = robot.getNewState(dt=dt, action=u_in, damp_fl=damping)
        return torch.cat([x_next[0], x_next[1]], dim=0)

    x_nom_next = get_next_state(q, dq, u)
    
    for i in range(nq):
        q_p = q.clone(); q_p[i] += eps
        x_p_next = get_next_state(q_p, dq, u)
        A[:, i] = (x_p_next - x_nom_next).flatten() / eps
    for i in range(nq):
        dq_p = dq.clone(); dq_p[i] += eps
        x_p_next = get_next_state(q, dq_p, u)
        A[:, nq + i] = (x_p_next - x_nom_next).flatten() / eps
    for i in range(nu):
        u_p = u.clone(); u_p[i] += eps
        x_p_next = get_next_state(q, dq, u_p)
        B[:, i] = (x_p_next - x_nom_next).flatten() / eps
        
    robot.setState(q=q_bak, dq=dq_bak)
    return A, B

def construct_lifted_model_linearized_nonlinear(robot, q_traj, dq_traj, u_traj, dt, samples, dimU, use_analytical=True):
    print(f"Linearizing dynamics along trajectory (Model-Based, Analytical={use_analytical})...")
    As, Bs = [], []
    C = torch.cat([torch.eye(dimU), torch.zeros(dimU, dimU)], dim=1)
    
    for k in range(samples):
        q_k = q_traj[:, k].view(-1,1)
        dq_k = dq_traj[:, k].view(-1,1)
        u_k = u_traj[:, k].view(-1,1)
        
        if use_analytical:
            # Get Continuous A, B from Pinocchio
            Ac, Bc = robot.getAnalyticalLinearizedDynamics(q=q_k, dq=dq_k, u=u_k, damp_fl=True)
            # Discretize (Euler for simplicity, matching finite diff order roughly)
            # x_{k+1} = x_k + (Ac x_k + Bc u_k) * dt
            # x_{k+1} = (I + Ac*dt) x_k + (Bc*dt) u_k
            Ak = torch.eye(2*dimU) + Ac * dt
            Bk = Bc * dt
        else:
            # Finite Difference
            Ak, Bk = get_linearized_matrices(robot, q_k, dq_k, u_k, dt)
            
        As.append(Ak); Bs.append(Bk)

    G = torch.zeros(dimU * samples, dimU * samples)
    print("Building Lifted G Matrix (Model-Based)...")
    for c_t in range(samples):
        Bj = Bs[c_t]
        for input_idx in range(dimU):
            x_curr = Bj[:, input_idx].view(-1,1)
            y_curr = C @ x_curr
            for r_t in range(c_t + 1, samples):
                 row_base = r_t * dimU
                 col_idx = c_t * dimU + input_idx
                 G[row_base:row_base+dimU, col_idx] = y_curr.flatten()
                 if r_t < samples - 1:
                     x_curr = As[r_t] @ x_curr
                     y_curr = C @ x_curr
    return G
